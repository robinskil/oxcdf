#![cfg(feature = "write")]
//! Random files from the writer read back equal to a model, also after sink failures.

use std::sync::Arc;

use oxcdf_hdf5::message::{Attribute, ByteOrder, DatatypeClass};
use oxcdf_hdf5::read::{read_hyperslab, resolve_vlen_strings, resolve_vlen_strings_of};
use oxcdf_hdf5::write::{
    ByteSink, DatasetId, DatasetSpec, Filter, GroupId, Hdf5Writer, MemorySink, ObjectId,
    WriteOptions,
};
use oxcdf_hdf5::{DType, Error, GroupIndex, Hdf5File, Hyperslab, MemorySource, RawData, Result};

/// A xorshift generator, so that each seed gives the same file on each run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value below `n`.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const DTYPES: [DType; 10] = [
    DType::Int(1),
    DType::Int(2),
    DType::Int(4),
    DType::Int(8),
    DType::Uint(1),
    DType::Uint(2),
    DType::Uint(4),
    DType::Uint(8),
    DType::Float(4),
    DType::Float(8),
];

/// The bytes of the small value `v` as one element of `dtype` in `order`.
fn element(v: u8, dtype: &DType, order: ByteOrder) -> Vec<u8> {
    let mut bytes = match dtype {
        DType::Int(1) | DType::Uint(1) => vec![v],
        DType::Int(2) | DType::Uint(2) => u16::from(v).to_le_bytes().to_vec(),
        DType::Int(4) | DType::Uint(4) => u32::from(v).to_le_bytes().to_vec(),
        DType::Int(8) | DType::Uint(8) => u64::from(v).to_le_bytes().to_vec(),
        DType::Float(4) => f32::from(v).to_le_bytes().to_vec(),
        _ => f64::from(v).to_le_bytes().to_vec(),
    };
    if order == ByteOrder::Big {
        bytes.reverse();
    }
    bytes
}

/// What the model expects for one dataset.
struct ModelDataset {
    path: Vec<String>,
    dtype: DType,
    order: ByteOrder,
    shape: Vec<u64>,
    /// The fill value. It is 0 for a dataset with no fill value.
    fill: u8,
    /// One small value for each element, in row-major order.
    values: Vec<u8>,
    /// Each element of a string dataset. Then the four fields above have no use.
    strings: Option<Vec<String>>,
    /// The chunk shape of a chunked dataset.
    chunks: Option<Vec<u64>>,
    /// The largest size of each axis of a chunked dataset. `None` is unlimited.
    max: Vec<Option<u64>>,
    /// The filters of a chunked dataset.
    filters: Vec<Filter>,
}

/// Unwrap a write result. Return false for a write that a dataset with filters rejects.
///
/// A chunk that is in the file with its filters takes no write to a part of
/// it. The writer then returns an error and changes nothing.
fn wrote(result: Result<()>, d: &ModelDataset) -> bool {
    match result {
        Err(Error::Unsupported(_)) if !d.filters.is_empty() => false,
        result => {
            result.unwrap();
            true
        }
    }
}

impl ModelDataset {
    /// Grow one axis in the writer and in the model, when the largest shape has room.
    fn grow<S: ByteSink>(&mut self, w: &mut Hdf5Writer<S>, rng: &mut Rng, id: DatasetId) {
        if self.chunks.is_none() || self.shape.is_empty() {
            return;
        }
        let axis = rng.below(self.shape.len() as u64) as usize;
        let mut shape = self.shape.clone();
        shape[axis] += 1 + rng.below(3);
        if self.max[axis].is_some_and(|max| shape[axis] > max) {
            assert!(w.resize(id, &shape).is_err());
            return;
        }
        w.resize(id, &shape).unwrap();
        // Each old element keeps its position on each axis.
        let old = positions(&shape, &vec![0; shape.len()], &self.shape);
        let elements: u64 = shape.iter().product();
        let mut values = vec![self.fill; elements as usize];
        for (&to, &v) in old.iter().zip(&self.values) {
            values[to] = v;
        }
        self.values = values;
        if let Some(strings) = &mut self.strings {
            let mut grown = vec![String::new(); elements as usize];
            for (&to, v) in old.iter().zip(strings.drain(..)) {
                grown[to] = v;
            }
            *strings = grown;
        }
        self.shape = shape;
    }
}

/// What the model expects for one group.
struct ModelGroup {
    path: Vec<String>,
    groups: Vec<String>,
}

/// The value of one attribute.
#[derive(Debug, Clone, PartialEq)]
enum Value {
    /// The data bytes of a fixed-size datatype.
    Bytes(Vec<u8>),
    /// Variable-length strings.
    Strings(Vec<String>),
}

/// The attributes of one object in creation order: name and value.
type ModelAttributes = Vec<(String, Value)>;

/// A random string. Some are empty, some hold a character above ASCII, a few are long.
fn random_string(rng: &mut Rng) -> String {
    match rng.below(60) {
        // Longer than one heap collection.
        0 => "y".repeat(70_000),
        1..=9 => String::new(),
        10..=19 => format!("zee\u{00eb}n {}", rng.below(1000)),
        _ => format!("v{}", rng.below(100_000)),
    }
}

/// The position of each element of the selection in the dataset, in row-major order.
fn positions(shape: &[u64], start: &[u64], count: &[u64]) -> Vec<usize> {
    let total: u64 = count.iter().product();
    let mut index = vec![0u64; count.len()];
    let mut out = Vec::new();
    for _ in 0..total {
        let mut flat = 0u64;
        for axis in 0..index.len() {
            flat = flat * shape[axis] + start[axis] + index[axis];
        }
        out.push(flat as usize);
        for axis in (0..index.len()).rev() {
            index[axis] += 1;
            if index[axis] < count[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    out
}

/// A random selection in `shape`, with one element or more on each axis.
fn random_selection(rng: &mut Rng, shape: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let start: Vec<u64> = shape.iter().map(|&n| rng.below(n)).collect();
    let count = shape
        .iter()
        .zip(&start)
        .map(|(&n, &s)| 1 + rng.below(n - s))
        .collect();
    (start, count)
}

#[derive(Default)]
struct Model {
    groups: Vec<ModelGroup>,
    datasets: Vec<ModelDataset>,
    /// Attributes for each group, then for each dataset.
    group_attributes: Vec<ModelAttributes>,
    dataset_attributes: Vec<ModelAttributes>,
}

/// Put a random number of attributes on `object`, with one replaced value.
fn put_attributes<S: ByteSink>(
    w: &mut Hdf5Writer<S>,
    rng: &mut Rng,
    object: ObjectId,
) -> ModelAttributes {
    let mut model = ModelAttributes::new();
    // Above eight, the HDF5 library default for compact attributes.
    let count = rng.below(12);
    for i in 0..count {
        let name = format!("attr_{i}");
        match rng.below(3) {
            0 => {
                let values: Vec<i32> = (0..=rng.below(3)).map(|k| (k + i) as i32).collect();
                w.put_numeric_attribute(object, &name, &values).unwrap();
                let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                model.push((name, Value::Bytes(data)));
            }
            1 => {
                let text = format!("text {}", rng.below(1000));
                w.put_text_attribute(object, &name, &text).unwrap();
                model.push((name, Value::Bytes(text.into_bytes())));
            }
            _ => {
                let values: Vec<String> = (0..=rng.below(3)).map(|_| random_string(rng)).collect();
                w.put_string_attribute(object, &name, &values).unwrap();
                model.push((name, Value::Strings(values)));
            }
        }
    }
    if count > 0 {
        let position = rng.below(count) as usize;
        w.put_numeric_attribute(object, &model[position].0, &[7.5f64])
            .unwrap();
        model[position].1 = Value::Bytes(7.5f64.to_le_bytes().to_vec());
    }
    model
}

/// Write a random selection to `id` and to the model.
fn write_random<S: ByteSink>(
    w: &mut Hdf5Writer<S>,
    rng: &mut Rng,
    id: DatasetId,
    d: &mut ModelDataset,
) {
    let (start, count) = random_selection(rng, &d.shape);
    let slab = Hyperslab::new(start.clone(), count.clone(), &d.shape).unwrap();
    let total: u64 = count.iter().product();
    if d.strings.is_some() {
        let values: Vec<String> = (0..total).map(|_| random_string(rng)).collect();
        if !wrote(w.write_strings(id, &slab, &values), d) {
            return;
        }
        let Some(strings) = &mut d.strings else {
            return;
        };
        for (flat, value) in positions(&d.shape, &start, &count).into_iter().zip(values) {
            strings[flat] = value;
        }
        return;
    }
    let values: Vec<u8> = (0..total).map(|_| rng.below(100) as u8).collect();

    let result = if d.dtype == DType::Int(4) {
        let typed: Vec<i32> = values.iter().map(|&v| i32::from(v)).collect();
        w.write(id, &slab, &typed)
    } else {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| element(v, &d.dtype, d.order))
            .collect();
        w.write_raw(id, &slab, &bytes)
    };
    if !wrote(result, d) {
        return;
    }

    for (flat, v) in positions(&d.shape, &start, &count).into_iter().zip(values) {
        d.values[flat] = v;
    }
}

/// Build a random file for `seed` and return its bytes with the model.
fn build(seed: u64) -> (Vec<u8>, Model) {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
    let mut model = Model::default();
    let mut group_ids: Vec<GroupId> = vec![w.root()];
    model.groups.push(ModelGroup {
        path: Vec::new(),
        groups: Vec::new(),
    });

    // Above eight in one group, the HDF5 library default for compact links.
    for i in 0..rng.below(20) {
        let parent = rng.below(group_ids.len() as u64) as usize;
        let name = format!("obj_{i}");
        let mut path = model.groups[parent].path.clone();
        path.push(name.clone());
        if rng.below(3) == 0 {
            group_ids.push(w.add_group(group_ids[parent], &name).unwrap());
            model.groups[parent].groups.push(name);
            model.groups.push(ModelGroup {
                path,
                groups: Vec::new(),
            });
            continue;
        }
        let dtype = DTYPES[rng.below(10) as usize].clone();
        let order = if rng.below(2) == 0 {
            ByteOrder::Little
        } else {
            ByteOrder::Big
        };
        let shape: Vec<u64> = (0..rng.below(4)).map(|_| rng.below(6)).collect();
        let strings = rng.below(5) == 0;
        let mut spec = DatasetSpec::numeric(dtype.clone(), order, &shape).unwrap();
        let fill = match rng.below(3) {
            _ if strings => {
                spec = DatasetSpec::strings(&shape);
                0
            }
            0 => 0,
            1 => {
                spec = spec.with_fill_value(element(0, &dtype, order));
                0
            }
            _ => {
                // At most 127, so that the value fits each type.
                let fill = 100 + rng.below(28) as u8;
                spec = spec.with_fill_value(element(fill, &dtype, order));
                fill
            }
        };
        // One dataset in three is chunked, with unlimited and limited axes.
        let mut chunks = None;
        let mut max = Vec::new();
        let mut filters = Vec::new();
        if !shape.is_empty() && rng.below(3) == 0 {
            for &dim in &shape {
                max.push((rng.below(2) == 0).then(|| dim + 1 + rng.below(3)));
            }
            let dims = max.iter().map(|max| 1 + rng.below(max.unwrap_or(3).min(3)));
            let dims: Vec<u64> = dims.collect();
            spec = spec.chunked(&dims).with_max_shape(&max);
            chunks = Some(dims);
            // One chunked dataset in two has filters.
            filters = match rng.below(8) {
                0 => vec![Filter::Shuffle, Filter::Deflate(rng.below(10) as u8)],
                1 => vec![Filter::Fletcher32, Filter::Shuffle, Filter::Deflate(4)],
                2 => vec![Filter::Shuffle, Filter::Zstd(1 + rng.below(9) as u8)],
                3 => vec![Filter::Deflate(1), Filter::Fletcher32],
                _ => Vec::new(),
            };
            spec = spec.with_filters(&filters);
        }
        let id = w.add_dataset(group_ids[parent], &name, spec).unwrap();
        let elements: u64 = shape.iter().product();
        let mut dataset = ModelDataset {
            path,
            dtype,
            order,
            shape,
            fill,
            values: vec![fill; elements as usize],
            strings: strings.then(|| vec![String::new(); elements as usize]),
            chunks,
            max,
            filters,
        };
        for _ in 0..rng.below(4) {
            if rng.below(2) == 0 {
                dataset.grow(&mut w, &mut rng, id);
            }
            if rng.below(4) == 0 {
                w.flush_chunks(id).unwrap();
            }
            if !dataset.shape.contains(&0) {
                write_random(&mut w, &mut rng, id, &mut dataset);
            }
        }
        model
            .dataset_attributes
            .push(put_attributes(&mut w, &mut rng, id.into()));
        model.datasets.push(dataset);
    }
    for &id in &group_ids {
        model
            .group_attributes
            .push(put_attributes(&mut w, &mut rng, id.into()));
    }
    (w.close().unwrap().into_inner(), model)
}

fn group<'a>(file: &'a Hdf5File, path: &[String]) -> &'a GroupIndex {
    path.iter()
        .fold(file.root(), |g, name| g.group(name).expect("the group"))
}

/// The attributes that the reader gives, with each string value read from the heap.
fn attributes_of(file: &Hdf5File, attributes: &[Attribute]) -> ModelAttributes {
    attributes
        .iter()
        .map(|a| {
            let value = match a.datatype.class {
                DatatypeClass::VariableLength { .. } => {
                    let raw = RawData {
                        bytes: a.data.clone(),
                        element_size: a.datatype.size as usize,
                        shape: a.dataspace.dims.clone(),
                    };
                    Value::Strings(resolve_vlen_strings_of(file.ctx(), &a.name, &raw).unwrap())
                }
                _ => Value::Bytes(a.data.clone()),
            };
            (a.name.clone(), value)
        })
        .collect()
}

/// Check that the reader gives each object of the model from `bytes`.
fn check(bytes: Vec<u8>, model: &Model, case: &str) {
    let len = bytes.len() as u64;
    let file = Hdf5File::from_source(Arc::new(MemorySource::new(bytes))).expect(case);
    assert_eq!(file.superblock().end_of_file_address, len, "{case}");

    for (g, attributes) in model.groups.iter().zip(&model.group_attributes) {
        let got = group(&file, &g.path);
        let names: Vec<_> = got.groups.iter().map(|c| c.name.clone()).collect();
        assert_eq!(names, g.groups, "{case} {:?}", g.path);
        let got = attributes_of(&file, &got.attributes);
        assert_eq!(&got, attributes, "{case} {:?}", g.path);
    }
    for (d, attributes) in model.datasets.iter().zip(&model.dataset_attributes) {
        let path = format!("/{}", d.path.join("/"));
        let got = file.dataset(&path).expect(&path);
        assert_eq!(got.shape, d.shape, "{case} {path}");
        assert_eq!(
            attributes_of(&file, &got.attributes),
            *attributes,
            "{case} {path}"
        );
        got.prepare(file.ctx()).unwrap();
        let raw = read_hyperslab(file.ctx(), got, &Hyperslab::all(&got.shape)).unwrap();
        if let Some(strings) = &d.strings {
            let got = resolve_vlen_strings(file.ctx(), got, &raw).unwrap();
            assert_eq!(&got, strings, "{case} {path}");
            continue;
        }
        assert_eq!(DType::of(&got.datatype), d.dtype, "{case} {path}");
        assert_eq!(got.datatype.byte_order(), Some(d.order), "{case} {path}");
        // The reader returns each element in the host byte order.
        let host = if cfg!(target_endian = "big") {
            ByteOrder::Big
        } else {
            ByteOrder::Little
        };
        let want: Vec<u8> = d
            .values
            .iter()
            .flat_map(|&v| element(v, &d.dtype, host))
            .collect();
        assert_eq!(raw.bytes, want, "{case} {path}");
    }
}

#[test]
fn random_files_read_back_equal_to_the_model() {
    let (mut datasets, mut strings, mut chunked, mut filtered) = (0, 0, 0, 0);
    for seed in 1..=300 {
        let (bytes, model) = build(seed);
        assert_eq!(bytes, build(seed).0, "seed {seed} gives two outputs");
        datasets += model.datasets.len();
        strings += model
            .datasets
            .iter()
            .filter(|d| d.strings.is_some())
            .count();
        chunked += model.datasets.iter().filter(|d| d.chunks.is_some()).count();
        filtered += model
            .datasets
            .iter()
            .filter(|d| !d.filters.is_empty())
            .count();
        check(bytes, &model, &format!("seed {seed}"));
    }
    assert!(datasets > 1000, "{datasets}");
    assert!(strings > 100, "{strings}");
    assert!(chunked > 100, "{chunked}");
    assert!(filtered > 50, "{filtered}");
}

/// A sink that fails one time, at call number `fail_at`.
#[derive(Debug)]
struct FailOnce {
    inner: MemorySink,
    calls: usize,
    fail_at: usize,
}

impl FailOnce {
    fn step(&mut self) -> Result<()> {
        self.calls += 1;
        if self.calls - 1 == self.fail_at {
            return Err(Error::Io(std::io::ErrorKind::Other.into()));
        }
        Ok(())
    }
}

impl ByteSink for FailOnce {
    fn write_all_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        self.step()?;
        self.inner.write_all_at(offset, data)
    }

    fn set_len(&mut self, len: u64) -> Result<()> {
        self.step()?;
        self.inner.set_len(len)
    }

    fn sync(&mut self) -> Result<()> {
        self.step()?;
        self.inner.sync()
    }
}

/// Write one fixed file. A failed write gets one more try.
///
/// Return the bytes and the number of failed writes, or `None` when close fails.
fn fixed_file(fail_at: usize) -> (Option<Vec<u8>>, usize) {
    let sink = FailOnce {
        inner: MemorySink::new(),
        calls: 0,
        fail_at,
    };
    let mut w = Hdf5Writer::create(sink, WriteOptions::default()).unwrap();
    let root = w.root();
    let g = w.add_group(root, "g").unwrap();
    w.put_text_attribute(g, "title", "fixed").unwrap();
    let mut failed = 0;
    let filled = DatasetSpec::numeric(DType::Int(2), ByteOrder::Big, &[4, 3])
        .unwrap()
        .with_fill_value(9i16.to_be_bytes().to_vec());
    let plain = DatasetSpec::numeric(DType::Float(8), ByteOrder::Little, &[5]).unwrap();
    let a = w.add_dataset(g, "filled", filled).unwrap();
    let b = w.add_dataset(root, "plain", plain).unwrap();
    let rows = |row| Hyperslab::new(vec![row, 1], vec![1, 2], &[4, 3]).unwrap();
    for row in [1u64, 3] {
        let values = [row as i16, 20];
        if w.write(a, &rows(row), &values).is_err() {
            failed += 1;
            w.write(a, &rows(row), &values).unwrap();
        }
    }
    let part = Hyperslab::new(vec![1], vec![3], &[5]).unwrap();
    if w.write(b, &part, &[1.5f64, 2.5, 3.5]).is_err() {
        failed += 1;
        w.write(b, &part, &[1.5f64, 2.5, 3.5]).unwrap();
    }
    let s = w
        .add_dataset(root, "strings", DatasetSpec::strings(&[4]))
        .unwrap();
    // The long strings fill a heap collection, so a put writes it to the sink.
    for (at, text) in fixed_strings().iter().enumerate() {
        let one = Hyperslab::new(vec![at as u64], vec![1], &[4]).unwrap();
        if w.write_strings(s, &one, &[text]).is_err() {
            failed += 1;
            w.write_strings(s, &one, &[text]).unwrap();
        }
    }
    if w.put_string_attribute(s, "labels", &fixed_strings())
        .is_err()
    {
        failed += 1;
        w.put_string_attribute(s, "labels", &fixed_strings())
            .unwrap();
    }
    (w.close().ok().map(|s| s.inner.into_inner()), failed)
}

/// The strings of the fixed file.
fn fixed_strings() -> [String; 4] {
    [
        "a".repeat(40_000),
        "b".repeat(40_000),
        "c".into(),
        "d".repeat(70_000),
    ]
}

/// Check the content of the fixed file through the reader.
fn check_fixed_file(bytes: Vec<u8>, case: &str) {
    let file = Hdf5File::from_source(Arc::new(MemorySource::new(bytes))).expect(case);
    let raw = |path: &str| {
        let d = file.dataset(path).unwrap();
        d.prepare(file.ctx()).unwrap();
        (
            d,
            read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape)).unwrap(),
        )
    };
    let (d, filled) = raw("/g/filled");
    assert_eq!(
        filled.get::<i16>(d).unwrap(),
        [9, 9, 9, 9, 1, 20, 9, 9, 9, 9, 3, 20],
        "{case}"
    );
    let (d, plain) = raw("/plain");
    assert_eq!(
        plain.get::<f64>(d).unwrap(),
        [0.0, 1.5, 2.5, 3.5, 0.0],
        "{case}"
    );
    let (d, strings) = raw("/strings");
    let strings = resolve_vlen_strings(file.ctx(), d, &strings).unwrap();
    assert_eq!(strings, fixed_strings(), "{case}");
    let labels = attributes_of(&file, &d.attributes);
    let want = Value::Strings(fixed_strings().to_vec());
    assert_eq!(labels, [("labels".to_owned(), want)], "{case}");
}

#[test]
fn a_sink_failure_at_each_call_leaves_a_correct_file_or_an_error() {
    let (reference, _) = fixed_file(usize::MAX);
    check_fixed_file(reference.unwrap(), "no failure");

    let (mut retried, mut lost) = (0, 0);
    for fail_at in 0..64 {
        let (bytes, failed) = fixed_file(fail_at);
        retried += failed;
        match bytes {
            Some(bytes) => check_fixed_file(bytes, &format!("failure at call {fail_at}")),
            None => lost += 1,
        }
    }
    // Writes of numbers, writes of strings and heap collections each fail in some run.
    assert!(retried >= 8, "{retried}");
    assert!(lost >= 4, "{lost}");
}

/// The model of one file as JSON for `scripts/check_writer.py`.
///
/// Each name in the model is plain ASCII, so no character needs an escape.
fn model_json(model: &Model) -> String {
    let list = |items: Vec<String>| format!("[{}]", items.join(","));
    let quoted = |names: &[String]| list(names.iter().map(|n| format!("{n:?}")).collect());
    let attributes = |attributes: &ModelAttributes| {
        list(
            attributes
                .iter()
                .map(|(name, value)| match value {
                    Value::Bytes(data) => {
                        let hex: String = data.iter().map(|b| format!("{b:02x}")).collect();
                        format!("[{name:?},\"{hex}\"]")
                    }
                    Value::Strings(strings) => format!("[{name:?},{}]", quoted(strings)),
                })
                .collect(),
        )
    };
    let groups = model.groups.iter().zip(&model.group_attributes);
    let groups = groups.map(|(g, a)| {
        format!(
            "{{\"path\":{:?},\"groups\":{},\"attrs\":{}}}",
            format!("/{}", g.path.join("/")),
            quoted(&g.groups),
            attributes(a)
        )
    });
    let datasets = model.datasets.iter().zip(&model.dataset_attributes);
    let datasets = datasets.map(|(d, a)| {
        let (kind, size) = match d.dtype {
            DType::Int(n) => ('i', n),
            DType::Uint(n) => ('u', n),
            DType::Float(n) => ('f', n),
            _ => ('?', 0),
        };
        let order = match (size, d.order) {
            (1, _) => '|',
            (_, ByteOrder::Little) => '<',
            (_, ByteOrder::Big) => '>',
        };
        let (dtype, values) = match &d.strings {
            Some(strings) => ("str".to_owned(), quoted(strings)),
            None => (format!("{order}{kind}{size}"), format!("{:?}", d.values)),
        };
        let filters = d.filters.iter().map(|filter| match filter {
            Filter::Shuffle => "\"shuffle\"",
            Filter::Deflate(_) => "\"deflate\"",
            Filter::Zstd(_) => "\"zstd\"",
            _ => "\"fletcher32\"",
        });
        let filters = list(filters.map(str::to_owned).collect());
        let (chunks, max) = match &d.chunks {
            Some(chunks) => {
                let max = d.max.iter().map(|max| match max {
                    Some(max) => max.to_string(),
                    None => "null".to_owned(),
                });
                (format!("{chunks:?}"), list(max.collect()))
            }
            None => ("null".to_owned(), "null".to_owned()),
        };
        format!(
            "{{\"path\":{:?},\"dtype\":{dtype:?},\"shape\":{:?},\"chunks\":{chunks},\"maxshape\":{max},\"filters\":{},\"fill\":{},\"values\":{values},\"attrs\":{}}}",
            format!("/{}", d.path.join("/")),
            d.shape,
            filters,
            d.fill,
            attributes(a)
        )
    });
    format!(
        "{{\"groups\":{},\"datasets\":{}}}",
        list(groups.collect()),
        list(datasets.collect())
    )
}

/// A file with fixed and variable-length string datasets and a known content.
fn strings_file() -> Vec<u8> {
    use oxcdf_hdf5::message::{CharSet, Datatype, DatatypeClass, StringPad};
    let string = |size, pad| Datatype {
        version: 1,
        size,
        class: DatatypeClass::String {
            pad,
            charset: CharSet::Ascii,
        },
    };
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
    let root = w.root();
    let spec = DatasetSpec::new(string(5, StringPad::NullPad), &[3]);
    let names = w.add_dataset(root, "names", spec).unwrap();
    w.write_raw(names, &Hyperslab::all(&[3]), b"alphabe\0\0\0gamma")
        .unwrap();
    let spec = DatasetSpec::new(string(1, StringPad::NullTerminate), &[2, 4]);
    let chars = w.add_dataset(root, "chars", spec).unwrap();
    let row = Hyperslab::new(vec![1, 0], vec![1, 4], &[2, 4]).unwrap();
    w.write_raw(chars, &row, b"wxyz").unwrap();
    // No fill value: netcdf-c 4.9.3 fails on a fixed string dataset that has one.
    let spec = DatasetSpec::new(string(4, StringPad::SpacePad), &[2]);
    let padded = w.add_dataset(root, "padded", spec).unwrap();
    w.write_raw(padded, &Hyperslab::all(&[2]), b"wxyzab  ")
        .unwrap();
    w.put_text_attribute(names, "long_name", "station names")
        .unwrap();

    let vlen = w
        .add_dataset(root, "vlen", DatasetSpec::strings(&[2, 3]))
        .unwrap();
    let row = Hyperslab::new(vec![0, 0], vec![1, 3], &[2, 3]).unwrap();
    w.write_strings(vlen, &row, &["alpha", "", "zee\u{00eb}n"])
        .unwrap();
    w.put_string_attribute(vlen, "one", &["solo"]).unwrap();
    let many = w
        .add_dataset(root, "many", DatasetSpec::strings(&[3000]))
        .unwrap();
    let values: Vec<String> = (0..3000).map(|i| format!("string {i}")).collect();
    w.write_strings(many, &Hyperslab::all(&[3000]), &values)
        .unwrap();
    let long = w
        .add_dataset(root, "long", DatasetSpec::strings(&[1]))
        .unwrap();
    w.write_strings(long, &Hyperslab::all(&[1]), &["x".repeat(100_000)])
        .unwrap();
    w.put_string_attribute(root, "labels", &["a", "bc"])
        .unwrap();
    w.close().unwrap().into_inner()
}

/// The directory that `scripts/check_writer.py` gives.
fn export_dir() -> std::path::PathBuf {
    std::env::var_os("OXCDF_EXPORT_DIR")
        .expect("set OXCDF_EXPORT_DIR to a directory")
        .into()
}

/// The seeds of the files for the external check.
const EXPORT_SEEDS: std::ops::RangeInclusive<u64> = 1..=80;

#[test]
#[ignore = "writes files for scripts/check_writer.py"]
fn export_files_for_the_external_check() {
    let dir = export_dir();
    for seed in EXPORT_SEEDS {
        let (bytes, model) = build(seed);
        std::fs::write(dir.join(format!("case_{seed}.h5")), bytes).unwrap();
        std::fs::write(dir.join(format!("case_{seed}.json")), model_json(&model)).unwrap();
    }
    std::fs::write(dir.join("strings.h5"), strings_file()).unwrap();
    std::fs::write(dir.join("chunks.h5"), chunks_file()).unwrap();
}

#[test]
#[ignore = "reads files that scripts/check_writer.py changed with the HDF5 library"]
fn files_changed_by_the_library_read_back() {
    let dir = export_dir();
    for seed in EXPORT_SEEDS {
        let (_, mut model) = build(seed);
        // The script adds one attribute to each object and one dataset to each group.
        let added = (
            "added_by_h5py".to_owned(),
            Value::Bytes(7i32.to_le_bytes().to_vec()),
        );
        for attributes in model
            .group_attributes
            .iter_mut()
            .chain(&mut model.dataset_attributes)
        {
            attributes.push(added.clone());
        }
        for g in 0..model.groups.len() {
            let mut path = model.groups[g].path.clone();
            path.push("added_by_h5py".to_owned());
            model.datasets.push(ModelDataset {
                path,
                dtype: DType::Int(4),
                order: ByteOrder::Little,
                shape: vec![3],
                fill: 0,
                values: vec![1, 2, 3],
                strings: None,
                chunks: None,
                max: Vec::new(),
                filters: Vec::new(),
            });
            model.dataset_attributes.push(Vec::new());
        }
        let bytes = std::fs::read(dir.join(format!("case_{seed}_mod.h5"))).unwrap();
        check(bytes, &model, &format!("changed seed {seed}"));
    }
}

#[test]
#[ignore = "reads a file that scripts/check_writer.py changed with the HDF5 library"]
fn strings_changed_by_the_library_read_back() {
    use oxcdf_hdf5::read::resolve_vlen_strings;
    let file = Hdf5File::open(export_dir().join("strings_mod.h5")).unwrap();
    let strings = |path: &str| {
        let d = file.dataset(path).unwrap();
        d.prepare(file.ctx()).unwrap();
        let raw = read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape)).unwrap();
        resolve_vlen_strings(file.ctx(), d, &raw).unwrap()
    };
    // The script writes one element of `vlen` and adds the dataset `added`.
    assert_eq!(
        strings("/vlen"),
        ["alpha", "", "zee\u{00eb}n", "added by h5py", "", ""]
    );
    assert_eq!(strings("/added"), ["p", "qr"]);
    assert_eq!(strings("/many")[2999], "string 2999");
}

/// The values of the dataset `grow` in the chunks file, for `rows` rows.
///
/// The writer leaves rows 100 to 109 at the fill value of 9.
fn grow_values(rows: i32) -> Vec<i32> {
    (0..rows)
        .flat_map(|row| {
            let filled = (100..110).contains(&row);
            (0..3).map(move |column| if filled { 9 } else { row * 3 + column })
        })
        .collect()
}

/// The values of the first 35 rows of the dataset `packed` in the chunks file.
fn packed_values() -> Vec<f32> {
    (0..35 * 50).map(|i| (i / 7) as f32).collect()
}

/// A file with chunked datasets and unlimited dimensions, and a known content.
fn chunks_file() -> Vec<u8> {
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
    let root = w.root();

    // 350 chunks, so the chunk index has two levels.
    let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[0, 3])
        .unwrap()
        .chunked(&[2, 3])
        .with_max_shape(&[None, Some(3)])
        .with_fill_value(9i32.to_le_bytes().to_vec());
    let grow = w.add_dataset(root, "grow", spec).unwrap();
    w.resize(grow, &[700, 3]).unwrap();
    let values = grow_values(700);
    for (start, count) in [(0u64, 100u64), (110, 590)] {
        let slab = Hyperslab::new(vec![start, 0], vec![count, 3], &[700, 3]).unwrap();
        let range = start as usize * 3..(start + count) as usize * 3;
        w.write(grow, &slab, &values[range]).unwrap();
    }
    w.put_text_attribute(grow, "units", "counts").unwrap();

    let spec = DatasetSpec::numeric(DType::Float(8), ByteOrder::Big, &[10, 7])
        .unwrap()
        .chunked(&[4, 3])
        .with_fill_value((-1.0f64).to_be_bytes().to_vec());
    let fixed = w.add_dataset(root, "fixed", spec).unwrap();
    let slab = Hyperslab::new(vec![2, 1], vec![7, 5], &[10, 7]).unwrap();
    let values: Vec<f64> = (0..35).map(f64::from).collect();
    w.write(fixed, &slab, &values).unwrap();

    let spec = DatasetSpec::strings(&[0])
        .chunked(&[4])
        .with_max_shape(&[None]);
    let names = w.add_dataset(root, "names", spec).unwrap();
    w.resize(names, &[6]).unwrap();
    let part = Hyperslab::new(vec![3], vec![3], &[6]).unwrap();
    w.write_strings(names, &part, &["d", "", "f"]).unwrap();

    let spec = DatasetSpec::numeric(DType::Int(2), ByteOrder::Little, &[0])
        .unwrap()
        .chunked(&[5])
        .with_max_shape(&[None]);
    w.add_dataset(root, "empty", spec).unwrap();

    let spec = DatasetSpec::numeric(DType::Uint(1), ByteOrder::Little, &[2, 2])
        .unwrap()
        .chunked(&[3, 3])
        .with_max_shape(&[None, None]);
    let two = w.add_dataset(root, "two", spec).unwrap();
    w.write(two, &Hyperslab::all(&[2, 2]), &[1u8, 2, 3, 4])
        .unwrap();

    // The filter order of netcdf-c: checksum, shuffle, deflate.
    let spec = DatasetSpec::numeric(DType::Float(4), ByteOrder::Little, &[40, 50])
        .unwrap()
        .chunked(&[10, 25])
        .with_fill_value((-1.0f32).to_le_bytes().to_vec())
        .with_filters(&[Filter::Fletcher32, Filter::Shuffle, Filter::Deflate(4)]);
    let packed = w.add_dataset(root, "packed", spec).unwrap();
    let slab = Hyperslab::new(vec![0, 0], vec![35, 50], &[40, 50]).unwrap();
    w.write(packed, &slab, &packed_values()).unwrap();

    let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[0, 8])
        .unwrap()
        .chunked(&[4, 8])
        .with_max_shape(&[None, Some(8)])
        .with_filters(&[Filter::Shuffle, Filter::Zstd(3)]);
    let zstd = w.add_dataset(root, "zstd", spec).unwrap();
    w.resize(zstd, &[10, 8]).unwrap();
    let values: Vec<i32> = (0..80).collect();
    w.write(zstd, &Hyperslab::all(&[10, 8]), &values).unwrap();
    w.close().unwrap().into_inner()
}

#[test]
fn the_chunks_file_reads_back() {
    let file = Hdf5File::from_source(Arc::new(MemorySource::new(chunks_file()))).unwrap();
    let grow = file.dataset("/grow").unwrap();
    grow.prepare(file.ctx()).unwrap();
    let raw = read_hyperslab(file.ctx(), grow, &Hyperslab::all(&grow.shape)).unwrap();
    assert_eq!(raw.get::<i32>(grow).unwrap(), grow_values(700));
    assert_eq!(file.dataset("/empty").unwrap().shape, [0]);
}

#[test]
#[ignore = "reads a file that scripts/check_writer.py changed with the HDF5 library"]
fn chunks_changed_by_the_library_read_back() {
    let file = Hdf5File::open(export_dir().join("chunks_mod.h5")).unwrap();
    let raw = |path: &str| {
        let d = file.dataset(path).unwrap();
        d.prepare(file.ctx()).unwrap();
        (
            d,
            read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape)).unwrap(),
        )
    };
    // The script grows `grow` to 1500 rows, `names` to 8 and `empty` to 3.
    let (d, grow) = raw("/grow");
    assert_eq!(d.shape, [1500, 3]);
    let mut want = grow_values(1500);
    want[0] = -5;
    assert_eq!(grow.get::<i32>(d).unwrap(), want);
    let (d, names) = raw("/names");
    let names = oxcdf_hdf5::read::resolve_vlen_strings(file.ctx(), d, &names).unwrap();
    assert_eq!(names, ["", "", "", "d", "", "f", "", "h"]);
    let (d, empty) = raw("/empty");
    assert_eq!(empty.get::<i16>(d).unwrap(), [7, 8, 9]);
    // The script writes one element of `packed` and grows `zstd` to 12 rows.
    let (d, packed) = raw("/packed");
    let mut want = packed_values();
    want.extend([-1.0; 5 * 50]);
    want[3] = 99.0;
    assert_eq!(packed.get::<f32>(d).unwrap(), want);
    let (d, zstd) = raw("/zstd");
    assert_eq!(zstd.get::<i32>(d).unwrap(), (0..96).collect::<Vec<i32>>());
}
