//! Turn fuzzer bytes into writer calls and compare the file with a model.
//!
//! Each call can return an error. An error must change nothing. A call that
//! returns `Ok` changes the model. At the end the reader must give the model.

use std::sync::Arc;

use oxcdf_hdf5::message::{
    Attribute, ByteOrder, CharSet, Dataspace, DataspaceKind, Datatype, DatatypeClass, StringPad,
};
use oxcdf_hdf5::read::{read_hyperslab, resolve_vlen_strings, resolve_vlen_strings_of};
use oxcdf_hdf5::write::{
    numeric_datatype, DatasetId, DatasetSpec, Filter, GroupId, Hdf5Writer, MemorySink, ObjectId,
    WriteOptions,
};
use oxcdf_hdf5::{DType, Element, GroupIndex, Hdf5File, Hyperslab, MemorySource, RawData};

/// Stop one input from a long run.
const MAX_OPS: usize = 64;

/// Link names. Some repeat, and the last four are wrong.
const LINK_NAMES: [&str; 12] = [
    "a", "b", "c", "d", "e", "f", "g", "h", "", ".", "x/y", "n\0",
];

/// Attribute names. Some repeat, and the last two are wrong.
const ATTRIBUTE_NAMES: [&str; 8] = ["p", "q", "r", "s", "t", "units", "", "n\0"];

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

/// The fuzzer bytes, read from the front.
struct Input<'a>(&'a [u8]);

impl Input<'_> {
    fn byte(&mut self) -> Option<u8> {
        let (&first, rest) = self.0.split_first()?;
        self.0 = rest;
        Some(first)
    }

    /// The next `n` bytes. A short input gives a fixed pad.
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let take = n.min(self.0.len());
        let (front, rest) = self.0.split_at(take);
        self.0 = rest;
        let mut out = front.to_vec();
        out.resize(n, 0xA5);
        out
    }
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
type Attributes = Vec<(String, Value)>;

struct GroupModel {
    id: GroupId,
    path: Vec<String>,
    groups: Vec<String>,
    attributes: Attributes,
}

struct DatasetModel {
    id: DatasetId,
    path: Vec<String>,
    datatype: Datatype,
    shape: Vec<u64>,
    /// Size of one element in bytes.
    element: usize,
    /// All elements in the dataset byte order. `None` for an extent above memory.
    bytes: Option<Vec<u8>>,
    /// All elements of a string dataset. `None` for each other dataset.
    strings: Option<Vec<String>>,
    /// Whether the storage is chunked.
    chunked: bool,
    /// One element with no write: the fill value or zeros.
    fill: Vec<u8>,
    attributes: Attributes,
}

/// A datatype from fuzzer bytes. Many of them are wrong.
fn fuzzed_datatype(input: &mut Input<'_>) -> Option<Datatype> {
    let order = if input.byte()? & 1 == 0 {
        ByteOrder::Little
    } else {
        ByteOrder::Big
    };
    let datatype = match input.byte()? % 13 {
        n @ 0..=9 => numeric_datatype(DTYPES[usize::from(n)].clone(), order).ok()?,
        10 => Datatype {
            version: 1,
            size: u32::from(input.byte()? % 18),
            class: DatatypeClass::FixedPoint {
                order,
                signed: input.byte()? & 1 == 1,
                bit_offset: u16::from(input.byte()?),
                bit_precision: u16::from(input.byte()?),
            },
        },
        11 => Datatype {
            version: 1,
            size: u32::from(input.byte()? % 18),
            class: DatatypeClass::String {
                pad: [
                    StringPad::NullTerminate,
                    StringPad::NullPad,
                    StringPad::SpacePad,
                ][usize::from(input.byte()? % 3)],
                charset: if input.byte()? & 1 == 0 {
                    CharSet::Ascii
                } else {
                    CharSet::Utf8
                },
            },
        },
        _ => {
            let mut datatype = numeric_datatype(DType::Float(4), order).ok()?;
            let (field, value) = (input.byte()? % 6, input.byte()?);
            if let DatatypeClass::FloatingPoint {
                bit_precision,
                exponent_location,
                exponent_size,
                mantissa_location,
                mantissa_size,
                sign_location,
                ..
            } = &mut datatype.class
            {
                match field {
                    0 => *bit_precision = u16::from(value),
                    1 => *exponent_location = value,
                    2 => *exponent_size = value,
                    3 => *mantissa_location = value,
                    4 => *mantissa_size = value,
                    _ => *sign_location = value,
                }
            }
            datatype
        }
    };
    Some(datatype)
}

/// Put `data` under `name`. A known name keeps its position.
fn upsert(attributes: &mut Attributes, name: &str, data: Value) {
    match attributes.iter_mut().find(|(n, _)| n == name) {
        Some(entry) => entry.1 = data,
        None => attributes.push((name.to_owned(), data)),
    }
}

/// The position of each element of the selection in the dataset, in row-major order.
fn positions(shape: &[u64], start: &[u64], count: &[u64]) -> Vec<usize> {
    let total: u64 = count.iter().product();
    let mut index = vec![0u64; count.len()];
    let mut out = Vec::new();
    for _ in 0..total {
        let mut flat = 0usize;
        for axis in 0..index.len() {
            flat = flat * shape[axis] as usize + (start[axis] + index[axis]) as usize;
        }
        out.push(flat);
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

/// Copy `bytes` to the elements of the selection.
fn apply(d: &mut DatasetModel, start: &[u64], count: &[u64], bytes: &[u8]) {
    let Some(model) = d.bytes.as_mut() else {
        return;
    };
    let elements = bytes.chunks_exact(d.element);
    for (flat, element) in positions(&d.shape, start, count).into_iter().zip(elements) {
        model[flat * d.element..(flat + 1) * d.element].copy_from_slice(element);
    }
}

/// Whether the selection lies in the shape.
fn inside(shape: &[u64], start: &[u64], count: &[u64]) -> bool {
    start
        .iter()
        .zip(count)
        .zip(shape)
        .all(|((&s, &c), &n)| s.checked_add(c).is_some_and(|end| end <= n))
}

/// Give `spec` chunks and a largest shape from fuzzer bytes. Many of them are wrong.
///
/// Return the spec and whether it asks for chunks.
fn fuzzed_chunks(
    input: &mut Input<'_>,
    kind: u8,
    mut spec: DatasetSpec,
) -> Option<(DatasetSpec, bool)> {
    if kind != 0 {
        return Some((spec, false));
    }
    let mut dims = Vec::new();
    let mut max = Vec::new();
    for &dim in &spec.shape.clone() {
        dims.push(u64::from(input.byte()? % 4));
        max.push(match input.byte()? % 4 {
            0 => None,
            1 => Some(dim),
            n => Some(dim + u64::from(n)),
        });
    }
    spec = spec.chunked(&dims);
    if input.byte()? % 4 != 0 {
        spec = spec.with_max_shape(&max);
    }
    // One bit for each filter. Some levels are wrong, and one filter can be there two times.
    let bits = input.byte()?;
    if bits & 0x80 != 0 {
        let level = input.byte()?;
        let mut filters = Vec::new();
        for (bit, filter) in [
            Filter::Fletcher32,
            Filter::Shuffle,
            Filter::Deflate(level % 11),
            Filter::Zstd(level % 24),
            Filter::Shuffle,
        ]
        .into_iter()
        .enumerate()
        {
            if bits & (1 << bit) != 0 {
                filters.push(filter);
            }
        }
        spec = spec.with_filters(&filters);
    }
    Some((spec, true))
}

/// Strings from fuzzer bytes. Some hold a NUL, and a few are long.
fn fuzzed_strings(input: &mut Input<'_>, count: usize) -> Option<Vec<String>> {
    let mut out = Vec::new();
    for _ in 0..count {
        out.push(match input.byte()? {
            // Longer than one heap collection.
            255 => "x".repeat(70_000),
            b => {
                let len = usize::from(b % 12);
                input.bytes(len).into_iter().map(char::from).collect()
            }
        });
    }
    Some(out)
}

/// The state of one run: the writer and what it must hold.
struct Run {
    w: Hdf5Writer<MemorySink>,
    groups: Vec<GroupModel>,
    datasets: Vec<DatasetModel>,
}

impl Run {
    fn add_group(&mut self, input: &mut Input<'_>) -> Option<()> {
        let parent = usize::from(input.byte()?) % self.groups.len();
        let name = LINK_NAMES[usize::from(input.byte()?) % LINK_NAMES.len()];
        if let Ok(id) = self.w.add_group(self.groups[parent].id, name) {
            let mut path = self.groups[parent].path.clone();
            path.push(name.to_owned());
            self.groups[parent].groups.push(name.to_owned());
            self.groups.push(GroupModel {
                id,
                path,
                groups: Vec::new(),
                attributes: Vec::new(),
            });
        }
        Some(())
    }

    fn add_dataset(&mut self, input: &mut Input<'_>) -> Option<()> {
        let parent = usize::from(input.byte()?) % self.groups.len();
        let name = LINK_NAMES[usize::from(input.byte()?) % LINK_NAMES.len()];
        let datatype = fuzzed_datatype(input)?;
        let rank = usize::from(input.byte()? % 4);
        let chunk_kind = input.byte()? % 3;
        let mut shape = Vec::new();
        for _ in 0..rank {
            // One value in 32 gives an extent that no file can hold, for contiguous storage.
            shape.push(match input.byte()? {
                b if b >= 248 && chunk_kind != 0 => u64::MAX,
                b => u64::from(b % 7),
            });
        }
        let element = datatype.size as usize;
        let mut spec = DatasetSpec::new(datatype.clone(), &shape);
        let fill = match input.byte()? % 8 {
            0..=2 => None,
            3 => Some(vec![0; element]),
            // A fill value of a wrong length.
            4 => Some(input.bytes(element + 1)),
            _ => Some(input.bytes(element)),
        };
        if let Some(fill) = &fill {
            spec = spec.with_fill_value(fill.clone());
        }
        let (spec, chunked) = fuzzed_chunks(input, chunk_kind, spec)?;
        let Ok(id) = self.w.add_dataset(self.groups[parent].id, name, spec) else {
            return Some(());
        };
        assert!(element > 0, "a dataset has elements of 0 bytes");
        let fill = fill.filter(|f| !f.is_empty());
        if let Some(fill) = &fill {
            assert_eq!(fill.len(), element, "a fill value of a wrong length");
        }
        let elements = shape.iter().try_fold(1u64, |n, &dim| n.checked_mul(dim));
        let bytes = match elements {
            Some(n) if n <= 1 << 16 => Some(match &fill {
                Some(fill) => fill.repeat(n as usize),
                None => vec![0; n as usize * element],
            }),
            // A zero axis hides an overflow in the other axes.
            None if shape.contains(&0) => Some(Vec::new()),
            _ => None,
        };
        let mut path = self.groups[parent].path.clone();
        path.push(name.to_owned());
        self.datasets.push(DatasetModel {
            id,
            path,
            datatype,
            shape,
            element,
            bytes,
            strings: None,
            chunked,
            fill: fill.unwrap_or_else(|| vec![0; element]),
            attributes: Vec::new(),
        });
        Some(())
    }

    fn add_string_dataset(&mut self, input: &mut Input<'_>) -> Option<()> {
        let parent = usize::from(input.byte()?) % self.groups.len();
        let name = LINK_NAMES[usize::from(input.byte()?) % LINK_NAMES.len()];
        let mut shape = Vec::new();
        for _ in 0..input.byte()? % 4 {
            shape.push(u64::from(input.byte()? % 7));
        }
        let kind = input.byte()? % 3;
        let (spec, chunked) = fuzzed_chunks(input, kind, DatasetSpec::strings(&shape))?;
        let datatype = spec.datatype.clone();
        let Ok(id) = self.w.add_dataset(self.groups[parent].id, name, spec) else {
            return Some(());
        };
        let elements: u64 = shape.iter().product();
        let mut path = self.groups[parent].path.clone();
        path.push(name.to_owned());
        self.datasets.push(DatasetModel {
            id,
            path,
            datatype,
            shape,
            element: 16,
            bytes: None,
            strings: Some(vec![String::new(); elements as usize]),
            chunked,
            fill: Vec::new(),
            attributes: Vec::new(),
        });
        Some(())
    }

    fn flush_chunks(&mut self, input: &mut Input<'_>) -> Option<()> {
        if self.datasets.is_empty() {
            return Some(());
        }
        let index = usize::from(input.byte()?) % self.datasets.len();
        self.w
            .flush_chunks(self.datasets[index].id)
            .expect("a flush to a memory sink");
        Some(())
    }

    fn resize(&mut self, input: &mut Input<'_>) -> Option<()> {
        if self.datasets.is_empty() {
            return Some(());
        }
        let index = usize::from(input.byte()?) % self.datasets.len();
        let d = &mut self.datasets[index];
        let mut shape = Vec::new();
        for &dim in &d.shape {
            shape.push(match input.byte()? % 5 {
                0 => dim.saturating_sub(1),
                1 | 2 => dim,
                n => dim.saturating_add(u64::from(n) - 2),
            });
        }
        if input.byte()? % 16 == 0 {
            shape.push(1);
        }
        // Keep the model small.
        let elements = shape.iter().try_fold(1u64, |n, &dim| n.checked_mul(dim));
        if elements.is_none_or(|n| n > 4096) {
            return Some(());
        }
        if self.w.resize(d.id, &shape).is_ok() {
            assert!(d.chunked, "a contiguous dataset changes its shape");
            assert_eq!(shape.len(), d.shape.len(), "a resize changes the rank");
            assert!(shape.iter().zip(&d.shape).all(|(new, old)| new >= old));
            // Each old element keeps its position on each axis.
            let old = positions(&shape, &vec![0; shape.len()], &d.shape);
            let elements = elements.unwrap_or(0) as usize;
            if let Some(bytes) = &mut d.bytes {
                let mut grown = d.fill.repeat(elements);
                for (&to, element) in old.iter().zip(bytes.chunks_exact(d.element)) {
                    grown[to * d.element..(to + 1) * d.element].copy_from_slice(element);
                }
                *bytes = grown;
            }
            if let Some(strings) = &mut d.strings {
                let mut grown = vec![String::new(); elements];
                for (&to, value) in old.iter().zip(strings.drain(..)) {
                    grown[to] = value;
                }
                *strings = grown;
            }
            d.shape = shape;
        }
        Some(())
    }

    fn write_strings(&mut self, input: &mut Input<'_>) -> Option<()> {
        if self.datasets.is_empty() {
            return Some(());
        }
        let index = usize::from(input.byte()?) % self.datasets.len();
        let d = &mut self.datasets[index];
        let (start, count) = Self::selection(input, &d.shape)?;
        let elements: u64 = count.iter().product();
        let mut values = fuzzed_strings(input, elements as usize)?;
        let valid_len = input.byte()? % 9 != 8;
        if !valid_len {
            values.push(String::new());
        }
        let slab = Hyperslab {
            start: start.clone(),
            count: count.clone(),
        };
        if self.w.write_strings(d.id, &slab, &values).is_ok() {
            let Some(model) = d.strings.as_mut() else {
                panic!("a string write to another datatype returns Ok");
            };
            assert!(
                inside(&d.shape, &start, &count),
                "a write outside the shape"
            );
            assert!(
                valid_len,
                "a write with a wrong number of strings returns Ok"
            );
            assert!(values.iter().all(|v| !v.contains('\0')), "a NUL is Ok");
            for (flat, value) in positions(&d.shape, &start, &count).into_iter().zip(values) {
                model[flat] = value;
            }
        }
        Some(())
    }

    fn put_string_attribute(&mut self, input: &mut Input<'_>) -> Option<()> {
        let name = ATTRIBUTE_NAMES[usize::from(input.byte()?) % ATTRIBUTE_NAMES.len()];
        let count = usize::from(input.byte()? % 4);
        let values = fuzzed_strings(input, count)?;
        let (object, attributes) = Self::object(&mut self.groups, &mut self.datasets, input)?;
        if self.w.put_string_attribute(object, name, &values).is_ok() {
            assert!(
                !name.is_empty() && !name.contains('\0'),
                "a wrong name is Ok"
            );
            assert!(!values.is_empty(), "an attribute with no strings is Ok");
            assert!(values.iter().all(|v| !v.contains('\0')), "a NUL is Ok");
            upsert(attributes, name, Value::Strings(values));
        }
        Some(())
    }

    /// A selection from fuzzer bytes. It can lie outside the shape.
    fn selection(input: &mut Input<'_>, shape: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        let mut start = Vec::new();
        let mut count = Vec::new();
        for _ in shape {
            start.push(u64::from(input.byte()? % 7));
            count.push(u64::from(input.byte()? % 7));
        }
        Some((start, count))
    }

    /// Check a write result against the shape, then change the model.
    fn after_write(
        d: &mut DatasetModel,
        ok: bool,
        valid_len: bool,
        start: &[u64],
        count: &[u64],
        bytes: &[u8],
    ) {
        if !ok {
            return;
        }
        assert!(
            d.strings.is_none(),
            "raw bytes to a string dataset return Ok"
        );
        assert!(inside(&d.shape, start, count), "a write outside the shape");
        assert!(valid_len, "a write with a wrong length returns Ok");
        // A dataset above memory has no space, so only an empty write is Ok.
        assert!(d.bytes.is_some() || bytes.is_empty(), "a huge write is Ok");
        apply(d, start, count, bytes);
    }

    fn write_raw(&mut self, input: &mut Input<'_>) -> Option<()> {
        if self.datasets.is_empty() {
            return Some(());
        }
        let index = usize::from(input.byte()?) % self.datasets.len();
        let d = &mut self.datasets[index];
        let (start, count) = Self::selection(input, &d.shape)?;
        let elements: u64 = count.iter().product();
        let mut len = elements as usize * d.element;
        let valid_len = input.byte()? % 9 != 8;
        if !valid_len {
            len += 1;
        }
        let bytes = input.bytes(len);
        let slab = Hyperslab {
            start: start.clone(),
            count: count.clone(),
        };
        let ok = self.w.write_raw(d.id, &slab, &bytes).is_ok();
        Self::after_write(d, ok, valid_len, &start, &count, &bytes);
        Some(())
    }

    fn write_typed<T: Element>(
        &mut self,
        input: &mut Input<'_>,
        make: fn(u8) -> T,
        to_bytes: fn(&T, bool) -> Vec<u8>,
    ) -> Option<()> {
        if self.datasets.is_empty() {
            return Some(());
        }
        let index = usize::from(input.byte()?) % self.datasets.len();
        let d = &mut self.datasets[index];
        let (start, count) = Self::selection(input, &d.shape)?;
        let elements: u64 = count.iter().product();
        let values: Vec<T> = input
            .bytes(elements as usize)
            .into_iter()
            .map(make)
            .collect();
        let slab = Hyperslab {
            start: start.clone(),
            count: count.clone(),
        };
        let ok = self.w.write(d.id, &slab, &values).is_ok();
        if ok {
            let exact = numeric_datatype(T::DTYPE, ByteOrder::Little)
                .is_ok_and(|e| e.size == d.datatype.size)
                && DType::of(&d.datatype) == T::DTYPE;
            assert!(exact, "a typed write to another datatype returns Ok");
        }
        let big = d.datatype.byte_order() == Some(ByteOrder::Big);
        let bytes: Vec<u8> = values.iter().flat_map(|v| to_bytes(v, big)).collect();
        Self::after_write(d, ok, true, &start, &count, &bytes);
        Some(())
    }

    /// The id and the attributes of the object that one fuzzer byte selects.
    fn object<'a>(
        groups: &'a mut [GroupModel],
        datasets: &'a mut [DatasetModel],
        input: &mut Input<'_>,
    ) -> Option<(ObjectId, &'a mut Attributes)> {
        let index = usize::from(input.byte()?) % (groups.len() + datasets.len());
        Some(match index.checked_sub(groups.len()) {
            None => {
                let g = &mut groups[index];
                (g.id.into(), &mut g.attributes)
            }
            Some(index) => {
                let d = &mut datasets[index];
                (d.id.into(), &mut d.attributes)
            }
        })
    }

    fn put_attribute(&mut self, input: &mut Input<'_>, kind: u8) -> Option<()> {
        let name = ATTRIBUTE_NAMES[usize::from(input.byte()?) % ATTRIBUTE_NAMES.len()];
        let (data, attribute) = match kind {
            0 => {
                let count = usize::from(input.byte()? % 4);
                let values: Vec<i16> = input.bytes(count).into_iter().map(i16::from).collect();
                let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                (data, Err(values))
            }
            1 => {
                let len = usize::from(input.byte()? % 6);
                let text: String = input.bytes(len).into_iter().map(char::from).collect();
                let (object, attributes) =
                    Self::object(&mut self.groups, &mut self.datasets, input)?;
                if self.w.put_text_attribute(object, name, &text).is_ok() {
                    assert!(
                        !text.is_empty() && !text.contains('\0'),
                        "a wrong text is Ok"
                    );
                    upsert(attributes, name, Value::Bytes(text.into_bytes()));
                }
                return Some(());
            }
            _ => {
                let datatype = fuzzed_datatype(input)?;
                let kind = [
                    DataspaceKind::Scalar,
                    DataspaceKind::Simple,
                    DataspaceKind::Null,
                ][usize::from(input.byte()? % 3)];
                let mut dims = Vec::new();
                for _ in 0..input.byte()? % 3 {
                    dims.push(u64::from(input.byte()? % 4));
                }
                let elements: u64 = match kind {
                    DataspaceKind::Null => 0,
                    _ => dims.iter().product(),
                };
                let mut len = elements as usize * datatype.size as usize;
                if input.byte()? % 9 == 8 {
                    len += 1;
                }
                let data = input.bytes(len);
                let attribute = Attribute {
                    name: name.to_owned(),
                    datatype,
                    dataspace: Dataspace {
                        kind,
                        dims,
                        max_dims: None,
                    },
                    data: data.clone(),
                };
                (data, Ok(attribute))
            }
        };
        let (object, attributes) = Self::object(&mut self.groups, &mut self.datasets, input)?;
        let ok = match attribute {
            Ok(attribute) => self.w.put_attribute(object, attribute).is_ok(),
            Err(values) => self.w.put_numeric_attribute(object, name, &values).is_ok(),
        };
        if ok {
            assert!(
                !name.is_empty() && !name.contains('\0'),
                "a wrong name is Ok"
            );
            upsert(attributes, name, Value::Bytes(data));
        }
        Some(())
    }
}

fn group<'a>(file: &'a Hdf5File, path: &[String]) -> &'a GroupIndex {
    path.iter()
        .fold(file.root(), |g, name| g.group(name).expect("the group"))
}

fn attributes_of(file: &Hdf5File, attributes: &[Attribute]) -> Attributes {
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
                    Value::Strings(
                        resolve_vlen_strings_of(file.ctx(), &a.name, &raw).expect("strings"),
                    )
                }
                _ => Value::Bytes(a.data.clone()),
            };
            (a.name.clone(), value)
        })
        .collect()
}

/// Run the writer calls that `data` encodes and check the file.
pub fn run(data: &[u8]) {
    let mut input = Input(data);
    let Some(first) = input.byte() else {
        return;
    };
    // A small block and no slack reach the allocator paths that the default hides.
    let options = match first % 3 {
        0 => WriteOptions::default(),
        1 => WriteOptions::default()
            .metadata_block_bytes(0)
            .header_slack_bytes(0),
        _ => WriteOptions::default().metadata_block_bytes(256),
    };
    let w = Hdf5Writer::create(MemorySink::new(), options).expect("create");
    let root = GroupModel {
        id: w.root(),
        path: Vec::new(),
        groups: Vec::new(),
        attributes: Vec::new(),
    };
    let mut run = Run {
        w,
        groups: vec![root],
        datasets: Vec::new(),
    };
    for _ in 0..MAX_OPS {
        let Some(op) = input.byte() else {
            break;
        };
        let done = match op % 10 {
            0 => run.add_group(&mut input),
            1 | 2 => run.add_dataset(&mut input),
            3 | 4 => run.write_raw(&mut input),
            5 => match op / 10 % 3 {
                0 => run.write_typed::<i32>(&mut input, i32::from, |v, big| {
                    if big {
                        v.to_be_bytes().to_vec()
                    } else {
                        v.to_le_bytes().to_vec()
                    }
                }),
                1 => run.write_typed::<f64>(&mut input, f64::from, |v, big| {
                    if big {
                        v.to_be_bytes().to_vec()
                    } else {
                        v.to_le_bytes().to_vec()
                    }
                }),
                _ => run.write_typed::<u8>(&mut input, |b| b, |v, _| vec![*v]),
            },
            9 => match op / 10 % 5 {
                0 => run.add_string_dataset(&mut input),
                1 => run.write_strings(&mut input),
                2 => run.resize(&mut input),
                3 => run.flush_chunks(&mut input),
                _ => run.put_string_attribute(&mut input),
            },
            n => run.put_attribute(&mut input, n - 6),
        };
        if done.is_none() {
            break;
        }
    }

    let Run {
        w,
        groups,
        datasets,
    } = run;
    let bytes = w.close().expect("close").into_inner();
    let len = bytes.len() as u64;
    let file = Hdf5File::from_source(Arc::new(MemorySource::new(bytes))).expect("open");
    assert_eq!(file.superblock().end_of_file_address, len);

    for g in &groups {
        let got = group(&file, &g.path);
        let names: Vec<_> = got.groups.iter().map(|c| c.name.clone()).collect();
        assert_eq!(names, g.groups, "{:?}", g.path);
        let attributes = attributes_of(&file, &got.attributes);
        assert_eq!(attributes, g.attributes, "{:?}", g.path);
    }
    for d in &datasets {
        let path = format!("/{}", d.path.join("/"));
        let got = file.dataset(&path).expect("the dataset");
        assert_eq!(got.shape, d.shape, "{path}");
        assert_eq!(got.datatype.size, d.datatype.size, "{path}");
        assert_eq!(got.datatype.class, d.datatype.class, "{path}");
        assert_eq!(
            attributes_of(&file, &got.attributes),
            d.attributes,
            "{path}"
        );
        if d.bytes.is_none() && d.strings.is_none() {
            continue;
        }
        got.prepare(file.ctx()).expect("prepare");
        let raw = read_hyperslab(file.ctx(), got, &Hyperslab::all(&got.shape)).expect("read");
        if let Some(want) = &d.strings {
            let strings = resolve_vlen_strings(file.ctx(), got, &raw).expect("strings");
            assert_eq!(&strings, want, "{path}");
            continue;
        }
        let Some(want) = &d.bytes else {
            continue;
        };
        let mut want = want.clone();
        // The reader swaps each big-endian element.
        if d.datatype.byte_order() == Some(ByteOrder::Big) {
            for element in want.chunks_exact_mut(d.element) {
                element.reverse();
            }
        }
        assert_eq!(raw.bytes, want, "{path}");
    }
}
