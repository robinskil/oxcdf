#![cfg(feature = "write")]
//! Files from the writer open in the reader.

use std::sync::Arc;

use oxcdf_hdf5::message::{ByteOrder, CharSet, DataspaceKind, Datatype, DatatypeClass, StringPad};
use oxcdf_hdf5::read::{read_hyperslab, resolve_vlen_strings, resolve_vlen_strings_of};
use oxcdf_hdf5::write::{DatasetSpec, FileSink, Filter, Hdf5Writer, MemorySink, WriteOptions};
use oxcdf_hdf5::{DType, Element, Error, Hdf5File, Hyperslab, MemorySource, RawData};

fn reopen(bytes: Vec<u8>) -> Hdf5File {
    Hdf5File::from_source(Arc::new(MemorySource::new(bytes))).expect("the file should open")
}

fn read_all<T: Element>(file: &Hdf5File, path: &str) -> Vec<T> {
    let d = file.dataset(path).unwrap();
    d.prepare(file.ctx()).unwrap();
    read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
        .unwrap()
        .get::<T>(d)
        .unwrap()
}

fn new_writer() -> Hdf5Writer<MemorySink> {
    Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap()
}

/// Whether `r` is [`Error::BadRequest`].
fn bad<T>(r: oxcdf_hdf5::Result<T>) -> bool {
    matches!(r, Err(Error::BadRequest(_)))
}

#[test]
fn an_empty_file_opens() {
    let w = new_writer();
    let bytes = w.close().unwrap().into_inner();
    // The superblock of 48 bytes and the default metadata block of 64 KiB.
    assert_eq!(bytes.len(), 48 + 65536);
    let file = reopen(bytes);
    assert_eq!(file.superblock().version, 2);
    assert!(file.root().groups.is_empty());
    assert!(file.datasets().is_empty());
}

#[test]
fn nested_groups_read_back_in_insert_order() {
    let mut w = new_writer();
    let root = w.root();
    let ocean = w.add_group(root, "ocean").unwrap();
    w.add_group(root, "air").unwrap();
    w.add_group(ocean, "deep").unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    let names: Vec<_> = file.root().groups.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(names, ["ocean", "air"]);
    assert!(file.root().group("ocean").unwrap().group("deep").is_some());
}

#[test]
fn the_same_calls_give_the_same_bytes() {
    let build = || {
        let mut w = new_writer();
        let root = w.root();
        let a = w.add_group(root, "a").unwrap();
        w.put_text_attribute(a, "title", "same").unwrap();
        w.put_numeric_attribute(root, "count", &[1i32, 2]).unwrap();
        let spec = DatasetSpec::numeric(DType::Float(8), ByteOrder::Little, &[3]).unwrap();
        let d = w.add_dataset(a, "v", spec).unwrap();
        w.write(d, &Hyperslab::all(&[3]), &[1.0f64, 2.0, 3.0])
            .unwrap();
        w.close().unwrap().into_inner()
    };
    assert_eq!(build(), build());
}

#[test]
fn wrong_group_names_are_bad_requests() {
    let mut w = new_writer();
    let root = w.root();
    w.add_group(root, "a").unwrap();
    for name in ["", "a", "x/y", ".", "nul\0"] {
        assert!(bad(w.add_group(root, name)), "{name:?}");
    }
    reopen(w.close().unwrap().into_inner());
}

#[test]
fn a_file_sink_gives_a_file_the_reader_opens() {
    let path = std::env::temp_dir().join(format!("oxcdf-write-{}.h5", std::process::id()));
    let mut w =
        Hdf5Writer::create(FileSink::create(&path).unwrap(), WriteOptions::default()).unwrap();
    let root = w.root();
    w.add_group(root, "g").unwrap();
    w.close().unwrap();
    assert!(Hdf5File::open(&path).unwrap().root().group("g").is_some());
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn group_attributes_read_back() {
    let mut w = new_writer();
    let root = w.root();
    w.put_text_attribute(root, "title", "Sea surface temperature")
        .unwrap();
    w.put_numeric_attribute(root, "version", &[3i32]).unwrap();
    w.put_numeric_attribute(root, "range", &[-2.0f64, 40.0])
        .unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    let root = file.root();

    let title = root.attribute("title").unwrap();
    assert_eq!(title.data, b"Sea surface temperature");
    assert_eq!(title.datatype.size, 23);
    assert_eq!(title.dataspace.kind, DataspaceKind::Scalar);
    let version = root.attribute("version").unwrap();
    assert_eq!(version.dataspace.kind, DataspaceKind::Simple);
    assert_eq!(version.dataspace.dims, vec![1]);
    assert_eq!(version.data, 3i32.to_le_bytes());
    let range = root.attribute("range").unwrap();
    assert_eq!(range.dataspace.dims, vec![2]);
    assert_eq!(
        range.data,
        [(-2.0f64).to_le_bytes(), 40.0f64.to_le_bytes()].concat()
    );
}

#[test]
fn a_second_put_replaces_the_value_and_keeps_the_order() {
    let mut w = new_writer();
    let root = w.root();
    w.put_numeric_attribute(root, "a", &[1u8]).unwrap();
    w.put_numeric_attribute(root, "b", &[2u8]).unwrap();
    w.put_numeric_attribute(root, "a", &[9u8]).unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    let names: Vec<_> = file
        .root()
        .attributes
        .iter()
        .map(|a| a.name.as_str())
        .collect();
    assert_eq!(names, ["a", "b"]);
    assert_eq!(file.root().attribute("a").unwrap().data, [9]);
}

#[test]
fn utf8_names_read_back() {
    let mut w = new_writer();
    let root = w.root();
    let g = w.add_group(root, "zee\u{00eb}n").unwrap();
    w.put_text_attribute(g, "omschr\u{00ef}jving", "co\u{00f6}rdinaat")
        .unwrap();
    let spec = DatasetSpec::numeric(DType::Float(4), ByteOrder::Little, &[2]).unwrap();
    let d = w.add_dataset(g, "temp\u{00e9}rature", spec).unwrap();
    w.write(d, &Hyperslab::all(&[2]), &[12.5f32, 13.0]).unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(
        read_all::<f32>(&file, "/zee\u{00eb}n/temp\u{00e9}rature"),
        [12.5, 13.0]
    );
    let g = file.root().group("zee\u{00eb}n").unwrap();
    assert_eq!(
        g.attribute("omschr\u{00ef}jving").unwrap().data,
        "co\u{00f6}rdinaat".as_bytes()
    );
}

#[test]
fn wrong_text_attributes_are_bad_requests() {
    let mut w = new_writer();
    let root = w.root();
    assert!(bad(w.put_text_attribute(root, "empty", "")));
    assert!(bad(w.put_text_attribute(root, "nul", "a\0b")));
    let file = reopen(w.close().unwrap().into_inner());
    assert!(file.root().attribute("empty").is_none());
    assert!(file.root().attribute("nul").is_none());
}

#[test]
fn an_oversize_attribute_fails_and_the_writer_stays_usable() {
    let mut w = new_writer();
    let root = w.root();
    let err = w
        .put_numeric_attribute(root, "big", &vec![0u8; 70000])
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)));
    w.put_numeric_attribute(root, "small", &[1u8]).unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    assert!(file.root().attribute("big").is_none());
    assert!(file.root().attribute("small").is_some());
}

#[test]
fn large_headers_read_back() {
    let mut w = new_writer();
    let root = w.root();
    for i in 0..2000 {
        w.add_group(root, &format!("group_{i:04}")).unwrap();
    }
    for i in 0..100i32 {
        w.put_numeric_attribute(root, &format!("attr_{i:03}"), &[i])
            .unwrap();
    }
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(file.root().groups.len(), 2000);
    assert_eq!(file.root().attributes.len(), 100);
    assert_eq!(
        file.root().attribute("attr_099").unwrap().data,
        99i32.to_le_bytes()
    );
}

#[test]
fn an_empty_numeric_attribute_is_a_bad_request() {
    let mut w = new_writer();
    let root = w.root();
    assert!(bad(w.put_numeric_attribute::<f32>(root, "none", &[])));
}

#[test]
fn every_numeric_type_round_trips_in_both_byte_orders() {
    macro_rules! check {
        ($t:ty, $dtype:expr) => {
            for order in [ByteOrder::Little, ByteOrder::Big] {
                let mut w = new_writer();
                let root = w.root();
                let spec = DatasetSpec::numeric($dtype, order, &[2, 3]).unwrap();
                let d = w.add_dataset(root, "v", spec).unwrap();
                let values: Vec<$t> = (0..6).map(|i| (i * 7 + 1) as $t).collect();
                w.write(d, &Hyperslab::all(&[2, 3]), &values).unwrap();
                let file = reopen(w.close().unwrap().into_inner());
                assert_eq!(read_all::<$t>(&file, "/v"), values);
                assert_eq!(
                    file.dataset("/v").unwrap().datatype.byte_order(),
                    Some(order)
                );
            }
        };
    }
    check!(i8, DType::Int(1));
    check!(i16, DType::Int(2));
    check!(i32, DType::Int(4));
    check!(i64, DType::Int(8));
    check!(u8, DType::Uint(1));
    check!(u16, DType::Uint(2));
    check!(u32, DType::Uint(4));
    check!(u64, DType::Uint(8));
    check!(f32, DType::Float(4));
    check!(f64, DType::Float(8));
}

#[test]
fn a_partial_write_leaves_the_fill_value_elsewhere() {
    let mut w = new_writer();
    let root = w.root();
    let spec = DatasetSpec::numeric(DType::Float(4), ByteOrder::Little, &[3, 4])
        .unwrap()
        .with_fill_value((-999.0f32).to_le_bytes().to_vec());
    let d = w.add_dataset(root, "t", spec).unwrap();
    let slab = Hyperslab::new(vec![1, 1], vec![2, 2], &[3, 4]).unwrap();
    w.write(d, &slab, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    let f = -999.0f32;
    assert_eq!(
        read_all::<f32>(&file, "/t"),
        [f, f, f, f, f, 1.0, 2.0, f, f, 3.0, 4.0, f]
    );
}

#[test]
fn two_partial_writes_add_up() {
    let mut w = new_writer();
    let root = w.root();
    let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[2, 3]).unwrap();
    let d = w.add_dataset(root, "v", spec).unwrap();
    w.write(
        d,
        &Hyperslab::new(vec![1, 0], vec![1, 3], &[2, 3]).unwrap(),
        &[4i32, 5, 6],
    )
    .unwrap();
    w.write(
        d,
        &Hyperslab::new(vec![0, 0], vec![1, 3], &[2, 3]).unwrap(),
        &[1i32, 2, 3],
    )
    .unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(read_all::<i32>(&file, "/v"), [1, 2, 3, 4, 5, 6]);
}

#[test]
fn a_dataset_with_no_write_reads_as_fill() {
    let mut w = new_writer();
    let root = w.root();
    let spec = DatasetSpec::numeric(DType::Int(2), ByteOrder::Little, &[4])
        .unwrap()
        .with_fill_value(7i16.to_le_bytes().to_vec());
    w.add_dataset(root, "v", spec).unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(read_all::<i16>(&file, "/v"), [7, 7, 7, 7]);
}

#[test]
fn a_scalar_dataset_round_trips() {
    let mut w = new_writer();
    let root = w.root();
    let d = w
        .add_dataset(
            root,
            "s",
            DatasetSpec::numeric(DType::Float(8), ByteOrder::Little, &[]).unwrap(),
        )
        .unwrap();
    w.write(d, &Hyperslab::all(&[]), &[2.5f64]).unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(read_all::<f64>(&file, "/s"), [2.5]);
}

#[test]
fn a_zero_length_dimension_gives_an_empty_dataset() {
    let mut w = new_writer();
    let root = w.root();
    let d = w
        .add_dataset(
            root,
            "e",
            DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[0, 5]).unwrap(),
        )
        .unwrap();
    w.write::<i32>(d, &Hyperslab::all(&[0, 5]), &[]).unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(file.dataset("/e").unwrap().shape, [0, 5]);
    assert!(read_all::<i32>(&file, "/e").is_empty());
}

#[test]
fn datasets_in_groups_carry_attributes() {
    let mut w = new_writer();
    let root = w.root();
    let g = w.add_group(root, "ocean").unwrap();
    let d = w
        .add_dataset(
            g,
            "temp",
            DatasetSpec::numeric(DType::Float(4), ByteOrder::Little, &[2]).unwrap(),
        )
        .unwrap();
    w.put_text_attribute(d, "units", "degC").unwrap();
    w.write(d, &Hyperslab::all(&[2]), &[1.5f32, 2.5]).unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(read_all::<f32>(&file, "/ocean/temp"), [1.5, 2.5]);
    assert_eq!(
        file.dataset("/ocean/temp")
            .unwrap()
            .attribute("units")
            .unwrap()
            .data,
        b"degC"
    );
}

#[test]
fn wrong_writes_are_rejected_and_change_nothing() {
    let mut w = new_writer();
    let root = w.root();
    let d = w
        .add_dataset(
            root,
            "v",
            DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[2, 2]).unwrap(),
        )
        .unwrap();
    let all = Hyperslab::all(&[2, 2]);
    assert!(bad(w.write(d, &all, &[1i32, 2, 3])));
    assert!(matches!(
        w.write(d, &all, &[1.0f32; 4]),
        Err(Error::TypeMismatch { .. })
    ));
    let outside = Hyperslab {
        start: vec![1, 1],
        count: vec![2, 2],
    };
    assert!(bad(w.write(d, &outside, &[1i32; 4])));
    let other = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[1]).unwrap();
    assert!(bad(w.add_dataset(root, "v", other)));
    w.write(d, &all, &[1i32, 2, 3, 4]).unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(read_all::<i32>(&file, "/v"), [1, 2, 3, 4]);
}

#[test]
fn a_wrong_fill_value_length_is_a_bad_request() {
    let mut w = new_writer();
    let root = w.root();
    let spec = DatasetSpec::numeric(DType::Float(8), ByteOrder::Little, &[2])
        .unwrap()
        .with_fill_value(vec![0; 4]);
    assert!(bad(w.add_dataset(root, "v", spec)));
}

#[test]
fn a_typed_write_needs_the_exact_datatype() {
    let mut w = new_writer();
    let root = w.root();
    let mut spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[2]).unwrap();
    if let DatatypeClass::FixedPoint { bit_precision, .. } = &mut spec.datatype.class {
        *bit_precision = 16;
    }
    let narrow = w.add_dataset(root, "narrow", spec).unwrap();
    let all = Hyperslab::all(&[2]);
    assert!(matches!(
        w.write(narrow, &all, &[1i32, 2]),
        Err(Error::TypeMismatch { .. })
    ));
    w.write_raw(narrow, &all, &[0; 8]).unwrap();

    let mut v3 = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[2]).unwrap();
    v3.datatype.version = 3;
    let v3 = w.add_dataset(root, "v3", v3).unwrap();
    w.write(v3, &all, &[1i32, 2]).unwrap();

    let wide = DatasetSpec::new(
        Datatype {
            version: 1,
            size: 260,
            class: DatatypeClass::FixedPoint {
                order: ByteOrder::Little,
                signed: true,
                bit_offset: 0,
                bit_precision: 2080,
            },
        },
        &[2],
    );
    let wide = w.add_dataset(root, "wide", wide).unwrap();
    assert!(matches!(
        w.write(wide, &all, &[1i32; 130]),
        Err(Error::TypeMismatch { .. })
    ));
    reopen(w.close().unwrap().into_inner());
}

#[test]
fn fixed_string_datasets_round_trip() {
    let string = |size, pad| Datatype {
        version: 1,
        size,
        class: DatatypeClass::String {
            pad,
            charset: CharSet::Ascii,
        },
    };
    let mut w = new_writer();
    let root = w.root();
    let spec = DatasetSpec::new(string(5, StringPad::NullPad), &[3]);
    let names = w.add_dataset(root, "names", spec).unwrap();
    let all = Hyperslab::all(&[3]);
    w.write_raw(names, &all, b"alphabe\0\0\0gamma").unwrap();
    assert!(matches!(
        w.write(names, &all, &[0u8; 3]),
        Err(Error::TypeMismatch { .. })
    ));

    // A netCDF character variable is a dataset of strings of 1 byte.
    let spec = DatasetSpec::new(string(1, StringPad::NullTerminate), &[2, 4]);
    let chars = w.add_dataset(root, "chars", spec).unwrap();
    let row = Hyperslab::new(vec![1, 0], vec![1, 4], &[2, 4]).unwrap();
    w.write_raw(chars, &row, b"wxyz").unwrap();

    let spec =
        DatasetSpec::new(string(4, StringPad::SpacePad), &[2]).with_fill_value(b"----".to_vec());
    let padded = w.add_dataset(root, "padded", spec).unwrap();
    let last = Hyperslab::new(vec![1], vec![1], &[2]).unwrap();
    w.write_raw(padded, &last, b"ab  ").unwrap();

    let file = reopen(w.close().unwrap().into_inner());
    let raw = |path: &str| {
        let d = file.dataset(path).unwrap();
        d.prepare(file.ctx()).unwrap();
        read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
            .unwrap()
            .bytes
    };
    assert_eq!(raw("/names"), b"alphabe\0\0\0gamma");
    assert_eq!(raw("/chars"), b"\0\0\0\0wxyz");
    assert_eq!(raw("/padded"), b"----ab  ");
    let names = file.dataset("/names").unwrap();
    assert_eq!(names.datatype, string(5, StringPad::NullPad));
}

fn read_strings(file: &Hdf5File, path: &str) -> Vec<String> {
    let d = file.dataset(path).unwrap();
    d.prepare(file.ctx()).unwrap();
    let raw = read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape)).unwrap();
    resolve_vlen_strings(file.ctx(), d, &raw).unwrap()
}

#[test]
fn vlen_strings_round_trip() {
    let mut w = new_writer();
    let root = w.root();
    let d = w
        .add_dataset(root, "names", DatasetSpec::strings(&[2, 3]))
        .unwrap();
    let row = Hyperslab::new(vec![0, 0], vec![1, 3], &[2, 3]).unwrap();
    w.write_strings(d, &row, &["alpha", "", "zee\u{00eb}n"])
        .unwrap();
    let long = "x".repeat(100_000);
    let last = Hyperslab::new(vec![1, 2], vec![1, 1], &[2, 3]).unwrap();
    w.write_strings(d, &last, &["first"]).unwrap();
    w.write_strings(d, &last, &[&long]).unwrap();
    w.put_string_attribute(d, "labels", &["a", "bc"]).unwrap();
    w.put_string_attribute(root, "one", &[String::from("solo")])
        .unwrap();

    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(
        read_strings(&file, "/names"),
        ["alpha", "", "zee\u{00eb}n", "", "", long.as_str()]
    );
    let attribute = |a: &oxcdf_hdf5::message::Attribute| {
        let raw = RawData {
            bytes: a.data.clone(),
            element_size: 16,
            shape: a.dataspace.dims.clone(),
        };
        resolve_vlen_strings_of(file.ctx(), &a.name, &raw).unwrap()
    };
    let names = file.dataset("/names").unwrap();
    assert_eq!(attribute(names.attribute("labels").unwrap()), ["a", "bc"]);
    let one = file.root().attribute("one").unwrap();
    assert_eq!(one.dataspace.dims, [1]);
    assert_eq!(attribute(one), ["solo"]);
}

#[test]
fn many_vlen_strings_span_heap_collections() {
    let mut w = new_writer();
    let root = w.root();
    let d = w
        .add_dataset(root, "s", DatasetSpec::strings(&[20_000]))
        .unwrap();
    let values: Vec<String> = (0..20_000).map(|i| format!("string {i}")).collect();
    for (i, part) in values.chunks(7000).enumerate() {
        let slab = Hyperslab::new(vec![i as u64 * 7000], vec![part.len() as u64], &[20_000]);
        w.write_strings(d, &slab.unwrap(), part).unwrap();
    }
    let bytes = w.close().unwrap().into_inner();
    assert!(bytes.windows(4).filter(|w| w == b"GCOL").count() > 5);
    assert_eq!(read_strings(&reopen(bytes), "/s"), values);
}

/// The heap address and object index of each element of a string dataset.
fn string_references(file: &Hdf5File, path: &str) -> Vec<(u64, u32)> {
    let d = file.dataset(path).unwrap();
    d.prepare(file.ctx()).unwrap();
    let raw = read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape)).unwrap();
    raw.bytes
        .chunks_exact(16)
        .map(|r| {
            let address = u64::from_le_bytes(r[4..12].try_into().unwrap());
            let index = u32::from_le_bytes(r[12..16].try_into().unwrap());
            (address, index)
        })
        .collect()
}

#[test]
fn a_string_element_with_no_value_holds_its_own_empty_string() {
    let mut w = new_writer();
    let root = w.root();
    let part = Hyperslab::new(vec![1], vec![2], &[6]).unwrap();
    let specs = [
        ("contiguous", DatasetSpec::strings(&[6])),
        ("chunked", DatasetSpec::strings(&[6]).chunked(&[4])),
        (
            "packed",
            DatasetSpec::strings(&[6])
                .chunked(&[4])
                .with_filters(&[Filter::Deflate(1)]),
        ),
    ];
    for (name, spec) in specs {
        let d = w.add_dataset(root, name, spec).unwrap();
        w.write_strings(d, &part, &["a", ""]).unwrap();
    }
    w.add_dataset(root, "none", DatasetSpec::strings(&[2]))
        .unwrap();
    let file = reopen(w.close().unwrap().into_inner());

    for name in ["/contiguous", "/chunked", "/packed"] {
        assert_eq!(read_strings(&file, name), ["", "a", "", "", "", ""]);
        let references = string_references(&file, name);
        // The HDF5 library reads an address of zero as no string.
        assert!(references.iter().all(|r| r.0 != 0), "{name}");
        // The library removes the object of an element at a new write to it.
        let mut distinct = references[..4].to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 4, "{name}");
    }
    assert_eq!(read_strings(&file, "/none"), ["", ""]);
    assert!(string_references(&file, "/none").iter().all(|r| r.0 != 0));
}

#[test]
fn wrong_string_calls_are_rejected() {
    let mut w = new_writer();
    let root = w.root();
    let d = w
        .add_dataset(root, "s", DatasetSpec::strings(&[2]))
        .unwrap();
    let numbers = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[2]).unwrap();
    let n = w.add_dataset(root, "n", numbers).unwrap();
    let all = Hyperslab::all(&[2]);
    assert!(bad(w.write_strings(d, &all, &["a\0b", "c"])));
    assert!(bad(w.write_strings(d, &all, &["only one"])));
    assert!(bad(w.write_raw(d, &all, &[0; 32])));
    assert!(bad(w.put_string_attribute::<&str>(root, "none", &[])));
    assert!(bad(w.put_string_attribute(root, "", &["a"])));
    assert!(bad(w.put_string_attribute(root, "nul", &["a\0"])));
    let filled = DatasetSpec::strings(&[2]).with_fill_value(vec![0; 16]);
    assert!(bad(w.add_dataset(root, "filled", filled)));
    assert!(matches!(
        w.write_strings(n, &all, &["a", "b"]),
        Err(Error::TypeMismatch { .. })
    ));
    assert!(matches!(
        w.write(d, &all, &[1u8, 2]),
        Err(Error::TypeMismatch { .. })
    ));
    let bytes = w.close().unwrap().into_inner();
    // The heap holds the empty string of the dataset. No rejected call put a string in it.
    assert_eq!(bytes.windows(4).filter(|w| w == b"GCOL").count(), 1);
    assert_eq!(bytes.windows(8).filter(|w| w == b"only one").count(), 0);
    let file = reopen(bytes);
    assert_eq!(read_strings(&file, "/s"), ["", ""]);
    assert!(file.root().attributes.is_empty());
}

#[test]
fn a_chunked_dataset_round_trips() {
    let mut w = new_writer();
    let root = w.root();
    let spec = DatasetSpec::numeric(DType::Float(8), ByteOrder::Little, &[10, 7])
        .unwrap()
        .chunked(&[4, 3])
        .with_fill_value((-1.0f64).to_le_bytes().to_vec());
    let d = w.add_dataset(root, "v", spec).unwrap();
    let slab = Hyperslab::new(vec![2, 1], vec![7, 5], &[10, 7]).unwrap();
    let values: Vec<f64> = (0..35).map(f64::from).collect();
    w.write(d, &slab, &values).unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    let mut want = vec![-1.0f64; 70];
    for row in 0..7 {
        for column in 0..5 {
            want[(row + 2) * 7 + column + 1] = (row * 5 + column) as f64;
        }
    }
    assert_eq!(read_all::<f64>(&file, "/v"), want);
}

#[test]
fn an_unlimited_dataset_grows() {
    let mut w = new_writer();
    let root = w.root();
    let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Big, &[0, 3])
        .unwrap()
        .chunked(&[2, 3])
        .with_max_shape(&[None, Some(3)])
        .with_fill_value(9i32.to_be_bytes().to_vec());
    let d = w.add_dataset(root, "t", spec).unwrap();
    let row = |at: u64, shape: &[u64]| Hyperslab::new(vec![at, 0], vec![1, 3], shape).unwrap();
    assert!(bad(w.write(d, &row(0, &[1, 3]), &[1i32, 2, 3])));
    w.resize(d, &[5, 3]).unwrap();
    for at in 0..5u64 {
        let base = at as i32 * 10;
        w.write(d, &row(at, &[5, 3]), &[base, base + 1, base + 2])
            .unwrap();
    }
    w.resize(d, &[9, 3]).unwrap();
    w.write(d, &row(8, &[9, 3]), &[80i32, 81, 82]).unwrap();

    let file = reopen(w.close().unwrap().into_inner());
    let t = file.dataset("/t").unwrap();
    assert_eq!(t.shape, [9, 3]);
    let mut want: Vec<i32> = (0..5)
        .flat_map(|r| [r * 10, r * 10 + 1, r * 10 + 2])
        .collect();
    want.extend([9; 9]);
    want.extend([80, 81, 82]);
    assert_eq!(read_all::<i32>(&file, "/t"), want);
}

#[test]
fn many_chunks_and_chunked_strings_round_trip() {
    let mut w = new_writer();
    let root = w.root();
    let spec = DatasetSpec::numeric(DType::Uint(2), ByteOrder::Little, &[5000])
        .unwrap()
        .chunked(&[10]);
    let d = w.add_dataset(root, "many", spec).unwrap();
    let values: Vec<u16> = (0..5000).collect();
    w.write(d, &Hyperslab::all(&[5000]), &values).unwrap();
    let spec = DatasetSpec::strings(&[0])
        .chunked(&[4])
        .with_max_shape(&[None]);
    let s = w.add_dataset(root, "names", spec).unwrap();
    w.resize(s, &[6]).unwrap();
    let part = Hyperslab::new(vec![3], vec![3], &[6]).unwrap();
    w.write_strings(s, &part, &["d", "", "f"]).unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(read_all::<u16>(&file, "/many"), values);
    assert_eq!(read_strings(&file, "/names"), ["", "", "", "d", "", "f"]);
}

#[test]
fn wrong_chunk_and_resize_calls_are_bad_requests() {
    let mut w = new_writer();
    let root = w.root();
    let int =
        |shape: &[u64]| DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, shape).unwrap();
    assert!(bad(w.add_dataset(root, "a", int(&[4, 4]).chunked(&[2]))));
    assert!(bad(w.add_dataset(root, "a", int(&[4, 4]).chunked(&[2, 0]))));
    assert!(bad(w.add_dataset(root, "a", int(&[4, 4]).chunked(&[2, 5]))));
    assert!(bad(w.add_dataset(root, "a", int(&[]).chunked(&[]))));
    assert!(bad(w.add_dataset(
        root,
        "a",
        int(&[1 << 31, 4]).chunked(&[1 << 31, 4])
    )));
    assert!(bad(w.add_dataset(
        root,
        "a",
        int(&[4]).with_max_shape(&[Some(3)])
    )));
    assert!(bad(w.add_dataset(
        root,
        "a",
        int(&[4]).with_max_shape(&[None])
    )));
    assert!(bad(w.add_dataset(
        root,
        "a",
        int(&[4]).with_max_shape(&[Some(4), None])
    )));
    let fixed = w.add_dataset(root, "fixed", int(&[4])).unwrap();
    assert!(bad(w.resize(fixed, &[5])));
    let spec = int(&[4, 2])
        .chunked(&[2, 2])
        .with_max_shape(&[Some(8), Some(2)]);
    let d = w.add_dataset(root, "d", spec).unwrap();
    assert!(bad(w.resize(d, &[3, 2])));
    assert!(bad(w.resize(d, &[9, 2])));
    assert!(bad(w.resize(d, &[8, 3])));
    assert!(bad(w.resize(d, &[8])));
    w.resize(d, &[8, 2]).unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(file.dataset("/d").unwrap().shape, [8, 2]);
    assert_eq!(read_all::<i32>(&file, "/d"), [0; 16]);
    assert_eq!(file.datasets().len(), 2);
}

#[test]
fn filtered_chunks_round_trip() {
    for filters in [
        vec![Filter::Shuffle, Filter::Deflate(4)],
        vec![Filter::Fletcher32, Filter::Shuffle, Filter::Deflate(9)],
        vec![Filter::Shuffle, Filter::Zstd(3)],
        vec![Filter::Fletcher32],
    ] {
        let mut w = new_writer();
        let root = w.root();
        let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[0, 50])
            .unwrap()
            .chunked(&[10, 20])
            .with_max_shape(&[None, Some(50)])
            .with_fill_value((-7i32).to_le_bytes().to_vec())
            .with_filters(&filters);
        let d = w.add_dataset(root, "v", spec).unwrap();
        w.resize(d, &[35, 50]).unwrap();
        let mut want = vec![-7i32; 35 * 50];
        // Full chunks, parts of chunks, and a second write to a chunk in memory.
        for (start, count) in [
            ([0u64, 0u64], [20u64, 40u64]),
            ([22, 5], [9, 30]),
            ([25, 0], [3, 50]),
        ] {
            let slab = Hyperslab::new(start.to_vec(), count.to_vec(), &[35, 50]).unwrap();
            let values: Vec<i32> = (0..count[0] * count[1]).map(|i| (i / 3) as i32).collect();
            w.write(d, &slab, &values).unwrap();
            for row in 0..count[0] {
                for column in 0..count[1] {
                    let at = (start[0] + row) * 50 + start[1] + column;
                    want[at as usize] = values[(row * count[1] + column) as usize];
                }
            }
        }
        let bytes = w.close().unwrap().into_inner();
        let file = reopen(bytes);
        assert_eq!(read_all::<i32>(&file, "/v"), want, "{filters:?}");
    }
}

#[test]
fn deflate_makes_the_file_smaller() {
    let build = |filters: &[Filter]| {
        let mut w = new_writer();
        let root = w.root();
        let spec = DatasetSpec::numeric(DType::Float(8), ByteOrder::Little, &[100, 1000])
            .unwrap()
            .chunked(&[10, 1000])
            .with_filters(filters);
        let d = w.add_dataset(root, "v", spec).unwrap();
        let values: Vec<f64> = (0..100_000).map(|i| f64::from(i / 100)).collect();
        w.write(d, &Hyperslab::all(&[100, 1000]), &values).unwrap();
        (w.close().unwrap().into_inner(), values)
    };
    let (plain, values) = build(&[]);
    let (small, _) = build(&[Filter::Shuffle, Filter::Deflate(6)]);
    assert!(
        small.len() < plain.len() / 5,
        "{} {}",
        small.len(),
        plain.len()
    );
    assert_eq!(read_all::<f64>(&reopen(small), "/v"), values);
}

#[test]
fn a_flushed_chunk_takes_only_a_full_write() {
    let mut w = new_writer();
    let root = w.root();
    let spec = DatasetSpec::numeric(DType::Uint(1), ByteOrder::Little, &[4, 4])
        .unwrap()
        .chunked(&[2, 4])
        .with_filters(&[Filter::Deflate(1)]);
    let d = w.add_dataset(root, "v", spec).unwrap();
    let row = |at: u64| Hyperslab::new(vec![at, 0], vec![1, 4], &[4, 4]).unwrap();
    w.write(d, &row(0), &[1u8; 4]).unwrap();
    w.write(d, &row(1), &[2u8; 4]).unwrap();
    w.flush_chunks(d).unwrap();
    assert!(matches!(
        w.write(d, &row(0), &[9u8; 4]),
        Err(Error::Unsupported(_))
    ));
    // A selection with one stored chunk and one new chunk changes nothing.
    let column = Hyperslab::new(vec![0, 0], vec![4, 1], &[4, 4]).unwrap();
    assert!(matches!(
        w.write(d, &column, &[9u8; 4]),
        Err(Error::Unsupported(_))
    ));
    let chunk = Hyperslab::new(vec![0, 0], vec![2, 4], &[4, 4]).unwrap();
    w.write(d, &chunk, &[5u8; 8]).unwrap();
    w.write(d, &row(3), &[3u8; 4]).unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(
        read_all::<u8>(&file, "/v"),
        [5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 3, 3, 3, 3]
    );
}

#[test]
fn wrong_filter_specs_are_bad_requests() {
    let mut w = new_writer();
    let root = w.root();
    let int = || DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[4]).unwrap();
    for spec in [
        int().with_filters(&[Filter::Deflate(1)]),
        int().chunked(&[2]).with_filters(&[Filter::Deflate(10)]),
        int().chunked(&[2]).with_filters(&[Filter::Zstd(0)]),
        int()
            .chunked(&[2])
            .with_filters(&[Filter::Shuffle, Filter::Shuffle]),
    ] {
        assert!(bad(w.add_dataset(root, "v", spec)));
    }
    let names = DatasetSpec::strings(&[3])
        .chunked(&[2])
        .with_filters(&[Filter::Shuffle, Filter::Deflate(2)]);
    let s = w.add_dataset(root, "s", names).unwrap();
    w.write_strings(s, &Hyperslab::all(&[3]), &["a", "bc", ""])
        .unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(read_strings(&file, "/s"), ["a", "bc", ""]);
}
