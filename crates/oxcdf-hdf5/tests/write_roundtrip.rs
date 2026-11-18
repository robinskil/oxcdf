#![cfg(feature = "write")]
//! Files from the writer open in the reader.

use std::sync::Arc;

use oxcdf_hdf5::message::{ByteOrder, DataspaceKind, Datatype, DatatypeClass};
use oxcdf_hdf5::read::read_hyperslab;
use oxcdf_hdf5::write::{DatasetSpec, FileSink, Hdf5Writer, MemorySink, WriteOptions};
use oxcdf_hdf5::{DType, Element, Error, Hdf5File, Hyperslab, MemorySource};

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

#[test]
fn an_empty_file_opens() {
    let w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
    let bytes = w.close().unwrap().into_inner();
    assert_eq!(bytes.len(), 48 + 65536);
    let file = reopen(bytes);
    assert_eq!(file.superblock().version, 2);
    assert!(file.root().groups.is_empty());
    assert!(file.datasets().is_empty());
}

#[test]
fn nested_groups_read_back_in_insert_order() {
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
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
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        w.add_group(root, "a").unwrap();
        w.close().unwrap().into_inner()
    };
    assert_eq!(build(), build());
}

#[test]
fn wrong_group_names_are_bad_requests() {
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
    let root = w.root();
    w.add_group(root, "a").unwrap();
    for name in ["", "a", "x/y", ".", "nul\0"] {
        assert!(
            matches!(w.add_group(root, name), Err(Error::BadRequest(_))),
            "{name:?}"
        );
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
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
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
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
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
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
    let root = w.root();
    let g = w.add_group(root, "zee\u{00eb}n").unwrap();
    w.put_text_attribute(g, "omschr\u{00ef}jving", "co\u{00f6}rdinaat")
        .unwrap();
    let file = reopen(w.close().unwrap().into_inner());
    let g = file.root().group("zee\u{00eb}n").unwrap();
    assert_eq!(
        g.attribute("omschr\u{00ef}jving").unwrap().data,
        "co\u{00f6}rdinaat".as_bytes()
    );
}

#[test]
fn wrong_text_attributes_are_bad_requests() {
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
    let root = w.root();
    assert!(matches!(
        w.put_text_attribute(root, "empty", ""),
        Err(Error::BadRequest(_))
    ));
    assert!(matches!(
        w.put_text_attribute(root, "nul", "a\0b"),
        Err(Error::BadRequest(_))
    ));
    let file = reopen(w.close().unwrap().into_inner());
    assert!(file.root().attribute("empty").is_none());
    assert!(file.root().attribute("nul").is_none());
}

#[test]
fn an_oversize_attribute_fails_and_the_writer_stays_usable() {
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
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
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
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
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
    let root = w.root();
    assert!(matches!(
        w.put_numeric_attribute::<f32>(root, "none", &[]),
        Err(Error::BadRequest(_))
    ));
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
    assert!(matches!(
        w.write(d, &all, &[1i32, 2, 3]),
        Err(Error::BadRequest(_))
    ));
    assert!(matches!(
        w.write(d, &all, &[1.0f32; 4]),
        Err(Error::TypeMismatch { .. })
    ));
    let outside = Hyperslab {
        start: vec![1, 1],
        count: vec![2, 2],
    };
    assert!(matches!(
        w.write(d, &outside, &[1i32; 4]),
        Err(Error::BadRequest(_))
    ));
    assert!(matches!(
        w.add_dataset(
            root,
            "v",
            DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[1]).unwrap()
        ),
        Err(Error::BadRequest(_))
    ));
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
    assert!(matches!(
        w.add_dataset(root, "v", spec),
        Err(Error::BadRequest(_))
    ));
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
