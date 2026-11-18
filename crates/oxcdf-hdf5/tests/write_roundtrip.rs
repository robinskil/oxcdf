#![cfg(feature = "write")]
//! Files from the writer open in the reader.

use std::sync::Arc;

use oxcdf_hdf5::message::DataspaceKind;
use oxcdf_hdf5::write::{FileSink, Hdf5Writer, MemorySink, WriteOptions};
use oxcdf_hdf5::{Error, Hdf5File, MemorySource};

fn reopen(bytes: Vec<u8>) -> Hdf5File {
    Hdf5File::from_source(Arc::new(MemorySource::new(bytes))).expect("the file should open")
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
    assert_eq!(title.data, b"Sea surface temperature\0");
    let version = root.attribute("version").unwrap();
    assert_eq!(version.dataspace.kind, DataspaceKind::Scalar);
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
        "co\u{00f6}rdinaat\0".as_bytes()
    );
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
    for i in 0..1000 {
        w.add_group(root, &format!("group_{i:04}")).unwrap();
    }
    for i in 0..100i32 {
        w.put_numeric_attribute(root, &format!("attr_{i:03}"), &[i])
            .unwrap();
    }
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(file.root().groups.len(), 1000);
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
