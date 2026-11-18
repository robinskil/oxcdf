#![cfg(feature = "write")]
//! Files from the writer open in the reader.

use std::sync::Arc;

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
