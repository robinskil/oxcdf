#![cfg(feature = "write")]
//! An edit session adds few bytes to the file when it changes little.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p oxcdf-hdf5 --features write --test write_edit_size
//! ```

use std::sync::Arc;

use oxcdf_hdf5::heap::{GlobalHeap, VlenDescriptor};
use oxcdf_hdf5::message::{ByteOrder, ChunkIndex, DatatypeClass, Layout};
use oxcdf_hdf5::read::{read_hyperslab, resolve_vlen_strings};
use oxcdf_hdf5::write::{DatasetSpec, Filter, Hdf5Writer, MemorySink, WriteOptions};
use oxcdf_hdf5::{DType, Element, Hdf5File, Hyperslab, MemorySource};

fn reopen(bytes: Vec<u8>) -> Hdf5File {
    Hdf5File::from_source(Arc::new(MemorySource::new(bytes))).unwrap()
}

fn read_all<T: Element>(file: &Hdf5File, path: &str) -> Vec<T> {
    let d = file.dataset(path).unwrap();
    d.prepare(file.ctx()).unwrap();
    read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
        .unwrap()
        .get(d)
        .unwrap()
}

fn read_strings(file: &Hdf5File, path: &str) -> Vec<String> {
    let d = file.dataset(path).unwrap();
    d.prepare(file.ctx()).unwrap();
    let raw = read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape)).unwrap();
    resolve_vlen_strings(file.ctx(), d, &raw).unwrap()
}

fn int4(shape: &[u64]) -> DatasetSpec {
    DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, shape).unwrap()
}

/// Open the file `bytes` for edits.
fn edit(bytes: Vec<u8>) -> Hdf5Writer<MemorySink> {
    let file = reopen(bytes.clone());
    Hdf5Writer::open(
        MemorySink::from_bytes(bytes),
        &file,
        WriteOptions::default(),
    )
    .unwrap()
}

/// The chunk index address and the index kind of the dataset `path`.
fn chunk_index(file: &Hdf5File, path: &str) -> (Option<u64>, ChunkIndex) {
    match &file.dataset(path).unwrap().layout {
        Layout::Chunked { address, index, .. } => (*address, index.clone()),
        other => panic!("{path} is not chunked: {other:?}"),
    }
}

/// The size field of each global heap collection that the file holds.
fn collection_sizes(bytes: &[u8]) -> Vec<u64> {
    (0..bytes.len().saturating_sub(16))
        .filter(|&at| &bytes[at..at + 4] == b"GCOL" && bytes[at + 4] == 1)
        .map(|at| u64::from_le_bytes(bytes[at + 8..at + 16].try_into().unwrap()))
        .collect()
}

/// A file with a dataset `d` and its scale `x` on axis 0.
fn file_with_a_scale() -> Vec<u8> {
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
    let root = w.root();
    let x = w.add_dataset(root, "x", int4(&[3])).unwrap();
    w.write(x, &Hyperslab::all(&[3]), &[10i32, 20, 30]).unwrap();
    let d = w.add_dataset(root, "d", int4(&[3])).unwrap();
    w.write(d, &Hyperslab::all(&[3]), &[1i32, 2, 3]).unwrap();
    w.attach_scale(d, 0, x).unwrap();
    w.close().unwrap().into_inner()
}

#[test]
fn an_edit_with_no_change_adds_a_small_heap_collection() {
    let before = file_with_a_scale();
    let after = edit(before.clone()).close().unwrap().into_inner();
    let growth = after.len() - before.len();
    println!("growth: {growth}");
    // The headers of three objects and one collection of 4 KiB.
    assert!(growth < 8192, "{growth}");

    let sizes = collection_sizes(&after[before.len()..]);
    assert_eq!(sizes, [4096]);
    let file = reopen(after);
    assert_eq!(read_all::<i32>(&file, "/d"), [1, 2, 3]);
    let d = file.dataset("/d").unwrap();
    let x = file.dataset("/x").unwrap();
    let list = d.attribute("DIMENSION_LIST").unwrap();
    let descriptor = VlenDescriptor::parse(&list.data, file.superblock().sizes).unwrap();
    let heap = GlobalHeap::read(file.ctx(), descriptor.collection_address).unwrap();
    let object = heap.object(descriptor.object_index as u16).unwrap();
    assert_eq!(object.data, x.address.to_le_bytes());
    let back = x.attribute("REFERENCE_LIST").unwrap();
    assert_eq!(back.data[..8], d.address.to_le_bytes());
}

#[test]
fn three_edit_sessions_add_three_small_collections() {
    let mut bytes = file_with_a_scale();
    let first = bytes.len();
    for _ in 0..3 {
        bytes = edit(bytes).close().unwrap().into_inner();
    }
    assert!(bytes.len() - first < 3 * 8192, "{}", bytes.len() - first);
    let file = reopen(bytes);
    assert_eq!(read_all::<i32>(&file, "/x"), [10, 20, 30]);
    assert!(file
        .dataset("/d")
        .unwrap()
        .attribute("DIMENSION_LIST")
        .is_some());
}

#[test]
fn many_strings_read_back_from_few_collections() {
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
    let root = w.root();
    let d = w
        .add_dataset(root, "s", DatasetSpec::strings(&[3000]))
        .unwrap();
    let values: Vec<String> = (0..3000).map(|i| format!("value {i}")).collect();
    w.write_strings(d, &Hyperslab::all(&[3000]), &values)
        .unwrap();
    w.put_string_attribute(root, "tags", &["a", "bc"]).unwrap();
    let bytes = w.close().unwrap().into_inner();

    let sizes = collection_sizes(&bytes);
    assert!(sizes.len() <= 6, "{sizes:?}");
    for size in &sizes {
        // The HDF5 library reads at least 4 KiB, and each object is aligned to 8 bytes.
        assert!(*size >= 4096 && size % 8 == 0, "{sizes:?}");
    }
    // Each string takes 32 bytes, so the heap is less than two times the strings.
    let total: u64 = sizes.iter().sum();
    assert!(total < 2 * 3000 * 32, "{sizes:?}");
    let file = reopen(bytes);
    assert_eq!(read_strings(&file, "/s"), values);
}

/// A file with two chunked datasets `a` and `b` of 10000 elements in chunks of 2.
fn file_with_two_chunked_datasets(filters: &[Filter]) -> Vec<u8> {
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
    let root = w.root();
    let values: Vec<i32> = (0..10_000).collect();
    for name in ["a", "b"] {
        let spec = int4(&[10_000])
            .chunked(&[2])
            .with_max_shape(&[None])
            .with_filters(filters)
            .with_fill_value((-1i32).to_le_bytes().to_vec());
        let d = w.add_dataset(root, name, spec).unwrap();
        w.write(d, &Hyperslab::all(&[10_000]), &values).unwrap();
    }
    w.close().unwrap().into_inner()
}

#[test]
fn an_edit_with_no_change_writes_no_chunk_index() {
    let before = file_with_two_chunked_datasets(&[]);
    let after = edit(before.clone()).close().unwrap().into_inner();
    let growth = after.len() - before.len();
    println!("growth: {growth}");
    // Two indexes of 5000 chunks take more than 300 KiB.
    assert!(growth < 4096, "{growth}");
    let (old, new) = (reopen(before), reopen(after));
    for path in ["/a", "/b"] {
        assert_eq!(chunk_index(&new, path), chunk_index(&old, path));
        assert_eq!(read_all::<i32>(&new, path), read_all::<i32>(&old, path));
    }
}

#[test]
fn a_new_chunk_writes_the_index_of_its_dataset_only() {
    let before = file_with_two_chunked_datasets(&[]);
    let mut w = edit(before.clone());
    let root = w.root();
    let a = w.find_dataset(root, "a").unwrap();
    w.resize(a, &[10_001]).unwrap();
    let tail = Hyperslab::new(vec![10_000], vec![1], &[10_001]).unwrap();
    w.write(a, &tail, &[77i32]).unwrap();
    let (old, new) = (reopen(before), reopen(w.close().unwrap().into_inner()));

    assert_ne!(chunk_index(&new, "/a").0, chunk_index(&old, "/a").0);
    assert_eq!(chunk_index(&new, "/b"), chunk_index(&old, "/b"));
    let mut expected: Vec<i32> = (0..10_000).collect();
    assert_eq!(read_all::<i32>(&new, "/b"), expected);
    expected.push(77);
    assert_eq!(read_all::<i32>(&new, "/a"), expected);
}

#[test]
fn a_write_in_place_and_a_resize_keep_the_index() {
    let before = file_with_two_chunked_datasets(&[]);
    let mut w = edit(before.clone());
    let root = w.root();
    let a = w.find_dataset(root, "a").unwrap();
    let one = Hyperslab::new(vec![5], vec![1], &[10_000]).unwrap();
    w.write(a, &one, &[-5i32]).unwrap();
    let b = w.find_dataset(root, "b").unwrap();
    // The new elements lie in no chunk, so they read as the fill value.
    w.resize(b, &[10_003]).unwrap();
    let (old, new) = (reopen(before), reopen(w.close().unwrap().into_inner()));

    assert_eq!(chunk_index(&new, "/a"), chunk_index(&old, "/a"));
    assert_eq!(chunk_index(&new, "/b"), chunk_index(&old, "/b"));
    let mut expected: Vec<i32> = (0..10_000).collect();
    expected[5] = -5;
    assert_eq!(read_all::<i32>(&new, "/a"), expected);
    expected[5] = 5;
    expected.extend([-1, -1, -1]);
    assert_eq!(read_all::<i32>(&new, "/b"), expected);
}

#[test]
fn a_chunk_with_filters_that_changes_writes_a_new_index() {
    let before = file_with_two_chunked_datasets(&[Filter::Deflate(1)]);
    let mut w = edit(before.clone());
    let root = w.root();
    let a = w.find_dataset(root, "a").unwrap();
    let one = Hyperslab::new(vec![5], vec![1], &[10_000]).unwrap();
    w.write(a, &one, &[-5i32]).unwrap();
    let (old, new) = (reopen(before), reopen(w.close().unwrap().into_inner()));

    assert_ne!(chunk_index(&new, "/a").0, chunk_index(&old, "/a").0);
    assert_eq!(chunk_index(&new, "/b"), chunk_index(&old, "/b"));
    let mut expected: Vec<i32> = (0..10_000).collect();
    assert_eq!(read_all::<i32>(&new, "/b"), expected);
    expected[5] = -5;
    assert_eq!(read_all::<i32>(&new, "/a"), expected);
}

#[test]
fn an_edit_keeps_each_version_1_index_of_the_corpus() {
    let dir = format!("{}/../../test_files", env!("CARGO_MANIFEST_DIR"));
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let mut paths: Vec<_> = entries.map(|e| e.unwrap().path()).collect();
    paths.sort();
    let (mut kept, mut new) = (0, 0);
    for path in paths {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let bytes = std::fs::read(&path).unwrap();
        if !bytes.starts_with(b"\x89HDF") {
            continue;
        }
        let before = reopen(bytes.clone());
        let sink = MemorySink::from_bytes(bytes);
        let Ok(writer) = Hdf5Writer::open(sink, &before, WriteOptions::default()) else {
            continue;
        };
        let after = reopen(writer.close().unwrap().into_inner());
        let superblock = before.superblock();
        // The new superblock gives each chunk index the default node size.
        let default_k = superblock.indexed_storage_internal_node_k == 32
            && superblock.extension_address.is_none();
        for d in before.datasets() {
            let Layout::Chunked { address, index, .. } = &d.layout else {
                continue;
            };
            let (found, found_index) = chunk_index(&after, &d.path);
            assert_eq!(found_index, ChunkIndex::BtreeV1, "{name} {}", d.path);
            // The close gives empty strings to each chunk of strings that the file does not have.
            let vlen = matches!(d.datatype.class, DatatypeClass::VariableLength { .. });
            if *index == ChunkIndex::BtreeV1 && default_k && !vlen {
                assert_eq!(found, *address, "{name} {}", d.path);
                kept += 1;
            } else {
                new += 1;
            }
            let read = |file: &Hdf5File| {
                let d = file.dataset(&d.path).unwrap();
                d.prepare(file.ctx())
                    .and_then(|()| read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape)))
                    .map(|raw| raw.bytes)
                    .map_err(|e| e.to_string())
            };
            assert_eq!(read(&after), read(&before), "{name} {}", d.path);
        }
    }
    println!("kept: {kept}, new: {new}");
    assert!(kept > 0 && new > 0, "kept: {kept}, new: {new}");
}
