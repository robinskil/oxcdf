//! The chunk grid: how a dataset tiles, and reading one tile at a time.
//!
//! These moved down from the netCDF layer. That layer now mirrors the `netcdf`
//! crate, which has no chunk API, so chunks are reached through HDF5.
//!
//! A chunk is an independent byte range with its own filter pipeline, so
//! reading chunks in parallel needs no coordination. That is the payoff of the
//! whole design, and it is what these tests hold in place.

use std::sync::Arc;

use oxcdf_hdf5::index::Hdf5File;
use oxcdf_hdf5::message::Layout;
use oxcdf_hdf5::read::{chunks_of, read_hyperslab};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_files/legacy_v1_objheader.h5"
);

fn argo() -> Option<Hdf5File> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files/test_file.nc");
    std::path::Path::new(path)
        .exists()
        .then(|| Hdf5File::open(path).unwrap())
}

/// The chunk shape a dataset was written with, or `None` when contiguous.
fn chunk_shape(file: &Hdf5File, path: &str) -> Option<Vec<u64>> {
    match &file.dataset(path)?.layout {
        Layout::Chunked { chunk_dims, .. } => Some(chunk_dims.iter().map(|&d| d as u64).collect()),
        _ => None,
    }
}

#[test]
fn exposes_the_chunk_grid_and_reads_one_chunk_at_a_time() {
    let file = Hdf5File::open(FIXTURE).unwrap();
    let d = file.dataset("/chunked_i32").unwrap();
    d.prepare(file.ctx()).unwrap();

    assert_eq!(chunk_shape(&file, "/chunked_i32"), Some(vec![7, 4]));

    let chunks = chunks_of(d).unwrap();
    assert_eq!(chunks.len(), 12, "a 40x6 dataset in 7x4 chunks");

    // Every chunk must be clipped to the dataset, and together they must cover
    // it exactly once.
    let mut covered = 0u64;
    for chunk in &chunks {
        assert!(chunk.stored_size > 0, "a written chunk has bytes on disk");
        for axis in 0..2 {
            assert!(chunk.offset[axis] + chunk.shape[axis] <= d.shape[axis]);
        }
        let block = read_hyperslab(file.ctx(), d, &chunk.selection()).unwrap();
        assert_eq!(block.shape, chunk.shape);
        assert_eq!(block.len() as u64, chunk.element_count());

        // Spot-check the first element of each chunk against the formula.
        let values = block.get::<i64>(d).unwrap();
        let global = chunk.offset[0] * 6 + chunk.offset[1];
        assert_eq!(values[0], global as i64 * 3 - 100);

        covered += chunk.element_count();
    }
    assert_eq!(covered, 240, "chunks must tile the dataset exactly");
}

#[test]
fn a_contiguous_dataset_reports_one_whole_chunk() {
    let file = Hdf5File::open(FIXTURE).unwrap();
    let d = file.dataset("/contig_f64").unwrap();
    d.prepare(file.ctx()).unwrap();

    assert_eq!(chunk_shape(&file, "/contig_f64"), None, "not chunked");
    let chunks = chunks_of(d).unwrap();
    assert_eq!(chunks.len(), 1, "callers get one uniform loop either way");
    assert_eq!(chunks[0].shape, vec![40, 6]);

    let block = read_hyperslab(file.ctx(), d, &chunks[0].selection()).unwrap();
    assert_eq!(block.len(), 240);
}

/// Chunks are independent byte ranges, so reading them in parallel needs no
/// coordination. This is the payoff of the whole design.
#[test]
fn chunks_can_be_read_from_many_threads_at_once() {
    let file = Arc::new(Hdf5File::open(FIXTURE).unwrap());
    let d = file.dataset("/chunked_i32").unwrap();
    d.prepare(file.ctx()).unwrap();
    let chunks = chunks_of(d).unwrap();

    let mut handles = Vec::new();
    for chunk in chunks {
        let file = Arc::clone(&file);
        handles.push(std::thread::spawn(move || {
            let d = file.dataset("/chunked_i32").unwrap();
            let block = read_hyperslab(file.ctx(), d, &chunk.selection()).unwrap();
            let values = block.get::<i64>(d).unwrap();
            let global = chunk.offset[0] * 6 + chunk.offset[1];
            assert_eq!(values[0], global as i64 * 3 - 100);
            values.len()
        }));
    }

    let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    assert_eq!(total, 240);
}

#[test]
fn every_readable_dataset_of_a_real_file_round_trips_through_chunks() {
    let Some(file) = argo() else { return };

    let mut checked = 0;
    for d in file.datasets() {
        if !d.is_readable() || d.shape.is_empty() {
            continue;
        }
        d.prepare(file.ctx()).unwrap();
        let Ok(chunks) = chunks_of(d) else { continue };
        let from_chunks: u64 = chunks.iter().map(|c| c.element_count()).sum();
        assert_eq!(
            d.element_count(),
            from_chunks,
            "{}: chunks must tile the dataset",
            d.path
        );
        checked += 1;
    }
    assert!(checked > 10, "expected many readable datasets");
}

/// A chunk read and the equivalent hyperslab read must agree.
#[test]
fn a_chunk_read_matches_the_equivalent_slice() {
    let Some(file) = argo() else { return };
    let d = file.dataset("/TEMP").unwrap();
    d.prepare(file.ctx()).unwrap();

    let chunks = chunks_of(d).unwrap();
    let total: u64 = chunks.iter().map(|c| c.element_count()).sum();
    assert_eq!(total, d.element_count());

    let chunk = &chunks[0];
    let from_chunk = read_hyperslab(file.ctx(), d, &chunk.selection())
        .unwrap()
        .get::<f64>(d)
        .unwrap();

    let slab = oxcdf_hdf5::read::Hyperslab {
        start: chunk.offset.clone(),
        count: chunk.shape.clone(),
    };
    let from_slice = read_hyperslab(file.ctx(), d, &slab)
        .unwrap()
        .get::<f64>(d)
        .unwrap();
    assert_eq!(from_chunk, from_slice);
}
