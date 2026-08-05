//! Exercises the netCDF/zarr-style interface: navigate to a variable, read its
//! metadata and attributes, then read all of it, a slice of it, or one stored
//! chunk at a time.

use std::sync::Arc;

use oxcdf::netcdf::{DType, NetcdfFile};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_files/legacy_v1_objheader.h5"
);

fn argo() -> Option<NetcdfFile> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_files/test_file.nc"
    );
    std::path::Path::new(path)
        .exists()
        .then(|| NetcdfFile::open(path).unwrap())
}

#[test]
fn navigates_variables_and_reports_their_types() {
    let file = NetcdfFile::open(FIXTURE).unwrap();

    let v = file.variable("/contig_f64").unwrap();
    assert_eq!(v.name, "contig_f64");
    assert_eq!(v.shape, vec![40, 6]);
    assert_eq!(v.dtype(), DType::Float(8));
    assert!(v.dtype().is_float());
    assert_eq!(v.element_count(), 240);
    assert!(v.is_readable());

    assert_eq!(file.variable("/chunked_i32").unwrap().dtype(), DType::Int(4));
    assert_eq!(
        file.variable("/fixed_strings").unwrap().dtype(),
        DType::String(8)
    );
    assert!(file.variable("/nope").is_none());
}

#[test]
fn reads_global_and_variable_attributes() {
    let file = NetcdfFile::open(FIXTURE).unwrap();

    let title = file.attribute("title").expect("a global attribute");
    assert_eq!(title.value.as_text(), Some("legacy fixture"));

    let v = file.variable("/contig_f64").unwrap();
    let range = v.attribute("valid_range").expect("a variable attribute");
    assert_eq!(range.value.as_f64(), Some(-1.0));
}

#[test]
fn reads_a_whole_variable() {
    let file = NetcdfFile::open(FIXTURE).unwrap();
    let values = file.variable("/contig_f64").unwrap().read().unwrap();

    assert_eq!(values.shape(), &[40, 6]);
    assert_eq!(values.len(), 240);
    let f = values.to_f64().unwrap();
    assert_eq!(f[0], 0.0);
    assert_eq!(f[239], 239.0 * 0.5);
}

#[test]
fn reads_a_slice_given_ranges_per_axis() {
    let file = NetcdfFile::open(FIXTURE).unwrap();
    let v = file.variable("/chunked_i32").unwrap();

    let block = v.read_slice(&[5..15, 2..5]).unwrap();
    assert_eq!(block.shape(), &[10, 3]);

    let values = block.to_i64().unwrap();
    for row in 0..10usize {
        for col in 0..3usize {
            let global = (5 + row) * 6 + (2 + col);
            assert_eq!(values[row * 3 + col], global as i64 * 3 - 100);
        }
    }
}

#[test]
fn rejects_a_slice_of_the_wrong_rank_or_a_reversed_range() {
    let file = NetcdfFile::open(FIXTURE).unwrap();
    let v = file.variable("/chunked_i32").unwrap();
    // Built indirectly so the deliberately-invalid bounds stay opaque to lints.
    let (lo, hi) = (0u64, 4u64);

    // One range for a rank-2 variable.
    let wrong_rank: Vec<std::ops::Range<u64>> = (0..1).map(|_| lo..hi).collect();
    assert!(v.read_slice(&wrong_rank).is_err(), "rank must match");

    let reversed = vec![hi..lo, lo..2];
    assert!(v.read_slice(&reversed).is_err(), "range must not reverse");

    let out_of_bounds = vec![lo..99, lo..2];
    assert!(v.read_slice(&out_of_bounds).is_err(), "must stay in bounds");
}

#[test]
fn exposes_the_chunk_grid_and_reads_one_chunk_at_a_time() {
    let file = NetcdfFile::open(FIXTURE).unwrap();
    let v = file.variable("/chunked_i32").unwrap();

    assert_eq!(v.chunk_shape(), Some(vec![7, 4]));

    let chunks = v.chunks();
    assert_eq!(chunks.len(), 12, "a 40x6 variable in 7x4 chunks");

    // Every chunk must be clipped to the variable, and together they must cover
    // it exactly once.
    let mut covered = 0u64;
    for chunk in &chunks {
        assert!(chunk.stored_size > 0, "a written chunk has bytes on disk");
        for axis in 0..2 {
            assert!(chunk.offset[axis] + chunk.shape[axis] <= v.shape[axis]);
        }
        let block = v.read_chunk(chunk).unwrap();
        assert_eq!(block.shape(), &chunk.shape[..]);
        assert_eq!(block.len() as u64, chunk.element_count());

        // Spot-check the first element of each chunk against the formula.
        let values = block.to_i64().unwrap();
        let global = chunk.offset[0] * 6 + chunk.offset[1];
        assert_eq!(values[0], global as i64 * 3 - 100);

        covered += chunk.element_count();
    }
    assert_eq!(covered, 240, "chunks must tile the variable exactly");
}

#[test]
fn a_contiguous_variable_reports_one_whole_chunk() {
    let file = NetcdfFile::open(FIXTURE).unwrap();
    let v = file.variable("/contig_f64").unwrap();

    assert_eq!(v.chunk_shape(), None, "not chunked");
    let chunks = v.chunks();
    assert_eq!(chunks.len(), 1, "callers get one uniform loop either way");
    assert_eq!(chunks[0].shape, vec![40, 6]);
    assert_eq!(v.read_chunk(&chunks[0]).unwrap().len(), 240);
}

/// Chunks are independent byte ranges, so reading them in parallel needs no
/// coordination. This is the payoff of the whole design.
#[test]
fn chunks_can_be_read_from_many_threads_at_once() {
    let file = Arc::new(NetcdfFile::open(FIXTURE).unwrap());
    let chunks = file.variable("/chunked_i32").unwrap().chunks();

    let mut handles = Vec::new();
    for chunk in chunks {
        let file = Arc::clone(&file);
        handles.push(std::thread::spawn(move || {
            let v = file.variable("/chunked_i32").unwrap();
            let values = v.read_chunk(&chunk).unwrap().to_i64().unwrap();
            let global = chunk.offset[0] * 6 + chunk.offset[1];
            assert_eq!(values[0], global as i64 * 3 - 100);
            values.len()
        }));
    }

    let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    assert_eq!(total, 240);
}

#[test]
fn reads_strings_and_nested_group_variables() {
    let file = NetcdfFile::open(FIXTURE).unwrap();

    let s = file.variable("/fixed_strings").unwrap().read().unwrap();
    assert_eq!(
        s.to_strings().unwrap(),
        vec!["alpha", "beta", "gamma", "delta", "epsilon"]
    );

    let n = file.variable("/subgroup/nested_i16").unwrap();
    assert_eq!(n.read().unwrap().to_i64().unwrap(), (1000..1006).collect::<Vec<_>>());
    assert!(file.group("/subgroup").is_some());
    assert!(file.group("/").is_some());
}

// ── against a real netCDF file ────────────────────────────────────────────

#[test]
fn dimensions_and_variable_axes_are_reachable_on_a_real_file() {
    let Some(file) = argo() else { return };

    assert!(!file.dimensions().is_empty());
    let n_prof = file.dimensions().iter().find(|d| d.name == "N_PROF");
    assert!(n_prof.is_some(), "the Argo file defines N_PROF");

    let temp = file.variable("/TEMP").expect("TEMP");
    assert_eq!(temp.dimensions, vec!["N_PROF", "N_LEVELS"]);
    assert_eq!(temp.dtype(), DType::Float(4));
    assert!(temp.attribute("units").is_some());
    assert!(
        temp.attribute("DIMENSION_LIST").is_none(),
        "bookkeeping attributes stay hidden"
    );
}

#[test]
fn slices_and_chunks_work_on_a_real_compressed_variable() {
    let Some(file) = argo() else { return };
    let temp = file.variable("/TEMP").expect("TEMP");

    // TEMP is chunked, shuffled and deflated in this file.
    assert!(temp.chunk_shape().is_some());
    assert!(!temp.dataset().pipeline.is_empty());

    let rows = temp.shape[0].min(2);
    let cols = temp.shape[1].min(3);
    let slice = temp.read_slice(&[0..rows, 0..cols]).unwrap();
    assert_eq!(slice.shape(), &[rows, cols]);

    // Reading every chunk must yield the same element count as the variable.
    let total: u64 = temp.chunks().iter().map(|c| c.element_count()).sum();
    assert_eq!(total, temp.element_count());

    // And a chunk read must agree with the equivalent slice read.
    let chunk = &temp.chunks()[0];
    let from_chunk = temp.read_chunk(chunk).unwrap().to_f64().unwrap();
    let ranges: Vec<_> = (0..chunk.offset.len())
        .map(|a| chunk.offset[a]..chunk.offset[a] + chunk.shape[a])
        .collect();
    let from_slice = temp.read_slice(&ranges).unwrap().to_f64().unwrap();
    assert_eq!(from_chunk, from_slice);
}

#[test]
fn every_readable_variable_of_a_real_file_round_trips_through_chunks() {
    let Some(file) = argo() else { return };

    let mut checked = 0;
    for v in file.variables() {
        if !v.is_readable() || v.shape.is_empty() {
            continue;
        }
        let whole = v.read().unwrap();
        let from_chunks: u64 = v.chunks().iter().map(|c| c.element_count()).sum();
        assert_eq!(
            whole.len() as u64,
            from_chunks,
            "{}: chunks must tile the variable",
            v.path
        );
        checked += 1;
    }
    assert!(checked > 10, "expected many readable variables");
}
