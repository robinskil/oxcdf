//! Exercises the netCDF interface: navigate to a variable, read its metadata
//! and attributes, then read all of it or a slice of it.
//!
//! The chunk grid moved to `oxcdf-hdf5`, so its tests live in that crate's
//! `chunk_grid.rs`.

use oxcdf::netcdf::{DType, NetcdfFile};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_files/legacy_v1_objheader.h5"
);

fn argo() -> Option<NetcdfFile> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files/test_file.nc");
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
    assert_eq!(v.vartype(), DType::Float(8));
    assert!(v.vartype().is_float());
    assert_eq!(v.len(), 240);
    assert!(v.is_readable());

    assert_eq!(
        file.variable("/chunked_i32").unwrap().vartype(),
        DType::Int(4)
    );
    assert_eq!(
        file.variable("/fixed_strings").unwrap().vartype(),
        // A fixed string wider than one byte is not a netCDF type. netcdf-c
        // never writes one; this fixture comes from plain HDF5.
        DType::FixedString(8)
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
    let v = file.variable("/contig_f64").unwrap();

    assert_eq!(v.shape, vec![40, 6]);
    assert_eq!(v.len(), 240);
    let f = v.get_values::<f64, _>(..).unwrap();
    assert_eq!(f.len(), 240);
    assert_eq!(f[0], 0.0);
    assert_eq!(f[239], 239.0 * 0.5);
}

#[test]
fn reads_a_slice_given_ranges_per_axis() {
    let file = NetcdfFile::open(FIXTURE).unwrap();
    let v = file.variable("/chunked_i32").unwrap();

    let values = v.get_values::<i64, _>([5..15, 2..5]).unwrap();
    assert_eq!(values.len(), 30);

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
    let (lo, hi) = (0usize, 4usize);

    // One range for a rank-2 variable.
    let wrong_rank: Vec<std::ops::Range<usize>> = (0..1).map(|_| lo..hi).collect();
    assert!(
        v.get_values::<i64, _>(wrong_rank).is_err(),
        "rank must match"
    );

    let reversed = vec![hi..lo, lo..2];
    assert!(
        v.get_values::<i64, _>(reversed).is_err(),
        "range must not reverse"
    );

    let out_of_bounds = vec![lo..99, lo..2];
    assert!(
        v.get_values::<i64, _>(out_of_bounds).is_err(),
        "must stay in bounds"
    );
}

#[test]
fn reads_strings_and_nested_group_variables() {
    let file = NetcdfFile::open(FIXTURE).unwrap();

    let s = file.variable("/fixed_strings").unwrap();
    assert_eq!(
        s.get_strings(..).unwrap(),
        vec!["alpha", "beta", "gamma", "delta", "epsilon"]
    );

    let n = file.variable("/subgroup/nested_i16").unwrap();
    assert_eq!(
        n.get_values::<i64, _>(..).unwrap(),
        (1000..1006).collect::<Vec<_>>()
    );
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
    assert_eq!(temp.vartype(), DType::Float(4));
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
    assert!(temp.chunking().unwrap().is_some());
    assert!(!temp.dataset().unwrap().pipeline.is_empty());

    let rows = temp.shape[0].min(2) as usize;
    let cols = temp.shape[1].min(3) as usize;
    let slice = temp.get_values::<f64, _>([0..rows, 0..cols]).unwrap();
    assert_eq!(slice.len(), rows * cols);

    // A whole read and a slice read must agree on the overlapping corner.
    let whole = temp.get_values::<f64, _>(..).unwrap();
    assert_eq!(whole.len() as u64, temp.len());
    assert_eq!(slice[0], whole[0]);
}
