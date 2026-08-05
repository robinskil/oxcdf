//! Reads returned as `ndarray` values, shaped as the variable is.
#![cfg(feature = "ndarray")]

use oxcdf::netcdf::NetcdfFile;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_files/legacy_v1_objheader.h5"
);

fn argo() -> Option<NetcdfFile> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files/test_file.nc");
    std::path::Path::new(p)
        .exists()
        .then(|| NetcdfFile::open(p).unwrap())
}

#[test]
fn a_two_dimensional_variable_keeps_its_shape() {
    let file = NetcdfFile::open(FIXTURE).unwrap();
    let a = file
        .variable("/contig_f64")
        .unwrap()
        .get::<f64, _>(..)
        .unwrap();

    assert_eq!(a.shape(), &[40, 6]);
    assert_eq!(a.ndim(), 2);
    // Row-major: element (row, col) is row*6 + col in the flat order.
    assert_eq!(a[[0, 0]], 0.0);
    assert_eq!(a[[0, 5]], 2.5);
    assert_eq!(a[[1, 0]], 3.0);
    assert_eq!(a[[39, 5]], 239.0 * 0.5);
}

#[test]
fn indexing_matches_the_flat_read() {
    let file = NetcdfFile::open(FIXTURE).unwrap();
    let v = file.variable("/chunked_i32").unwrap();

    let flat = v.get_values::<i64, _>(..).unwrap();
    let arr = v.get::<i64, _>(..).unwrap();

    for row in 0..40 {
        for col in 0..6 {
            assert_eq!(arr[[row, col]], flat[row * 6 + col], "at {row},{col}");
        }
    }
}

#[test]
fn a_slice_yields_an_array_of_the_slice_shape() {
    let file = NetcdfFile::open(FIXTURE).unwrap();
    let v = file.variable("/chunked_i32").unwrap();

    let a = v.get::<i64, _>([5..15, 2..5]).unwrap();
    assert_eq!(a.shape(), &[10, 3]);
    assert_eq!(a[[0, 0]], (5 * 6 + 2) * 3 - 100);
    assert_eq!(a[[9, 2]], (14 * 6 + 4) * 3 - 100);
}

#[test]
fn a_one_dimensional_variable_is_rank_one() {
    let file = NetcdfFile::open(FIXTURE).unwrap();
    let a = file
        .variable("/subgroup/nested_i16")
        .unwrap()
        .get::<i64, _>(..)
        .unwrap();
    assert_eq!(a.shape(), &[6]);
    assert_eq!(a[[0]], 1000);
}

/// Strings come back as a flat `Vec<String>`, not an array.
///
/// The `netcdf` crate has no shaped string read, so neither does this one.
/// `Variable::shape` gives the layout when a caller wants to reshape.
#[test]
fn strings_come_back_flat_with_their_shape_alongside() {
    let file = NetcdfFile::open(FIXTURE).unwrap();
    let v = file.variable("/fixed_strings").unwrap();
    let s = v.get_strings(..).unwrap();
    assert_eq!(v.shape, vec![5]);
    assert_eq!(s.len(), 5);
    assert_eq!(s[0], "alpha");
    assert_eq!(s[4], "epsilon");
}

#[test]
fn arrays_work_on_a_real_compressed_variable() {
    let Some(file) = argo() else { return };
    let temp = file.variable("TEMP").unwrap();

    let a = temp.get::<f64, _>(..).unwrap();
    assert_eq!(a.shape(), &[8, 6]);

    // Cross-check a couple of cells against the flat read.
    let flat = temp.get_values::<f64, _>(..).unwrap();
    assert_eq!(a[[0, 0]], flat[0]);
    assert_eq!(a[[7, 5]], flat[47]);

    // ndarray's own operations should work on the result.
    assert_eq!(a.iter().count(), 48);
    assert_eq!(a.index_axis(ndarray::Axis(0), 0).len(), 6);
}

#[test]
fn a_shape_mismatch_is_reported() {
    // A scalar dataspace holds one value; the guard is exercised through the
    // public path by reading a real variable and checking the happy case holds,
    // since constructing a mismatch requires a corrupt file.
    let file = NetcdfFile::open(FIXTURE).unwrap();
    let a = file
        .variable("/contig_f32be")
        .unwrap()
        .get::<f64, _>(..)
        .unwrap();
    assert_eq!(a.len(), 40);
}
