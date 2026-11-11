//! An unlimited dimension is as long as its longest variable.
//!
//! HDF5 stores only the records a variable holds. netCDF reports the longest
//! length for every variable on the dimension, and a read past a shorter one
//! returns its fill value. The expected values come from netCDF4-python.
//!
//! `unlimited_uneven.nc` comes from `test_files/generate_unlimited_uneven.py`.

use oxcdf::netcdf::NetcdfFile;

const PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_files/unlimited_uneven.nc"
);

/// netCDF's default fill value for `short`.
const FILL_I16: i16 = -32767;

fn grid() -> Vec<i16> {
    let mut want = vec![FILL_I16; 18];
    for (i, cell) in want.iter_mut().take(6).enumerate() {
        *cell = i as i16;
    }
    want
}

#[test]
fn every_variable_on_an_unlimited_dimension_has_its_longest_length() {
    let file = NetcdfFile::open(PATH).unwrap();
    assert_eq!(file.dimension("time").unwrap().len, 6);
    for name in ["time", "empty", "short", "big", "names"] {
        assert_eq!(file.variable(name).unwrap().shape, vec![6], "{name}");
    }
    assert_eq!(file.variable("grid").unwrap().shape, vec![6, 3]);
    assert_eq!(file.variable("/child/nested").unwrap().shape, vec![6]);
}

#[test]
fn a_dimension_with_no_coordinate_variable_takes_its_longest_variable() {
    let file = NetcdfFile::open(PATH).unwrap();
    assert_eq!(file.dimension("obs").unwrap().len, 4);
    assert_eq!(file.variable("a").unwrap().shape, vec![4]);
    assert_eq!(file.variable("b").unwrap().shape, vec![4]);
}

#[test]
fn a_read_past_the_stored_records_returns_the_fill_value() {
    let file = NetcdfFile::open(PATH).unwrap();
    let var = |name: &str| file.variable(name).unwrap();

    assert_eq!(
        var("empty").get_values::<f32, _>(..).unwrap(),
        vec![-999.0; 6]
    );
    assert_eq!(
        var("short").get_values::<i32, _>(..).unwrap(),
        vec![10, 11, -1, -1, -1, -1]
    );
    assert_eq!(var("grid").get_values::<i16, _>(..).unwrap(), grid());
    assert_eq!(
        var("big").get_values::<f64, _>(..).unwrap(),
        vec![1.5, -5.0, -5.0, -5.0, -5.0, -5.0]
    );
    assert_eq!(
        var("names").get_strings(..).unwrap(),
        vec!["first", "", "", "", "", ""]
    );
    assert_eq!(var("b").get_values::<f32, _>(..).unwrap(), vec![-2.0; 4]);
    assert_eq!(
        var("/child/nested").get_values::<i32, _>(..).unwrap(),
        vec![7, -3, -3, -3, -3, -3]
    );
}

#[allow(clippy::single_range_in_vec_init)]
#[test]
fn a_selection_that_straddles_the_stored_records_is_padded() {
    let file = NetcdfFile::open(PATH).unwrap();
    let short = file.variable("short").unwrap();
    assert_eq!(
        short.get_values::<i32, _>([1..4]).unwrap(),
        vec![11, -1, -1]
    );
    let grid = file.variable("grid").unwrap();
    assert_eq!(
        grid.get_values::<i16, _>([1..3, 1..3]).unwrap(),
        vec![4, 5, FILL_I16, FILL_I16]
    );
}

#[allow(clippy::single_range_in_vec_init)]
#[test]
fn a_selection_past_the_dimension_is_still_an_error() {
    let file = NetcdfFile::open(PATH).unwrap();
    let short = file.variable("short").unwrap();
    assert!(short.get_values::<i32, _>([0..7]).is_err());
}

#[allow(clippy::single_range_in_vec_init)]
#[cfg(feature = "async")]
#[tokio::test]
async fn the_async_engine_pads_the_same_way() {
    let file = oxcdf::open_async(std::sync::Arc::new(oxcdf::SyncAsAsync(
        oxcdf::FileSource::open(PATH).unwrap(),
    )))
    .await
    .unwrap();
    let var = |name: &str| file.variable(name).unwrap();

    assert_eq!(
        var("empty").get_values::<f32, _>(..).await.unwrap(),
        vec![-999.0; 6]
    );
    assert_eq!(
        var("short").get_values::<i32, _>([1..4]).await.unwrap(),
        vec![11, -1, -1]
    );
    assert_eq!(var("grid").get_values::<i16, _>(..).await.unwrap(), grid());
    assert_eq!(
        var("big").get_values::<f64, _>(..).await.unwrap(),
        vec![1.5, -5.0, -5.0, -5.0, -5.0, -5.0]
    );
    assert_eq!(
        var("names").get_strings(..).await.unwrap(),
        vec!["first", "", "", "", "", ""]
    );
}
