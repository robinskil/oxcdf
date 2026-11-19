#![cfg(feature = "write")]
#![expect(
    clippy::single_range_in_vec_init,
    reason = "a selection of one axis is an array of one range"
)]
//! The netCDF writer makes files that the reader of this crate reads back.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p oxcdf --features write --test write_netcdf
//! ```

use std::sync::Arc;

use oxcdf::hdf5::message::ByteOrder;
use oxcdf::{AttributeValue, DType, Error, FileMut, MemorySource, NetcdfFile};

fn reopen(file: FileMut) -> NetcdfFile {
    let bytes = file.into_bytes().unwrap();
    NetcdfFile::from_source(Arc::new(MemorySource::new(bytes))).unwrap()
}

fn dimensions(file: &NetcdfFile) -> Vec<(String, u64, bool)> {
    file.dimensions()
        .iter()
        .map(|d| (d.name.clone(), d.len, d.is_unlimited))
        .collect()
}

fn bad<T>(result: Result<T, Error>) -> bool {
    matches!(result, Err(Error::BadRequest(_)))
}

#[test]
fn a_variable_reads_back_with_its_dimension_names() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("lat", 2).unwrap();
    file.add_dimension("lon", 3).unwrap();
    let mut temp = file.add_variable::<f32>("TEMP", &["lat", "lon"]).unwrap();
    temp.put_values(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], ..)
        .unwrap();

    let file = reopen(file);
    assert_eq!(
        dimensions(&file),
        [("lat".to_owned(), 2, false), ("lon".to_owned(), 3, false)]
    );
    let temp = file.variable("TEMP").unwrap();
    assert_eq!(temp.dimensions, ["lat", "lon"]);
    assert_eq!(temp.vartype(), DType::Float(4));
    assert_eq!(
        temp.get_values::<f32, _>(..).unwrap(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(file.variables().len(), 1);
}

#[test]
fn a_dimension_with_no_variable_is_not_a_variable() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("x", 4).unwrap();
    let file = reopen(file);
    let x = file.dimension("x").unwrap();
    assert_eq!((x.len, x.id, x.is_phony), (4, Some(0), false));
    assert!(!x.has_coordinate_variable);
    assert!(file.variables().is_empty());
}

#[test]
fn a_variable_with_the_name_of_its_dimension_is_the_coordinate() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("lat", 3).unwrap();
    let mut lat = file.add_variable::<f64>("lat", &["lat"]).unwrap();
    lat.put_values(&[-10.0, 0.0, 10.0], ..).unwrap();
    file.add_variable::<i32>("count", &["lat"]).unwrap();

    let file = reopen(file);
    assert!(file.dimension("lat").unwrap().has_coordinate_variable);
    let lat = file.variable("lat").unwrap();
    assert!(lat.is_coordinate);
    assert_eq!(lat.dimensions, ["lat"]);
    assert_eq!(lat.get_values::<f64, _>(..).unwrap(), [-10.0, 0.0, 10.0]);
    assert_eq!(file.variable("count").unwrap().dimensions, ["lat"]);
}

#[test]
fn a_variable_with_the_name_of_another_dimension_keeps_its_name() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("lat", 3).unwrap();
    file.add_dimension("nv", 2).unwrap();
    let mut nv = file.add_variable::<i16>("nv", &["lat"]).unwrap();
    nv.put_values(&[7i16, 8, 9], ..).unwrap();

    let file = reopen(file);
    let nv = file.variable("nv").unwrap();
    assert!(!nv.is_coordinate);
    assert_eq!(nv.dimensions, ["lat"]);
    assert_eq!(nv.get_values::<i16, _>(..).unwrap(), [7, 8, 9]);
    assert!(!file.dimension("nv").unwrap().has_coordinate_variable);
}

#[test]
fn an_unlimited_dimension_grows_with_a_write() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_unlimited_dimension("time").unwrap();
    file.add_dimension("lat", 2).unwrap();
    let mut temp = file.add_variable::<f32>("TEMP", &["time", "lat"]).unwrap();
    temp.put_values(&[1.0f32, 2.0], [0..1, 0..2]).unwrap();
    temp.put_values(&[5.0f32, 6.0], [2..3, 0..2]).unwrap();
    file.add_variable::<i32>("other", &["time"]).unwrap();
    assert_eq!(file.dimension_len("time"), Some(3));

    let file = reopen(file);
    let time = file.dimension("time").unwrap();
    assert_eq!((time.len, time.is_unlimited), (3, true));
    let temp = file.variable("TEMP").unwrap();
    assert_eq!(temp.shape, [3, 2]);
    let fill = 9.969_21e36_f32;
    assert_eq!(
        temp.get_values::<f32, _>(..).unwrap(),
        [1.0, 2.0, fill, fill, 5.0, 6.0]
    );
    let other = file.variable("other").unwrap();
    assert_eq!(other.shape, [3]);
    assert_eq!(other.get_values::<i32, _>(..).unwrap(), [-2147483647; 3]);
}

#[test]
fn a_full_selection_on_an_unlimited_axis_takes_the_dimension_length() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_unlimited_dimension("time").unwrap();
    let mut a = file.add_variable::<i32>("a", &["time"]).unwrap();
    a.put_values(&[1, 2, 3], [0..3]).unwrap();
    let mut b = file.add_variable::<i32>("b", &["time"]).unwrap();
    b.put_values(&[4, 5, 6], ..).unwrap();
    assert!(bad(b.put_values(&[4, 5], ..)));

    let file = reopen(file);
    let b = file.variable("b").unwrap();
    assert_eq!(b.get_values::<i32, _>(..).unwrap(), [4, 5, 6]);
}

#[test]
fn a_set_fill_value_reads_back_for_elements_with_no_write() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("x", 3).unwrap();
    let mut v = file.add_variable::<i16>("v", &["x"]).unwrap();
    v.set_fill_value(-5i16).unwrap();
    v.put_values(&[1i16], [1..2]).unwrap();
    file.add_variable::<u8>("unset", &["x"]).unwrap();

    let file = reopen(file);
    let v = file.variable("v").unwrap();
    assert_eq!(v.get_values::<i16, _>(..).unwrap(), [-5, 1, -5]);
    assert_eq!(v.fill_value::<i16>().unwrap(), Some(-5));
    assert!(v.attribute("_FillValue").is_some());
    let unset = file.variable("unset").unwrap();
    assert_eq!(unset.get_values::<u8, _>(..).unwrap(), [255; 3]);
}

#[test]
fn values_of_another_numeric_type_convert_to_the_variable_type() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("x", 3).unwrap();
    let mut v = file.add_variable::<i16>("v", &["x"]).unwrap();
    v.put_values(&[1.9f64, -2.0, 70000.0], ..).unwrap();
    let mut big = file.add_variable::<f64>("big", &["x"]).unwrap();
    big.set_endianness(ByteOrder::Big).unwrap();
    big.put_values(&[1i32, 2, 3], ..).unwrap();

    let file = reopen(file);
    let v = file.variable("v").unwrap();
    assert_eq!(v.get_values::<i16, _>(..).unwrap(), [1, -2, i16::MAX]);
    let big = file.variable("big").unwrap();
    assert_eq!(big.get_values::<f64, _>(..).unwrap(), [1.0, 2.0, 3.0]);
}

#[test]
fn a_compressed_variable_reads_back_and_is_smaller() {
    let values: Vec<f64> = (0..100_000).map(|i| f64::from(i / 100)).collect();
    let size = |compress: bool| {
        let mut file = oxcdf::create_in_memory().unwrap();
        file.add_dimension("x", 100_000).unwrap();
        let mut v = file.add_variable::<f64>("v", &["x"]).unwrap();
        if compress {
            v.set_chunking(&[10_000]).unwrap();
            v.set_compression(4, true).unwrap();
        }
        v.put_values(&values, ..).unwrap();
        let bytes = file.into_bytes().unwrap();
        let len = bytes.len();
        let file = NetcdfFile::from_source(Arc::new(MemorySource::new(bytes))).unwrap();
        let v = file.variable("v").unwrap();
        assert_eq!(v.get_values::<f64, _>(..).unwrap(), values);
        len
    };
    assert!(size(true) * 4 < size(false));
}

#[test]
fn compression_with_no_chunk_shape_selects_one() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("x", 50).unwrap();
    let mut v = file.add_variable::<i32>("v", &["x"]).unwrap();
    v.set_zstd(3).unwrap();
    let values: Vec<i32> = (0..50).collect();
    v.put_values(&values, ..).unwrap();
    let file = reopen(file);
    let v = file.variable("v").unwrap();
    assert_eq!(v.get_values::<i32, _>(..).unwrap(), values);
}

#[test]
fn a_definition_change_after_a_write_is_a_bad_request() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("x", 4).unwrap();
    let mut v = file.add_variable::<i32>("v", &["x"]).unwrap();
    v.put_values(&[1, 2, 3, 4], ..).unwrap();
    assert!(bad(v.set_chunking(&[2])));
    assert!(bad(v.set_compression(4, true)));
    assert!(bad(v.set_fill_value(0i32)));
    assert!(bad(v.set_endianness(ByteOrder::Big)));
    let file = reopen(file);
    let v = file.variable("v").unwrap();
    assert_eq!(v.get_values::<i32, _>(..).unwrap(), [1, 2, 3, 4]);
}

#[test]
fn a_fill_value_follows_a_later_byte_order() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("x", 3).unwrap();
    let mut v = file.add_variable::<i32>("v", &["x"]).unwrap();
    v.set_fill_value(-5i32).unwrap();
    v.set_endianness(ByteOrder::Big).unwrap();
    v.put_value(7i32, [1]).unwrap();

    let file = reopen(file);
    let v = file.variable("v").unwrap();
    assert_eq!(v.get_values::<i32, _>(..).unwrap(), [-5, 7, -5]);
    assert_eq!(v.fill_value::<i32>().unwrap(), Some(-5));
}

#[test]
fn a_second_attribute_put_replaces_the_value() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_attribute("title", "first").unwrap();
    file.add_attribute("title", 2i32).unwrap();
    let file = reopen(file);
    assert_eq!(file.attributes().len(), 1);
    let title = &file.attribute("title").unwrap().value;
    assert!(matches!(title, AttributeValue::Int(2)));
}

#[test]
fn strings_and_characters_read_back() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("station", 2).unwrap();
    file.add_dimension("len", 3).unwrap();
    let mut names = file.add_string_variable("name", &["station"]).unwrap();
    names.put_string("Texel", [0]).unwrap();
    names.put_strings(&["Den Helder"], [1..2]).unwrap();
    let mut code = file
        .add_variable_with_type("code", &["station", "len"], &DType::Char)
        .unwrap();
    code.put_values(b"TEXDHR".as_slice(), ..).unwrap();

    let file = reopen(file);
    let names = file.variable("name").unwrap();
    assert_eq!(names.vartype(), DType::String);
    assert_eq!(names.get_strings(..).unwrap(), ["Texel", "Den Helder"]);
    let code = file.variable("code").unwrap();
    assert_eq!(code.vartype(), DType::Char);
    assert_eq!(code.get_strings(..).unwrap().concat(), "TEXDHR");
}

#[test]
fn each_attribute_kind_reads_back() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_attribute("title", "Sea surface temperature")
        .unwrap();
    file.add_attribute("empty", "").unwrap();
    file.add_attribute("version", 3i32).unwrap();
    file.add_attribute("range", vec![0.5f64, 1.5]).unwrap();
    file.add_attribute("names", vec!["a".to_owned(), "bc".to_owned()])
        .unwrap();
    file.add_dimension("x", 1).unwrap();
    let mut v = file.add_variable::<u8>("v", &["x"]).unwrap();
    v.put_attribute("units", "degC").unwrap();
    v.put_attribute("scale", 0.1f32).unwrap();

    let file = reopen(file);
    let value = |name: &str| file.attribute(name).unwrap().value.clone();
    assert_eq!(value("title").as_text(), Some("Sea surface temperature"));
    assert_eq!(value("empty").as_text(), Some(""));
    assert!(matches!(value("version"), AttributeValue::Int(3)));
    assert!(matches!(value("range"), AttributeValue::Doubles(v) if v == [0.5, 1.5]));
    assert!(matches!(value("names"), AttributeValue::Strs(v) if v == ["a", "bc"]));
    assert_eq!(file.attributes().len(), 5);
    let v = file.variable("v").unwrap();
    assert_eq!(v.attribute("units").unwrap().value.as_text(), Some("degC"));
    assert!(matches!(
        v.attribute("scale").unwrap().value,
        AttributeValue::Float(s) if s == 0.1
    ));
    assert_eq!(v.attributes().len(), 2);
}

#[test]
fn a_group_uses_the_dimensions_of_its_parent() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("lat", 2).unwrap();
    let mut sub = file.add_group("sub").unwrap();
    sub.add_dimension("z", 3).unwrap();
    sub.add_attribute("kind", "deep").unwrap();
    let mut deep = sub.add_variable::<i32>("deep", &["z", "lat"]).unwrap();
    deep.put_values(&[1, 2, 3, 4, 5, 6], ..).unwrap();
    let mut inner = sub.add_group("inner").unwrap();
    inner.add_variable::<u8>("flag", &["lat"]).unwrap();
    assert!(file.group_mut("sub").is_some());
    assert!(file.group_mut("none").is_none());

    let file = reopen(file);
    let sub = file.group("/sub").unwrap();
    assert_eq!(sub.dimensions.len(), 1);
    assert_eq!(sub.dimensions[0].name, "z");
    assert_eq!(sub.attribute("kind").unwrap().value.as_text(), Some("deep"));
    let deep = file.variable("/sub/deep").unwrap();
    assert_eq!(deep.dimensions, ["z", "lat"]);
    assert_eq!(deep.get_values::<i32, _>(..).unwrap(), [1, 2, 3, 4, 5, 6]);
    let flag = file.variable("/sub/inner/flag").unwrap();
    assert_eq!(flag.dimensions, ["lat"]);
}

#[test]
fn a_dimension_in_a_group_hides_the_one_of_the_parent() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("x", 2).unwrap();
    let mut sub = file.add_group("sub").unwrap();
    sub.add_dimension("x", 5).unwrap();
    sub.add_variable::<i32>("v", &["x"]).unwrap();
    let file = reopen(file);
    assert_eq!(file.variable("/sub/v").unwrap().shape, [5]);
    assert_eq!(file.dimension_len("x"), Some(2));
}

#[test]
fn wrong_definitions_are_bad_requests() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("x", 2).unwrap();
    assert!(bad(file.add_dimension("x", 3)));
    assert!(bad(file.add_variable::<i32>("v", &["none"]).map(|_| ())));
    assert!(bad(file.add_attribute("_NCProperties", "x")));
    assert!(bad(file.add_attribute("_FillValue", 1i32)));
    assert!(bad(file.add_attribute("", 1i32)));
    file.add_variable::<i32>("v", &["x"]).unwrap();
    assert!(bad(file.add_variable::<i32>("v", &["x"]).map(|_| ())));
    assert!(bad(file.add_group("v").map(|_| ())));
    assert!(bad(file.add_group("x").map(|_| ())));
    assert!(bad(file
        .add_variable::<i32>("_nc4_non_coord_y", &["x"])
        .map(|_| ())));
    let mut v = file.variable_mut("v").unwrap();
    assert!(bad(v.put_attribute("DIMENSION_LIST", 1i32)));
    assert!(bad(v.put_attribute("_FillValue", 1i32)));
    assert!(bad(v.put_values(&[1, 2, 3], ..)));
    assert!(bad(v.put_values(&[1, 2], [0..3])));
    assert!(bad(v.put_string("a", [0])));
    assert!(bad(v.set_compression(10, false)));
    assert!(bad(v.set_chunking(&[3])));
    assert!(bad(v.set_chunking(&[1, 1])));
    v.set_zstd(3).unwrap();
    assert!(bad(v.set_compression(4, false)));
    let mut scalar = file.add_variable::<i32>("scalar", &[]).unwrap();
    assert!(bad(scalar.set_chunking(&[])));
    assert!(bad(scalar.set_fletcher32(true)));
    let mut text = file.add_string_variable("text", &["x"]).unwrap();
    let unsupported = text.set_fill_value(0u8);
    assert!(matches!(unsupported, Err(Error::Unsupported(_))));
    assert!(file.variable_mut("none").is_none());

    let file = reopen(file);
    assert_eq!(file.variables().len(), 3);
    assert_eq!(dimensions(&file), [("x".to_owned(), 2, false)]);
    assert!(file.attributes().is_empty());
}

#[test]
fn a_length_of_zero_gives_an_unlimited_dimension() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("t", 0).unwrap();
    let file = reopen(file);
    let t = file.dimension("t").unwrap();
    assert_eq!((t.len, t.is_unlimited), (0, true));
}

#[test]
fn a_scalar_variable_reads_back() {
    let mut file = oxcdf::create_in_memory().unwrap();
    let mut v = file.add_variable::<f64>("v", &[]).unwrap();
    v.put_value(2.5f64, ..).unwrap();
    let file = reopen(file);
    let v = file.variable("v").unwrap();
    assert!(v.shape.is_empty());
    assert_eq!(v.get_values::<f64, _>(..).unwrap(), [2.5]);
}

#[test]
fn a_file_appears_at_its_path_only_after_close() {
    let dir = std::env::temp_dir().join(format!("oxcdf-write-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("out.nc");
    let mut file = oxcdf::create(&path).unwrap();
    file.add_dimension("x", 2).unwrap();
    let mut v = file.add_variable::<i32>("v", &["x"]).unwrap();
    v.put_values(&[4, 5], ..).unwrap();
    assert!(!path.exists());
    file.close().unwrap();

    let read = oxcdf::open(&path).unwrap();
    let v = read.variable("v").unwrap();
    assert_eq!(v.get_values::<i32, _>(..).unwrap(), [4, 5]);
    drop(read);
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);

    let dropped = dir.join("dropped.nc");
    let mut file = oxcdf::create(&dropped).unwrap();
    file.add_dimension("x", 2).unwrap();
    drop(file);
    assert_eq!(oxcdf::open(&dropped).unwrap().dimension_len("x"), Some(2));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_failed_write_changes_nothing() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_unlimited_dimension("time").unwrap();
    let mut name = file.add_string_variable("name", &["time"]).unwrap();
    assert!(bad(name.put_string("a\0b", [4])));
    name.set_chunking(&[8]).unwrap();
    assert_eq!(file.dimension_len("time"), Some(0));

    let file = reopen(file);
    assert_eq!(file.dimension_len("time"), Some(0));
    assert_eq!(file.variable("name").unwrap().shape, [0]);
}

#[test]
fn a_panic_puts_no_file_at_the_path() {
    let path = std::env::temp_dir().join(format!("oxcdf-panic-{}.nc", std::process::id()));
    let mut temporary = path.clone().into_os_string();
    temporary.push(".tmp");
    let target = path.clone();
    let result = std::thread::spawn(move || {
        let mut file = oxcdf::create(&target).unwrap();
        file.add_dimension("x", 2).unwrap();
        panic!("the caller stops before the close");
    })
    .join();
    assert!(result.is_err());
    assert!(!path.exists());
    assert!(!std::path::Path::new(&temporary).exists());
}

#[test]
fn wrong_options_leave_no_file() {
    let path = std::env::temp_dir().join(format!("oxcdf-options-{}.nc", std::process::id()));
    let mut temporary = path.clone().into_os_string();
    temporary.push(".tmp");
    let options = oxcdf::WriteOptions::default().header_slack_bytes(1 << 20);
    assert!(bad(oxcdf::create_with(&path, options)));
    assert!(!path.exists());
    assert!(!std::path::Path::new(&temporary).exists());
}

#[test]
fn a_file_is_send() {
    fn send<T: Send>() {}
    send::<FileMut>();
    send::<oxcdf::GroupMut<'_>>();
    send::<oxcdf::VariableMut<'_>>();
}

#[test]
fn bytes_of_a_file_on_disk_are_a_bad_request() {
    let path = std::env::temp_dir().join(format!("oxcdf-bytes-{}.nc", std::process::id()));
    let file = oxcdf::create(&path).unwrap();
    assert!(bad(file.into_bytes()));
    assert!(!path.exists());
}

/// Write the file that `scripts/check_writer.py` gives to other libraries.
///
/// The script sets `OXCDF_EXPORT_DIR` and compares the file with this content.
#[test]
#[ignore = "run by scripts/check_writer.py"]
fn export_a_netcdf_file_for_the_external_check() {
    let dir = std::env::var("OXCDF_EXPORT_DIR").expect("OXCDF_EXPORT_DIR names a directory");
    let path = std::path::Path::new(&dir).join("netcdf.nc");
    let mut file = oxcdf::create(&path).unwrap();
    file.add_unlimited_dimension("time").unwrap();
    file.add_dimension("lat", 3).unwrap();
    file.add_dimension("lon", 4).unwrap();
    file.add_dimension("nv", 2).unwrap();
    file.add_attribute("title", "check").unwrap();
    file.add_attribute("levels", vec![1i32, 2, 3]).unwrap();
    file.add_attribute("tags", vec!["a".to_owned(), "bc".to_owned()])
        .unwrap();

    let mut lat = file.add_variable::<f32>("lat", &["lat"]).unwrap();
    lat.put_attribute("units", "degrees_north").unwrap();
    lat.put_values(&[-10.0f32, 0.0, 10.0], ..).unwrap();
    let mut time = file.add_variable::<f64>("time", &["time"]).unwrap();
    time.put_values(&[0.5f64, 1.5], [0..2]).unwrap();

    let mut temp = file
        .add_variable::<f32>("TEMP", &["time", "lat", "lon"])
        .unwrap();
    temp.set_chunking(&[1, 3, 4]).unwrap();
    temp.set_compression(4, true).unwrap();
    temp.set_fill_value(-999.0f32).unwrap();
    temp.put_attribute("units", "degC").unwrap();
    let first: Vec<f32> = (0..12).map(|i| i as f32).collect();
    temp.put_values(&first, [0..1, 0..3, 0..4]).unwrap();
    temp.put_values(&[50.0f32, 51.0], [2..3, 1..2, 0..2])
        .unwrap();

    file.add_variable::<f64>("PSAL", &["time", "lat"]).unwrap();
    let mut code = file
        .add_variable_with_type("code", &["lat", "nv"], &DType::Char)
        .unwrap();
    code.put_values(b"abcdef".as_slice(), ..).unwrap();
    let mut name = file.add_string_variable("name", &["lat"]).unwrap();
    name.put_strings(&["south", "equator"], [0..2]).unwrap();
    let mut scalar = file.add_variable::<i32>("scalar", &[]).unwrap();
    scalar.put_value(42i32, ..).unwrap();
    let mut nv = file.add_variable::<i16>("nv", &["lat"]).unwrap();
    nv.put_values(&[7i16, 8, 9], ..).unwrap();
    let mut big = file.add_variable::<u16>("big", &["lon"]).unwrap();
    big.set_endianness(ByteOrder::Big).unwrap();
    big.set_zstd(3).unwrap();
    big.put_values(&[1u16, 2, 3, 4], ..).unwrap();

    let mut sub = file.add_group("sub").unwrap();
    sub.add_dimension("z", 5).unwrap();
    sub.add_attribute("kind", "deep").unwrap();
    let mut deep = sub.add_variable::<i32>("deep", &["z", "lat"]).unwrap();
    deep.put_values(&(0..15).collect::<Vec<i32>>(), ..).unwrap();
    file.close().unwrap();
}

/// Read the exported file after netcdf-c changed it in `scripts/check_writer.py`.
#[test]
#[ignore = "run by scripts/check_writer.py"]
fn a_file_changed_by_netcdf_c_reads_back() {
    let dir = std::env::var("OXCDF_EXPORT_DIR").expect("OXCDF_EXPORT_DIR names a directory");
    let file = oxcdf::open(std::path::Path::new(&dir).join("netcdf.nc")).unwrap();
    assert_eq!(file.dimension_len("time"), Some(4));
    assert_eq!(file.dimension_len("depth"), Some(2));

    let temp = file.variable("TEMP").unwrap();
    assert_eq!(temp.shape, [4, 3, 4]);
    let values = temp.get_values::<f32, _>(..).unwrap();
    let first: Vec<f32> = (0..12).map(|i| i as f32).collect();
    assert_eq!(values[..12], first);
    assert_eq!(values[12..24], [-999.0; 12]);
    assert_eq!(values[28..30], [50.0, 51.0]);
    assert_eq!(values[36..], [7.0; 12]);
    assert_eq!(temp.attributes().len(), 14);
    assert_eq!(
        temp.attribute("units").unwrap().value.as_text(),
        Some("degC")
    );
    assert!(matches!(
        temp.attribute("extra_11").unwrap().value,
        AttributeValue::Double(v) if v == 11.0
    ));

    let time = file.variable("time").unwrap();
    assert_eq!(time.get_values::<f64, _>(..).unwrap(), [0.5, 1.5, 2.5, 3.5]);
    let name = file.variable("name").unwrap();
    assert_eq!(name.get_strings(..).unwrap(), ["south", "equator", "north"]);
    let extra = file.variable("extra").unwrap();
    assert_eq!(extra.dimensions, ["time", "depth"]);
    assert_eq!(
        extra.get_values::<i32, _>(..).unwrap(),
        (0..8).collect::<Vec<i32>>()
    );
    let more = file.variable("/sub/more").unwrap();
    assert_eq!(more.dimensions, ["z"]);
    assert_eq!(
        more.get_values::<f64, _>(..).unwrap(),
        [0.0, 0.5, 1.0, 1.5, 2.0]
    );
    let history = file.attribute("history").unwrap();
    assert_eq!(history.value.as_text(), Some("changed by netcdf-c"));
    assert_eq!(
        file.attribute("title").unwrap().value.as_text(),
        Some("check")
    );
}

#[test]
fn a_coordinate_variable_with_two_dimensions_reads_back() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("station", 2).unwrap();
    file.add_dimension("len", 3).unwrap();
    let mut station = file
        .add_variable_with_type("station", &["station", "len"], &DType::Char)
        .unwrap();
    station.put_values(b"TEXDHR".as_slice(), ..).unwrap();
    let mut square = file.add_variable::<i32>("square", &["len", "len"]).unwrap();
    square
        .put_values(&(0..9).collect::<Vec<i32>>(), ..)
        .unwrap();

    let file = reopen(file);
    let station = file.variable("station").unwrap();
    assert!(station.is_coordinate);
    assert_eq!(station.dimensions, ["station", "len"]);
    assert_eq!(station.get_strings(..).unwrap().concat(), "TEXDHR");
    let square = file.variable("square").unwrap();
    assert_eq!(square.dimensions, ["len", "len"]);
    assert_eq!(dimensions(&file).len(), 2);
}

#[test]
fn a_dimension_can_come_after_a_variable_of_its_name() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("y", 2).unwrap();
    let mut x = file.add_variable::<i32>("x", &["y"]).unwrap();
    x.put_values(&[4, 5], ..).unwrap();
    file.add_dimension("x", 3).unwrap();
    let mut x = file.variable_mut("x").unwrap();
    x.put_attribute("units", "m").unwrap();
    file.add_variable::<u8>("on_x", &["x"]).unwrap();

    let file = reopen(file);
    assert_eq!(
        dimensions(&file),
        [("y".to_owned(), 2, false), ("x".to_owned(), 3, false)]
    );
    assert!(!file.dimension("x").unwrap().has_coordinate_variable);
    let x = file.variable("x").unwrap();
    assert!(!x.is_coordinate);
    assert_eq!(x.dimensions, ["y"]);
    assert_eq!(x.get_values::<i32, _>(..).unwrap(), [4, 5]);
    assert_eq!(x.attribute("units").unwrap().value.as_text(), Some("m"));
    assert_eq!(file.variable("on_x").unwrap().dimensions, ["x"]);
}

#[test]
fn names_follow_the_rules_of_netcdf_c() {
    let mut file = oxcdf::create_in_memory().unwrap();
    for name in ["", ".", "-a", " a", "a ", "a/b", "a\u{7}b", "a\u{7f}"] {
        assert!(bad(file.add_dimension(name, 1)), "{name:?}");
        assert!(bad(file.add_group(name).map(|_| ())), "{name:?}");
        assert!(
            bad(file.add_variable::<i32>(name, &[]).map(|_| ())),
            "{name:?}"
        );
    }
    for name in ["a b", "_x", "9x", "température", "a.b-c+d@e"] {
        file.add_variable::<i32>(name, &[]).unwrap();
    }
    let file = reopen(file);
    assert_eq!(file.variables().len(), 5);
    assert!(file.variable("température").is_some());
}

#[test]
fn a_part_of_a_compressed_chunk_takes_a_second_write() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_unlimited_dimension("t").unwrap();
    let mut v = file.add_variable::<i32>("v", &["t"]).unwrap();
    v.set_chunking(&[2]).unwrap();
    v.set_compression(1, false).unwrap();
    v.put_values(&[1, 2], [0..2]).unwrap();
    // The chunk at 0 is in the file with its filter. The writer reads it back.
    v.put_values(&[7, 8, 9], [1..4]).unwrap();
    assert_eq!(file.dimension_len("t"), Some(4));

    let file = reopen(file);
    let v = file.variable("v").unwrap();
    assert_eq!(v.get_values::<i32, _>(..).unwrap(), [1, 7, 8, 9]);
}

#[test]
fn shuffle_works_alone_and_with_zstandard() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("x", 6).unwrap();
    let values: Vec<i32> = (0..6).collect();
    let mut alone = file.add_variable::<i32>("alone", &["x"]).unwrap();
    alone.set_shuffle(true).unwrap();
    alone.put_values(&values, ..).unwrap();
    let mut both = file.add_variable::<i32>("both", &["x"]).unwrap();
    both.set_shuffle(true).unwrap();
    both.set_zstd(3).unwrap();
    both.put_values(&values, ..).unwrap();
    let mut scalar = file.add_variable::<i32>("scalar", &[]).unwrap();
    assert!(bad(scalar.set_shuffle(true)));

    let file = reopen(file);
    let ids = |name: &str| -> Vec<u16> {
        let dataset = file.variable(name).unwrap().dataset().unwrap();
        dataset.pipeline.filters.iter().map(|f| f.id).collect()
    };
    assert_eq!(ids("alone"), [2]);
    assert_eq!(ids("both"), [2, 32015]);
    for name in ["alone", "both"] {
        let v = file.variable(name).unwrap();
        assert_eq!(v.get_values::<i32, _>(..).unwrap(), values);
    }
}

#[test]
fn the_fill_value_of_a_char_variable_is_scalar_text() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_dimension("x", 3).unwrap();
    let mut code = file
        .add_variable_with_type("code", &["x"], &DType::Char)
        .unwrap();
    code.set_fill_value(b'#').unwrap();
    code.put_values(b"a".as_slice(), [1..2]).unwrap();
    let mut number = file.add_variable::<i16>("number", &["x"]).unwrap();
    number.set_fill_value(-1i16).unwrap();

    let file = reopen(file);
    let code = file.variable("code").unwrap();
    assert_eq!(code.get_strings(..).unwrap().concat(), "#a#");
    let dims = |name: &str| {
        let dataset = file.variable(name).unwrap().dataset().unwrap();
        dataset
            .attribute("_FillValue")
            .unwrap()
            .dataspace
            .dims
            .clone()
    };
    // netcdf-c writes a scalar for `char` and one element for a number.
    assert!(dims("code").is_empty());
    assert_eq!(dims("number"), [1]);
}
