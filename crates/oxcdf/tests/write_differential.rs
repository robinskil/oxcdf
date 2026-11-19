#![cfg(all(feature = "write", feature = "diff-tests"))]
#![allow(clippy::single_range_in_vec_init)]
//! netcdf-c reads a file from the HDF5 writer.
//!
//! The root holds more than eight compact attributes and more than eight links.
//! The header of the group `wide` is larger than the metadata block.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p oxcdf --features write,diff-tests --test write_differential
//! ```

use oxcdf::hdf5::message::ByteOrder;
use oxcdf::hdf5::write::{DatasetSpec, FileSink, Hdf5Writer, WriteOptions};
use oxcdf::hdf5::{DType, Hyperslab};

#[test]
fn netcdf_c_reads_a_file_from_the_writer() {
    let path = std::env::temp_dir().join(format!("oxcdf-diff-{}.nc", std::process::id()));
    let mut w =
        Hdf5Writer::create(FileSink::create(&path).unwrap(), WriteOptions::default()).unwrap();
    let root = w.root();
    w.put_text_attribute(root, "title", "writer test").unwrap();
    for i in 0..20i32 {
        w.put_numeric_attribute(root, &format!("attr_{i:02}"), &[i])
            .unwrap();
    }
    for i in 0..20 {
        let spec = DatasetSpec::numeric(DType::Float(8), ByteOrder::Little, &[3, 4]).unwrap();
        let d = w.add_dataset(root, &format!("var_{i:02}"), spec).unwrap();
        let values: Vec<f64> = (0..12).map(|k| (i * 100 + k) as f64 * 0.5).collect();
        w.write(d, &Hyperslab::all(&[3, 4]), &values).unwrap();
        w.put_text_attribute(d, "units", "m").unwrap();
    }
    let g = w.add_group(root, "sub").unwrap();
    let d = w
        .add_dataset(
            g,
            "big_endian",
            DatasetSpec::numeric(DType::Int(4), ByteOrder::Big, &[5]).unwrap(),
        )
        .unwrap();
    w.write(d, &Hyperslab::all(&[5]), &[1i32, -2, 3, -4, 5])
        .unwrap();
    let wide = w.add_group(root, "wide").unwrap();
    for i in 0..2000 {
        w.add_group(wide, &format!("group_{i:04}")).unwrap();
    }
    w.close().unwrap();

    let file = netcdf::open(&path).unwrap();
    assert_eq!(file.variables().count(), 20);
    assert_eq!(file.attributes().count(), 21);
    for i in 0..20 {
        let var = file.variable(&format!("var_{i:02}")).unwrap();
        let expected: Vec<f64> = (0..12).map(|k| (i * 100 + k) as f64 * 0.5).collect();
        assert_eq!(var.get_values::<f64, _>(..).unwrap(), expected);
        assert_eq!(
            var.dimensions().iter().map(|d| d.len()).collect::<Vec<_>>(),
            [3, 4]
        );
    }
    let title = file.attribute("title").unwrap().value().unwrap();
    assert!(matches!(title, netcdf::AttributeValue::Str(ref s) if s == "writer test"));
    let attr_19 = file.attribute("attr_19").unwrap().value().unwrap();
    assert!(matches!(attr_19, netcdf::AttributeValue::Int(19)));
    let wide = file.group("wide").unwrap().unwrap();
    assert_eq!(wide.groups().count(), 2000);
    let sub = file.group("sub").unwrap().unwrap();
    assert_eq!(
        sub.variable("big_endian")
            .unwrap()
            .get_values::<i32, _>(..)
            .unwrap(),
        [1, -2, 3, -4, 5]
    );
    // Windows cannot remove a file that netcdf-c holds open.
    drop(file);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn netcdf_c_reads_a_file_from_the_netcdf_writer() {
    let path = std::env::temp_dir().join(format!("oxcdf-diff-nc-{}.nc", std::process::id()));
    let mut file = oxcdf::create(&path).unwrap();
    file.add_unlimited_dimension("time").unwrap();
    file.add_dimension("lat", 3).unwrap();
    file.add_dimension("nv", 2).unwrap();
    file.add_attribute("title", "writer test").unwrap();
    file.add_attribute("empty", "").unwrap();
    file.add_attribute("range", vec![0.5f64, 1.5]).unwrap();
    file.add_attribute("tags", vec!["a".to_owned(), "bc".to_owned()])
        .unwrap();
    let mut lat = file.add_variable::<f32>("lat", &["lat"]).unwrap();
    lat.put_values(&[-10.0f32, 0.0, 10.0], ..).unwrap();
    let mut temp = file.add_variable::<f32>("TEMP", &["time", "lat"]).unwrap();
    temp.set_compression(4, true).unwrap();
    temp.set_fill_value(-999.0f32).unwrap();
    temp.put_attribute("units", "degC").unwrap();
    temp.put_values(&[1.0f32, 2.0, 3.0], [0..1, 0..3]).unwrap();
    temp.put_values(&[9.0f32], [2..3, 1..2]).unwrap();
    file.add_variable::<i32>("count", &["time"]).unwrap();
    let mut nv = file.add_variable::<i16>("nv", &["lat"]).unwrap();
    nv.put_values(&[7i16, 8, 9], ..).unwrap();
    let mut code = file
        .add_variable_with_type("code", &["lat", "nv"], &DType::Char)
        .unwrap();
    code.put_values(b"abcdef".as_slice(), ..).unwrap();
    let mut name = file.add_string_variable("name", &["lat"]).unwrap();
    name.put_strings(&["south", "equator"], [0..2]).unwrap();
    let mut sub = file.add_group("sub").unwrap();
    sub.add_dimension("z", 2).unwrap();
    let mut deep = sub.add_variable::<i32>("deep", &["z", "lat"]).unwrap();
    deep.put_values(&[1, 2, 3, 4, 5, 6], ..).unwrap();
    file.close().unwrap();

    let file = netcdf::open(&path).unwrap();
    let dims: Vec<(String, usize, bool)> = file
        .dimensions()
        .map(|d| (d.name(), d.len(), d.is_unlimited()))
        .collect();
    assert_eq!(
        dims,
        [
            ("time".to_owned(), 3, true),
            ("lat".to_owned(), 3, false),
            ("nv".to_owned(), 2, false)
        ]
    );
    let mut names: Vec<String> = file.variables().map(|v| v.name()).collect();
    names.sort();
    assert_eq!(names, ["TEMP", "code", "count", "lat", "name", "nv"]);
    let dim_names = |name: &str| -> Vec<String> {
        let var = file.variable(name).unwrap();
        var.dimensions().iter().map(|d| d.name()).collect()
    };
    assert_eq!(dim_names("TEMP"), ["time", "lat"]);
    assert_eq!(dim_names("lat"), ["lat"]);
    assert_eq!(dim_names("nv"), ["lat"]);
    assert_eq!(dim_names("code"), ["lat", "nv"]);

    let lat = file.variable("lat").unwrap();
    assert_eq!(lat.get_values::<f32, _>(..).unwrap(), [-10.0, 0.0, 10.0]);
    let temp = file.variable("TEMP").unwrap();
    assert_eq!(
        temp.get_values::<f32, _>(..).unwrap(),
        [1.0, 2.0, 3.0, -999.0, -999.0, -999.0, -999.0, 9.0, -999.0]
    );
    assert_eq!(temp.fill_value::<f32>().unwrap(), Some(-999.0));
    let units = temp.attribute("units").unwrap().value().unwrap();
    assert!(matches!(units, netcdf::AttributeValue::Str(ref s) if s == "degC"));
    let count = file.variable("count").unwrap();
    assert_eq!(count.get_values::<i32, _>(..).unwrap(), [-2147483647; 3]);
    let nv = file.variable("nv").unwrap();
    assert_eq!(nv.get_values::<i16, _>(..).unwrap(), [7, 8, 9]);
    let name = file.variable("name").unwrap();
    assert_eq!(name.get_string([0]).unwrap(), "south");
    assert_eq!(name.get_string([1]).unwrap(), "equator");
    assert_eq!(name.get_string([2]).unwrap(), "");
    let code = file.variable("code").unwrap();
    assert_eq!(code.get_raw_values(..).unwrap(), b"abcdef");

    let title = file.attribute("title").unwrap().value().unwrap();
    assert!(matches!(title, netcdf::AttributeValue::Str(ref s) if s == "writer test"));
    let empty = file.attribute("empty").unwrap().value().unwrap();
    assert!(matches!(empty, netcdf::AttributeValue::Str(ref s) if s.is_empty()));
    let range = file.attribute("range").unwrap().value().unwrap();
    assert!(matches!(range, netcdf::AttributeValue::Doubles(ref v) if v == &[0.5, 1.5]));
    let tags = file.attribute("tags").unwrap().value().unwrap();
    assert!(matches!(tags, netcdf::AttributeValue::Strs(ref v) if v == &["a", "bc"]));
    assert_eq!(file.attributes().count(), 4);

    let sub = file.group("sub").unwrap().unwrap();
    let deep = sub.variable("deep").unwrap();
    let deep_dims: Vec<String> = deep.dimensions().iter().map(|d| d.name()).collect();
    assert_eq!(deep_dims, ["z", "lat"]);
    assert_eq!(deep.get_values::<i32, _>(..).unwrap(), [1, 2, 3, 4, 5, 6]);
    // Windows cannot remove a file that netcdf-c holds open.
    drop(file);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn netcdf_c_reads_a_file_from_netcdf_c_after_an_append() {
    let path = std::env::temp_dir().join(format!("oxcdf-diff-append-{}.nc", std::process::id()));
    {
        let mut file = netcdf::create(&path).unwrap();
        file.add_unlimited_dimension("time").unwrap();
        file.add_dimension("lat", 3).unwrap();
        file.add_attribute("title", "from netcdf-c").unwrap();
        // More than eight attributes or links go to dense storage in netcdf-c.
        for i in 0..12i32 {
            file.add_attribute(&format!("attr_{i:02}"), i).unwrap();
        }
        let mut lat = file.add_variable::<f32>("lat", &["lat"]).unwrap();
        lat.put_values(&[-10.0f32, 0.0, 10.0], ..).unwrap();
        let mut temp = file.add_variable::<f32>("TEMP", &["time", "lat"]).unwrap();
        temp.set_chunking(&[2, 3]).unwrap();
        temp.set_compression(4, true).unwrap();
        temp.set_fill_value(-999.0f32).unwrap();
        temp.put_attribute("units", "degC").unwrap();
        temp.put_values(&[1.0f32, 2.0, 3.0], [0..1, 0..3]).unwrap();
        let mut name = file.add_string_variable("name", &["lat"]).unwrap();
        name.put_string("south", [0]).unwrap();
        for i in 0..20i32 {
            let mut var = file
                .add_variable::<i32>(&format!("var_{i:02}"), &["lat"])
                .unwrap();
            var.put_values(&[i, i + 1, i + 2], ..).unwrap();
        }
        let mut sub = file.add_group("sub").unwrap();
        sub.add_dimension("z", 2).unwrap();
        let mut deep = sub.add_variable::<i32>("deep", &["z", "lat"]).unwrap();
        deep.put_values(&[1, 2, 3, 4, 5, 6], ..).unwrap();
    }

    let mut file = oxcdf::append(&path).unwrap();
    assert_eq!(file.dimension_len("time"), Some(1));
    let mut temp = file.variable_mut("TEMP").unwrap();
    temp.put_values(&[4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0], [1..3, 0..3])
        .unwrap();
    temp.put_attribute("long_name", "temperature").unwrap();
    let mut name = file.variable_mut("name").unwrap();
    name.put_string("north", [2]).unwrap();
    file.add_attribute("history", "appended by oxcdf").unwrap();
    file.add_attribute("attr_03", 33i32).unwrap();
    file.add_dimension("depth", 2).unwrap();
    let mut psal = file
        .add_variable::<f64>("PSAL", &["time", "depth"])
        .unwrap();
    psal.put_values(&[0.5f64, 1.5, 2.5, 3.5, 4.5, 5.5], ..)
        .unwrap();
    let mut sub = file.group_mut("sub").unwrap();
    let mut more = sub.add_variable::<i16>("more", &["z"]).unwrap();
    more.put_values(&[7i16, 8], ..).unwrap();
    file.close().unwrap();

    let file = netcdf::open(&path).unwrap();
    let dims: Vec<(String, usize, bool)> = file
        .dimensions()
        .map(|d| (d.name(), d.len(), d.is_unlimited()))
        .collect();
    assert_eq!(
        dims,
        [
            ("time".to_owned(), 3, true),
            ("lat".to_owned(), 3, false),
            ("depth".to_owned(), 2, false)
        ]
    );
    assert_eq!(file.variables().count(), 24);
    assert_eq!(file.attributes().count(), 14);
    let attr_03 = file.attribute("attr_03").unwrap().value().unwrap();
    assert!(matches!(attr_03, netcdf::AttributeValue::Int(33)));
    let attr_11 = file.attribute("attr_11").unwrap().value().unwrap();
    assert!(matches!(attr_11, netcdf::AttributeValue::Int(11)));
    let history = file.attribute("history").unwrap().value().unwrap();
    assert!(matches!(history, netcdf::AttributeValue::Str(ref s) if s == "appended by oxcdf"));

    let temp = file.variable("TEMP").unwrap();
    let temp_dims: Vec<String> = temp.dimensions().iter().map(|d| d.name()).collect();
    assert_eq!(temp_dims, ["time", "lat"]);
    assert_eq!(
        temp.get_values::<f32, _>(..).unwrap(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
    assert_eq!(temp.fill_value::<f32>().unwrap(), Some(-999.0));
    assert_eq!(temp.attributes().count(), 3);
    let lat = file.variable("lat").unwrap();
    assert_eq!(lat.get_values::<f32, _>(..).unwrap(), [-10.0, 0.0, 10.0]);
    let name = file.variable("name").unwrap();
    assert_eq!(name.get_string([0]).unwrap(), "south");
    assert_eq!(name.get_string([2]).unwrap(), "north");
    for i in 0..20i32 {
        let var = file.variable(&format!("var_{i:02}")).unwrap();
        assert_eq!(var.get_values::<i32, _>(..).unwrap(), [i, i + 1, i + 2]);
        let var_dims: Vec<String> = var.dimensions().iter().map(|d| d.name()).collect();
        assert_eq!(var_dims, ["lat"]);
    }
    let psal = file.variable("PSAL").unwrap();
    let psal_dims: Vec<String> = psal.dimensions().iter().map(|d| d.name()).collect();
    assert_eq!(psal_dims, ["time", "depth"]);
    assert_eq!(
        psal.get_values::<f64, _>(..).unwrap(),
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    );
    let sub = file.group("sub").unwrap().unwrap();
    let deep = sub.variable("deep").unwrap();
    assert_eq!(deep.get_values::<i32, _>(..).unwrap(), [1, 2, 3, 4, 5, 6]);
    let more = sub.variable("more").unwrap();
    assert_eq!(more.get_values::<i16, _>(..).unwrap(), [7, 8]);
    // Windows cannot remove a file that netcdf-c holds open.
    drop(file);

    // netcdf-c appends to the file again, and this reader reads the result.
    {
        let mut file = netcdf::append(&path).unwrap();
        let mut temp = file.variable_mut("TEMP").unwrap();
        temp.put_values(&[10.0f32, 11.0, 12.0], [3..4, 0..3])
            .unwrap();
    }
    let file = oxcdf::open(&path).unwrap();
    let temp = file.variable("TEMP").unwrap();
    assert_eq!(temp.shape, [4, 3]);
    assert_eq!(
        temp.get_values::<f32, _>([3..4, 0..3]).unwrap(),
        [10.0, 11.0, 12.0]
    );
    drop(file);
    std::fs::remove_file(&path).unwrap();
}
