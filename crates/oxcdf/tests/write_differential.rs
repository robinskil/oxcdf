#![cfg(all(feature = "write", feature = "diff-tests"))]
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
