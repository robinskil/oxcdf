//! Files that declare a netCDF user-defined type.
//!
//! netCDF-4 stores a compound, enum, opaque or vlen type as an HDF5 committed
//! datatype: an object of its own, beside the variables that use it. Two things
//! follow, and both are checked here.
//!
//! The type object is not a group. It holds no values, so reporting it as one
//! would invent a group the file does not have.
//!
//! A variable of that type does not hold the type inline. Its datatype message
//! points at the committed object, and the reader has to follow the pointer.
//! `committed_datatype.rs` in `oxcdf-hdf5` covers that pointer byte by byte.

use oxcdf::netcdf::{DType, NetcdfFile};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_files/committed_types.nc"
);

fn fixture() -> NetcdfFile {
    NetcdfFile::open(FIXTURE).unwrap()
}

#[test]
fn a_committed_datatype_is_not_reported_as_a_group() {
    let file = fixture();

    // `ncdump` reports no group for this file. The four type objects
    // (reading_t, quality_t, tag_t, profile_t) must not appear as groups.
    assert!(
        file.groups().is_empty(),
        "committed datatypes leaked in as groups: {:?}",
        file.groups().iter().map(|g| &g.name).collect::<Vec<_>>()
    );
}

#[test]
fn every_variable_of_a_user_defined_type_is_indexed() {
    let file = fixture();

    let mut names: Vec<_> = file.variables().iter().map(|v| v.name.clone()).collect();
    names.sort();

    // The same five `ncdump` lists: one for each declared type, plus `plain`.
    // The dimension `n` has no coordinate variable, so it is not among them.
    assert_eq!(names, ["plain", "profiles", "quality", "readings", "tags"]);
}

#[test]
fn reports_the_type_of_each_user_defined_variable() {
    let file = fixture();

    // A compound, an enum and an opaque have no netCDF scalar type, so the
    // summary reports `Other` rather than guessing.
    assert_eq!(file.variable("/readings").unwrap().vartype(), DType::Other);
    assert_eq!(file.variable("/quality").unwrap().vartype(), DType::Other);
    assert_eq!(file.variable("/tags").unwrap().vartype(), DType::Other);

    // A vlen of float is modelled, because the reader can read it.
    assert_eq!(
        file.variable("/profiles").unwrap().vartype(),
        DType::Vlen(Box::new(DType::Float(4)))
    );

    // The plain variable beside them is unaffected.
    assert_eq!(file.variable("/plain").unwrap().vartype(), DType::Int(4));
}

#[test]
fn a_plain_variable_still_reads_its_values() {
    let file = fixture();
    let plain = file.variable("/plain").unwrap();

    assert_eq!(plain.shape, vec![3]);
    assert_eq!(plain.get_values::<i32, _>(..).unwrap(), vec![7, 8, 9]);
    assert_eq!(
        plain.attribute("units").unwrap().value.as_text(),
        Some("count")
    );
}

#[test]
fn a_compound_variable_carries_its_attributes() {
    let file = fixture();
    let readings = file.variable("/readings").unwrap();

    assert_eq!(readings.shape, vec![3]);
    assert_eq!(
        readings.attribute("long_name").unwrap().value.as_text(),
        Some("compound readings")
    );
}

#[test]
fn a_compound_variable_reads_its_raw_bytes() {
    let file = fixture();
    let readings = file.variable("/readings").unwrap();

    // The type is `{int station; float value; char label(8)}`, so each record
    // is 16 bytes: the int at 0, the float at 4, the label at 8.
    let raw = readings.get_raw_values(..).unwrap();
    assert_eq!(raw.len(), 3 * 16, "three records of sixteen bytes");

    let record = |i: usize| -> (i32, f32, String) {
        let r = &raw[i * 16..(i + 1) * 16];
        let label_end = r[8..].iter().position(|&b| b == 0).unwrap_or(8);
        (
            i32::from_le_bytes(r[0..4].try_into().unwrap()),
            f32::from_le_bytes(r[4..8].try_into().unwrap()),
            String::from_utf8_lossy(&r[8..8 + label_end]).into_owned(),
        )
    };

    assert_eq!(record(0), (11, 1.5, "alpha".to_string()));
    assert_eq!(record(1), (22, -2.25, "beta".to_string()));
    assert_eq!(record(2), (33, 0.0, "gamma".to_string()));
}
