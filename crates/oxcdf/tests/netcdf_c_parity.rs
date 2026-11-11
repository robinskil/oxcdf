//! Cases where this reader gave a different result from netcdf-c.
//!
//! The fixtures come from `test_files/generate_netcdf_c_parity.py`. The
//! expected values come from netCDF4-python, which reads through netcdf-c.

use oxcdf::netcdf::NetcdfFile;

fn open(name: &str) -> NetcdfFile {
    let path = format!("{}/../../test_files/{name}", env!("CARGO_MANIFEST_DIR"));
    NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn names(file: &NetcdfFile) -> Vec<&str> {
    file.root()
        .variables
        .iter()
        .map(|v| v.name.as_str())
        .collect()
}

#[test]
fn a_cdf5_file_with_records_opens() {
    let file = open("cdf5_records.nc");
    assert_eq!(file.dimension("time").unwrap().len, 4);
    assert_eq!(
        file.variable("rec")
            .unwrap()
            .get_values::<i64, _>(..)
            .unwrap(),
        vec![-(1 << 62), 0, 1, 1 << 62]
    );
    assert_eq!(
        file.variable("fixed")
            .unwrap()
            .get_values::<u64, _>(..)
            .unwrap(),
        vec![0, 1 << 63, u64::MAX - 1]
    );
}

#[test]
fn a_variable_with_the_name_of_a_dimension_keeps_its_netcdf_name() {
    let file = open("non_coord_names.nc");
    assert_eq!(names(&file), vec!["b", "a"]);

    let b = file.variable("b").unwrap();
    assert_eq!(b.info().path, "/b");
    assert_eq!(b.info().dimensions, vec!["a"]);
    assert_eq!(b.get_values::<i32, _>(..).unwrap(), vec![1, 2, 3]);

    let a = file.variable("/a").unwrap();
    assert_eq!(a.info().dimensions, vec!["b", "a"]);
    assert_eq!(
        a.get_values::<f32, _>(..).unwrap(),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    );

    let inner = file.variable("/grp/b").unwrap();
    assert_eq!(inner.info().path, "/grp/b");
    assert_eq!(inner.get_values::<i16, _>(..).unwrap(), vec![7, 8, 9]);
}

#[test]
fn a_renamed_variable_in_a_large_group_is_found() {
    let file = open("renamed_variable.nc");
    let mut want: Vec<String> = (0..12).map(|i| format!("v{i:02}")).collect();
    want[1] = "renamed".to_string();
    // netcdf-c reports the variables in creation order.
    assert_eq!(names(&file), want);
    assert_eq!(
        file.variable("renamed")
            .unwrap()
            .get_values::<i32, _>(..)
            .unwrap(),
        vec![1, -1]
    );
}

#[test]
fn an_attribute_above_64_kib_is_read() {
    let file = open("huge_attribute.nc");
    let big = file.attribute("big").unwrap().value.as_text().unwrap();
    assert_eq!(big.len(), 66000);
    assert!(big.bytes().all(|b| b == b'x'));

    let v = file.variable("v").unwrap();
    assert!(v.info().attributes_complete);
    let big = v.attribute("big").unwrap().value.as_text().unwrap();
    assert_eq!(big.len(), 66000);
    assert!(big.bytes().all(|b| b == b'y'));
    assert_eq!(v.attribute("small").unwrap().value.as_text(), Some("s"));
}

#[test]
fn a_soft_link_to_a_dataset_is_a_variable() {
    for name in ["soft_link.h5", "soft_link_latest.h5"] {
        let file = open(name);
        assert_eq!(names(&file), vec!["a", "chain", "soft"], "{name}");
        for path in ["/soft", "/chain", "/grp/rel", "/grp/b"] {
            assert_eq!(
                file.variable(path)
                    .unwrap_or_else(|| panic!("{name}: {path} is missing"))
                    .get_values::<f64, _>(..)
                    .unwrap(),
                vec![0.0, 1.0],
                "{name}: {path}"
            );
        }
        assert_eq!(
            file.variable("soft").unwrap().info().dimensions,
            vec!["phony_dim_2"],
            "{name}"
        );
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn the_async_engine_reads_the_same_files() {
    let open = |name: &str| {
        let path = format!("{}/../../test_files/{name}", env!("CARGO_MANIFEST_DIR"));
        oxcdf::open_async(std::sync::Arc::new(oxcdf::SyncAsAsync(
            oxcdf::FileSource::open(&path).unwrap(),
        )))
    };

    let file = open("renamed_variable.nc").await.unwrap();
    let renamed = file.variable("renamed").unwrap();
    assert_eq!(renamed.get_values::<i32, _>(..).await.unwrap(), vec![1, -1]);

    let file = open("non_coord_names.nc").await.unwrap();
    let b = file.variable("b").unwrap();
    assert_eq!(b.get_values::<i32, _>(..).await.unwrap(), vec![1, 2, 3]);

    let file = open("huge_attribute.nc").await.unwrap();
    assert_eq!(
        file.attribute("big")
            .unwrap()
            .value
            .as_text()
            .unwrap()
            .len(),
        66000
    );

    let file = open("soft_link_latest.h5").await.unwrap();
    let chain = file.variable("chain").unwrap();
    assert_eq!(
        chain.get_values::<f64, _>(..).await.unwrap(),
        vec![0.0, 1.0]
    );

    let file = open("cdf5_records.nc").await.unwrap();
    assert_eq!(file.variable("rec").unwrap().len(), 4);
}
