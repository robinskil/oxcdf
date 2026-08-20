//! Nested groups, and the dimensions netCDF invents for a plain HDF5 file.
//!
//! An instrument writes HDF5 directly: groups nested several levels, and no
//! dimension scale anywhere. netCDF still needs a dimension for every axis, so
//! it invents one and calls it `phony_dim_N`.
//!
//! `netcdf_layer.rs` checks the whole fixture against `ncdump`. These tests
//! pin each rule on its own, so a failure names the rule that broke, and so the
//! rules stay covered where `ncdump` is not installed.
//!
//! `test_files/generate_groups.c` writes the fixture and lists the rules.

use oxcdf::netcdf::{NcGroup, NetcdfFile};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_files/nested_groups.h5"
);

fn open() -> NetcdfFile {
    NetcdfFile::open(FIXTURE).expect("the fixture opens")
}

/// A group's dimensions as `(name, length, unlimited)`, in the reported order.
fn dims(group: &NcGroup) -> Vec<(&str, u64, bool)> {
    group
        .dimensions
        .iter()
        .map(|d| (d.name.as_str(), d.len, d.is_unlimited))
        .collect()
}

#[test]
fn every_group_is_reachable_by_path() {
    let file = open();

    let mut paths: Vec<&str> = Vec::new();
    fn walk<'a>(g: &'a NcGroup, out: &mut Vec<&'a str>) {
        out.push(&g.path);
        for c in &g.groups {
            walk(c, out);
        }
    }
    walk(file.root(), &mut paths);
    paths.sort_unstable();

    assert_eq!(
        paths,
        vec!["/", "/edges", "/outer", "/outer/inner", "/scaled"]
    );

    // The same groups answer to an absolute path.
    assert_eq!(file.group("/outer/inner").unwrap().name, "inner");
    assert_eq!(file.group("/").unwrap().path, "/");
    assert!(file.group("/outer/missing").is_none());
}

/// A group's children are numbered before its own variables, and one counter
/// runs over the whole file.
#[test]
fn phony_numbers_run_over_the_whole_file_children_first() {
    let file = open();

    // `/scaled/row` is a named dimension, so it takes id 0 and the invented
    // ones start at 1.
    assert_eq!(
        dims(file.group("/scaled").unwrap()),
        vec![("row", 4, false)]
    );

    // `/outer/inner` is numbered before `/outer`, and the root last of all.
    assert_eq!(
        dims(file.group("/outer/inner").unwrap()),
        vec![("phony_dim_5", 6, false)]
    );
    assert_eq!(
        dims(file.group("/outer").unwrap()),
        vec![
            ("phony_dim_6", 4, false),
            ("phony_dim_7", 6, false),
            ("phony_dim_8", 4, false),
        ]
    );
    assert_eq!(dims(file.root()), vec![("phony_dim_9", 6, false)]);

    // Every invented dimension says so, and carries its number as its id.
    for d in &file.group("/outer").unwrap().dimensions {
        assert!(d.is_phony, "{} should be marked invented", d.name);
    }
    assert_eq!(file.dimensions()[0].id, Some(9));
    assert!(!file.group("/scaled").unwrap().dimensions[0].is_phony);
}

/// One dimension serves every axis of that length in its own group, but never
/// two axes of one variable, and never an axis in another group.
#[test]
fn one_dimension_serves_every_axis_of_its_length_in_its_group() {
    let file = open();

    // `flat` is 4 long and `square` is 4 by 4. The two axes of `square` cannot
    // share a dimension, so the second needs one of its own.
    let flat = file.variable("/outer/flat").unwrap();
    assert_eq!(flat.dimensions, vec!["phony_dim_6"]);
    let square = file.variable("/outer/square").unwrap();
    assert_eq!(square.dimensions, vec!["phony_dim_6", "phony_dim_8"]);

    // `pair` is 6 by 4. Its second axis reuses the 4 that `flat` established.
    let pair = file.variable("/outer/pair").unwrap();
    assert_eq!(pair.dimensions, vec!["phony_dim_7", "phony_dim_6"]);

    // A child group does not reach its parent's dimensions, so this 6 is not
    // the 6 in `/outer`.
    let six = file.variable("/outer/inner/six").unwrap();
    assert_eq!(six.dimensions, vec!["phony_dim_5"]);

    // Nor does the root reach into a child.
    let top = file.variable("/top").unwrap();
    assert_eq!(top.dimensions, vec!["phony_dim_9"]);
}

/// netCDF has no fixed dimension of length zero, so an empty axis becomes an
/// unlimited dimension, and never serves a second empty axis.
#[test]
fn an_empty_axis_gets_a_dimension_of_its_own() {
    let file = open();
    let edges = file.group("/edges").unwrap();

    assert_eq!(
        dims(edges),
        vec![
            ("phony_dim_1", 0, true),
            ("phony_dim_2", 0, true),
            ("phony_dim_3", 3, false),
            ("phony_dim_4", 3, true),
        ]
    );

    assert_eq!(
        file.variable("/edges/empty_a").unwrap().dimensions,
        vec!["phony_dim_1"]
    );
    assert_eq!(
        file.variable("/edges/empty_b").unwrap().dimensions,
        vec!["phony_dim_2"]
    );
}

/// A growable axis matches only another growable one, whatever the lengths say.
#[test]
fn a_growable_axis_does_not_share_with_a_fixed_one() {
    let file = open();

    // Both are 3 long. Only one can grow, so they take separate dimensions.
    let growable = file.variable("/edges/growable").unwrap();
    let fixed = file.variable("/edges/fixed_three").unwrap();
    assert_eq!(growable.dimensions, vec!["phony_dim_4"]);
    assert_eq!(fixed.dimensions, vec!["phony_dim_3"]);

    let edges = file.group("/edges").unwrap();
    assert!(edges.dimension("phony_dim_4").unwrap().is_unlimited);
    assert!(!edges.dimension("phony_dim_3").unwrap().is_unlimited);
}

/// An axis with no scale of its own still lands on a named dimension of the
/// right length, rather than getting an invented one beside it.
#[test]
fn an_unattached_axis_lands_on_a_named_dimension() {
    let file = open();

    let uses = file.variable("/scaled/uses_scale").unwrap();
    assert_eq!(uses.dimensions, vec!["row"]);

    // The scale is a coordinate variable, so it is a dimension and a variable.
    let row = file.variable("/scaled/row").unwrap();
    assert_eq!(row.dimensions, vec!["row"]);
    let scaled = file.group("/scaled").unwrap();
    assert_eq!(scaled.dimensions.len(), 1, "no dimension was invented here");
    assert!(scaled.dimension("row").unwrap().has_coordinate_variable);
}

/// Every variable in the tree is listed and readable, at any depth.
#[test]
fn values_read_from_any_depth() {
    let file = open();

    let mut paths: Vec<String> = file.variables().iter().map(|v| v.path.clone()).collect();
    paths.sort();
    assert_eq!(
        paths,
        vec![
            "/edges/empty_a",
            "/edges/empty_b",
            "/edges/fixed_three",
            "/edges/growable",
            "/outer/flat",
            "/outer/inner/six",
            "/outer/pair",
            "/outer/square",
            "/scaled/row",
            "/scaled/uses_scale",
            "/top",
        ]
    );

    // The generator fills every dataset with its own element index.
    let six = file.variable("/outer/inner/six").unwrap();
    assert_eq!(
        six.get_values::<i32, _>(..).unwrap(),
        vec![0, 1, 2, 3, 4, 5]
    );

    let pair = file.variable("/outer/pair").unwrap();
    assert_eq!(pair.shape, vec![6, 4]);
    assert_eq!(
        pair.get_values::<i32, _>([1..3, 0..2]).unwrap(),
        vec![4, 5, 8, 9]
    );

    // An empty variable reads as nothing rather than failing.
    let empty = file.variable("/edges/empty_a").unwrap();
    assert!(empty.get_values::<i32, _>(..).unwrap().is_empty());
}

/// The asynchronous engine builds the same view. It shares this layer, so this
/// guards the sharing rather than the rules.
#[cfg(feature = "async")]
#[test]
fn the_asynchronous_engine_reports_the_same_groups() {
    use oxcdf::AsyncNetcdfFile;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        let source = std::sync::Arc::new(oxcdf::SyncAsAsync(
            oxcdf::FileSource::open(FIXTURE).unwrap(),
        ));
        let file = AsyncNetcdfFile::open(source).await.unwrap();

        assert_eq!(
            dims(file.group("/outer/inner").unwrap()),
            vec![("phony_dim_5", 6, false)]
        );
        let six = file.variable("/outer/inner/six").unwrap();
        assert_eq!(six.info().dimensions, vec!["phony_dim_5"]);
        assert_eq!(
            six.get_values::<i32, _>(..).await.unwrap(),
            vec![0, 1, 2, 3, 4, 5]
        );
    });
}
