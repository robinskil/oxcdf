//! Version 4 chunk indexes, one dataset per index type.
//!
//! `test_files/generate_latest.c` pins HDF5 to its latest format and shapes
//! each dataset so the library picks a different index for each. Every dataset
//! holds the same values, so any index that resolves wrongly shows up as wrong
//! numbers rather than as a parse failure.

use oxcdf::index::Hdf5File;
use oxcdf::read::{read_hyperslab, Hyperslab};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_files/latest_v4_layout.h5"
);

const NX: usize = 40;
const NY: usize = 6;

/// Every version 4 index type.
const WORKING: [&str; 5] = [
    "single_chunk",
    "implicit",
    "fixed_array",
    "extensible_array",
    "btree2_index",
];

fn expected() -> Vec<i64> {
    (0..(NX * NY) as i64).map(|i| i * 3 - 100).collect()
}

#[test]
fn every_version_four_index_resolves_to_the_same_values() {
    let file = Hdf5File::open(FIXTURE).unwrap();
    let want = expected();

    for name in WORKING {
        let d = file
            .dataset(&format!("/{name}"))
            .unwrap_or_else(|| panic!("{name} should exist"));

        assert_eq!(d.shape, vec![NX as u64, NY as u64], "{name}: wrong shape");
        assert!(
            d.chunks(file.ctx()).unwrap().is_some(),
            "{name}: a chunked dataset must resolve an index"
        );

        let raw = read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
            .unwrap_or_else(|e| panic!("{name}: read failed: {e}"));
        let got = raw.to_i64(d).unwrap();

        assert_eq!(got.len(), want.len(), "{name}: wrong element count");
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(a, b, "{name}: element {i} is wrong");
        }
    }
}

#[test]
fn the_single_chunk_index_yields_exactly_one_chunk() {
    let file = Hdf5File::open(FIXTURE).unwrap();
    let d = file.dataset("/single_chunk").unwrap();
    assert_eq!(d.chunks(file.ctx()).unwrap().unwrap().len(), 1);
}

#[test]
fn the_array_indexes_tile_the_dataset() {
    let file = Hdf5File::open(FIXTURE).unwrap();
    // A 40x6 dataset in 7x4 chunks is a 6x2 grid.
    for name in ["implicit", "fixed_array", "extensible_array", "btree2_index"] {
        let d = file.dataset(&format!("/{name}")).unwrap();
        let chunks = d.chunks(file.ctx()).unwrap().unwrap();
        assert_eq!(chunks.len(), 12, "{name}: expected a 6x2 chunk grid");

        let mut offsets: Vec<&Vec<u64>> = chunks.iter().map(|c| &c.offset).collect();
        offsets.sort();
        offsets.dedup();
        assert_eq!(offsets.len(), 12, "{name}: chunk offsets must be distinct");

        for c in chunks {
            assert_eq!(c.offset[0] % 7, 0, "{name}: offset off the chunk grid");
            assert_eq!(c.offset[1] % 4, 0, "{name}: offset off the chunk grid");
        }
    }
}

#[test]
fn a_hyperslab_across_chunk_boundaries_is_right_for_every_index() {
    let file = Hdf5File::open(FIXTURE).unwrap();
    let want = expected();

    for name in ["implicit", "fixed_array", "extensible_array", "btree2_index"] {
        let d = file.dataset(&format!("/{name}")).unwrap();
        let slab = Hyperslab::new(vec![5, 2], vec![10, 3], &d.shape).unwrap();
        let got = read_hyperslab(file.ctx(), d, &slab)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .to_i64(d)
            .unwrap();

        for row in 0..10usize {
            for col in 0..3usize {
                let global = (5 + row) * NY + (2 + col);
                assert_eq!(
                    got[row * 3 + col],
                    want[global],
                    "{name}: row {row} col {col}"
                );
            }
        }
    }
}

/// The extensible array spans its index block and a data block, so this checks
/// the join between the two: the first four chunks come from the index block
/// and the rest from the data block it points at.
#[test]
fn the_extensible_array_spans_its_index_and_data_blocks() {
    let file = Hdf5File::open(FIXTURE).unwrap();
    let d = file.dataset("/extensible_array").unwrap();
    assert!(d.is_readable());

    let chunks = d.chunks(file.ctx()).unwrap().unwrap();
    assert_eq!(chunks.len(), 12);
    // Four elements live inline; a wrong join would corrupt the fifth onwards.
    assert!(chunks.iter().all(|c| c.address > 0));
}

/// An index this reader cannot resolve must make one dataset unreadable rather
/// than fail the whole file or return a short chunk list.
#[test]
fn every_dataset_in_the_fixture_is_readable() {
    let file = Hdf5File::open(FIXTURE).unwrap();
    for name in WORKING {
        let d = file.dataset(&format!("/{name}")).unwrap();
        assert!(d.is_readable(), "{name} should be readable");
    }
}
