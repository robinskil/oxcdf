//! Ragged arrays, which HDF5 calls a variable-length sequence.
//!
//! These moved down from the netCDF layer. That layer now mirrors the `netcdf`
//! crate, which has no ragged read, so a caller reaches one through HDF5.
//!
//! Each element stores a descriptor pointing into the global heap. A read is
//! only complete once the reader follows that pointer, so these tests check the
//! resolved values, never the descriptors.

use oxcdf_hdf5::dtype::{DType, Element};
use oxcdf_hdf5::index::Hdf5File;
use oxcdf_hdf5::message::{DatatypeClass, VlenKind};
use oxcdf_hdf5::read::{read_hyperslab, resolve_vlen_sequences, Hyperslab};

const VLEN_SEQ: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files/vlen_seq.nc");
const VLEN_STR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_files/vlen_strings.nc"
);

fn exists(path: &str) -> bool {
    std::path::Path::new(path).exists()
}

/// Read one ragged dataset and decode every sequence as `T`.
fn sequences<T: Element>(path: &str, dataset: &str, slab: Option<Hyperslab>) -> Vec<Vec<T>> {
    let file = Hdf5File::open(path).unwrap();
    let d = file.dataset(dataset).unwrap();
    d.prepare(file.ctx()).unwrap();

    let slab = slab.unwrap_or_else(|| Hyperslab::all(&d.shape));
    let raw = read_hyperslab(file.ctx(), d, &slab).unwrap();
    let buffers = resolve_vlen_sequences(file.ctx(), d, &raw).unwrap();

    let DatatypeClass::VariableLength { base, .. } = &d.datatype.class else {
        panic!("{dataset} is not a variable-length type");
    };
    let width = base.size as usize;

    buffers
        .iter()
        .map(|bytes| {
            bytes
                .chunks_exact(width)
                .map(|b| decode::<T>(b, base))
                .collect()
        })
        .collect()
}

/// Decode one element, converting the same way a `get_*` read would.
fn decode<T: Element>(bytes: &[u8], base: &oxcdf_hdf5::message::Datatype) -> T {
    let raw = oxcdf_hdf5::read::RawData {
        bytes: bytes.to_vec(),
        element_size: base.size as usize,
        shape: vec![1],
    };
    raw.get_of::<T>(base, "sequence element").unwrap()[0]
}

#[test]
fn reads_ragged_float_sequences() {
    if !exists(VLEN_SEQ) {
        return;
    }
    let file = Hdf5File::open(VLEN_SEQ).unwrap();
    let d = file.dataset("/rows").expect("rows");

    assert_eq!(
        DType::of(&d.datatype),
        DType::Vlen(Box::new(DType::Float(4)))
    );
    assert_eq!(d.shape, vec![4]);

    assert_eq!(
        sequences::<f64>(VLEN_SEQ, "/rows", None),
        vec![vec![1.5, 2.5], vec![], vec![3.5, 4.5, 5.5], vec![-0.25]],
        "an empty sequence is a real value, not a missing one"
    );
}

#[test]
fn reads_ragged_integer_sequences() {
    if !exists(VLEN_SEQ) {
        return;
    }
    let file = Hdf5File::open(VLEN_SEQ).unwrap();
    let d = file.dataset("/indices").unwrap();
    assert_eq!(DType::of(&d.datatype), DType::Vlen(Box::new(DType::Int(4))));

    assert_eq!(
        sequences::<i64>(VLEN_SEQ, "/indices", None),
        vec![vec![1], vec![2, 3], vec![], vec![4, 5, 6, 7]]
    );
}

#[test]
fn reads_a_slice_of_a_sequence_dataset() {
    if !exists(VLEN_SEQ) {
        return;
    }
    let slab = Hyperslab {
        start: vec![2],
        count: vec![2],
    };
    assert_eq!(
        sequences::<f64>(VLEN_SEQ, "/rows", Some(slab)),
        vec![vec![3.5, 4.5, 5.5], vec![-0.25]]
    );
}

/// Asking a sequence dataset for strings, or the reverse, must fail rather
/// than reinterpret the descriptors.
#[test]
fn sequence_and_string_resolvers_do_not_mix() {
    if !exists(VLEN_SEQ) || !exists(VLEN_STR) {
        return;
    }

    let seq = Hdf5File::open(VLEN_SEQ).unwrap();
    let rows = seq.dataset("/rows").unwrap();
    rows.prepare(seq.ctx()).unwrap();
    let raw = read_hyperslab(seq.ctx(), rows, &Hyperslab::all(&rows.shape)).unwrap();
    assert!(oxcdf_hdf5::read::resolve_vlen_strings(seq.ctx(), rows, &raw).is_err());

    let text = Hdf5File::open(VLEN_STR).unwrap();
    let names = text.dataset("/station_name").unwrap();
    names.prepare(text.ctx()).unwrap();
    let raw = read_hyperslab(text.ctx(), names, &Hyperslab::all(&names.shape)).unwrap();
    assert!(resolve_vlen_sequences(text.ctx(), names, &raw).is_err());

    // And the kinds really are different.
    let DatatypeClass::VariableLength { kind, .. } = &rows.datatype.class else {
        panic!()
    };
    assert_eq!(*kind, VlenKind::Sequence);
}
