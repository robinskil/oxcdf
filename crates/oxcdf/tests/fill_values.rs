//! Storage that was never written must read as the variable's fill value.
//!
//! A read starts from the fill value and copies stored data over it. That copy
//! covers the whole selection for most reads, so the read path skips the fill
//! pass when it can prove it does. These tests hold the other half of that
//! bargain: where a chunk is absent, its elements still read as the fill value.
//!
//! `sparse_chunks.nc` comes from `test_files/generate_sparse_chunks.py`. Its
//! `part` variable has four chunks and only the first was written, `rows` has
//! the top half written, and `whole` has all of it.

use oxcdf::netcdf::NetcdfFile;

const PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_files/sparse_chunks.nc"
);

/// netCDF's default fill value for `int`, which is not zero.
const FILL_I32: i32 = -2147483647;
/// And for `short`.
const FILL_I16: i16 = -32767;

/// `part`: the top-left 4x4 holds 0..16 and every other chunk is absent.
fn part() -> Vec<i32> {
    let mut want = vec![FILL_I32; 64];
    for y in 0..4 {
        for x in 0..4 {
            want[y * 8 + x] = (y * 4 + x) as i32;
        }
    }
    want
}

/// `rows`: the top four rows hold 0..32 and the bottom four are absent.
fn rows() -> Vec<i16> {
    let mut want = vec![FILL_I16; 64];
    for y in 0..4 {
        for x in 0..8 {
            want[y * 8 + x] = (y * 8 + x) as i16;
        }
    }
    want
}

#[test]
fn absent_chunks_read_as_the_fill_value() {
    let file = NetcdfFile::open(PATH).unwrap();
    let got = file
        .variable("part")
        .unwrap()
        .get_values::<i32, _>(..)
        .unwrap();
    assert_eq!(got, part());
}

#[test]
fn a_selection_inside_an_absent_chunk_is_all_fill_value() {
    let file = NetcdfFile::open(PATH).unwrap();
    let got = file
        .variable("part")
        .unwrap()
        .get_values::<i32, _>([5..7, 5..7])
        .unwrap();
    assert_eq!(got, vec![FILL_I32; 4]);
}

#[test]
fn a_selection_straddling_the_written_edge_holds_both() {
    let file = NetcdfFile::open(PATH).unwrap();
    let got = file
        .variable("rows")
        .unwrap()
        .get_values::<i16, _>([2..6, 0..8])
        .unwrap();
    assert_eq!(got, rows()[2 * 8..6 * 8].to_vec());
}

#[test]
fn a_fully_written_variable_never_shows_its_fill_value() {
    let file = NetcdfFile::open(PATH).unwrap();
    let got = file
        .variable("whole")
        .unwrap()
        .get_values::<f32, _>(..)
        .unwrap();
    assert_eq!(got, (0..64).map(|i| i as f32).collect::<Vec<f32>>());
}

/// The asynchronous engine assembles its output separately, so it has to be
/// asked the same questions.
#[cfg(feature = "async")]
mod asynchronous {
    use super::{part, rows, FILL_I32, PATH};

    use std::sync::Arc;

    use oxcdf::source::FileSource;
    use oxcdf::{AsyncNetcdfFile, SyncAsAsync};

    fn open() -> (tokio::runtime::Runtime, AsyncNetcdfFile) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let source = Arc::new(SyncAsAsync(FileSource::open(PATH).unwrap()));
        let file = runtime.block_on(AsyncNetcdfFile::open(source)).unwrap();
        (runtime, file)
    }

    #[test]
    fn absent_chunks_read_as_the_fill_value() {
        let (runtime, file) = open();
        let got = runtime
            .block_on(file.variable("part").unwrap().get_values::<i32, _>(..))
            .unwrap();
        assert_eq!(got, part());
    }

    #[test]
    fn a_selection_inside_an_absent_chunk_is_all_fill_value() {
        let (runtime, file) = open();
        let got = runtime
            .block_on(
                file.variable("part")
                    .unwrap()
                    .get_values::<i32, _>([5..7, 5..7]),
            )
            .unwrap();
        assert_eq!(got, vec![FILL_I32; 4]);
    }

    #[test]
    fn a_selection_straddling_the_written_edge_holds_both() {
        let (runtime, file) = open();
        let got = runtime
            .block_on(
                file.variable("rows")
                    .unwrap()
                    .get_values::<i16, _>([2..6, 0..8]),
            )
            .unwrap();
        assert_eq!(got, rows()[2 * 8..6 * 8].to_vec());
    }
}
