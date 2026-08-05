//! Strings, in every representation netCDF uses.
//!
//! netCDF stores strings two ways, and this reader keeps them distinct.
//!
//! * A `string` variable holds one variable-length string in each element. The
//!   value lives in the global heap. `get_strings` returns one string for each
//!   element.
//! * A `char` variable holds one byte in each element. Its last dimension is
//!   the string length. `get_strings` returns one string for each character,
//!   because that is one element. The caller joins the last axis.
//!
//! An HDF5 fixed-length string wider than one byte holds one string in each
//! element. netcdf-c does not write those, but other writers do.

use oxcdf::DType;

fn corpus(name: &str) -> Option<String> {
    let path = format!("{}/test_files/{name}", env!("CARGO_MANIFEST_DIR"));
    std::path::Path::new(&path).exists().then_some(path)
}

// ─── variable-length strings ───────────────────────────────────────────────

#[test]
fn a_string_variable_returns_one_string_for_each_element() {
    let Some(path) = corpus("vlen_strings.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("station_name").unwrap();

    assert_eq!(v.dtype(), DType::String);
    assert_eq!(v.shape, vec![4]);

    let names = v.get_strings(..).unwrap();
    assert_eq!(names.len(), 4, "one string for each element, not one per byte");
    assert_eq!(names[0], "Ålesund", "multi-byte UTF-8 survives");
    assert_eq!(names[2], "", "an empty string is a value, not a gap");
    assert_eq!(
        names[3], "a much longer station name than the others",
        "a long value is not truncated"
    );
}

#[test]
// A rank-1 variable takes a one-element selection. That is the point here.
#[allow(clippy::single_range_in_vec_init)]
fn a_string_selection_reads_only_what_it_names() {
    let Some(path) = corpus("vlen_strings.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("station_name").unwrap();
    let all = v.get_strings(..).unwrap();

    assert_eq!(v.get_strings([1..3]).unwrap(), &all[1..3]);
    assert_eq!(v.get_string([0]).unwrap(), all[0]);
    assert_eq!(v.get_string([3]).unwrap(), all[3]);

    // More than one element is a bad request for `get_string`.
    assert!(v.get_string([0..2]).is_err());
}

#[test]
fn a_string_variable_read_as_a_number_reports_the_stored_type() {
    let Some(path) = corpus("vlen_strings.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("station_name").unwrap();

    let err = v.get_values::<f64, _>(..).unwrap_err();
    assert!(
        err.to_string().contains("stores string"),
        "the message should name the stored type, got: {err}"
    );
}

#[cfg(feature = "ndarray")]
#[test]
fn string_values_keep_their_shape_as_an_array() {
    let Some(path) = corpus("vlen_strings.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();
    let a = file
        .variable("station_name")
        .unwrap()
        .read()
        .unwrap()
        .to_array_strings()
        .unwrap();
    assert_eq!(a.shape(), &[4]);
    assert_eq!(a[[0]], "Ålesund");
}

// ─── char variables ────────────────────────────────────────────────────────

#[test]
fn a_char_variable_reports_one_element_for_each_character() {
    let Some(path) = corpus("wod_ctd_1964.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("country").unwrap();

    // The last dimension is the string length. The reader reports the elements
    // as stored; joining that axis is the caller's job.
    assert_eq!(v.dtype(), DType::Char, "netCDF calls this `char`, not a string");
    assert_eq!(v.shape, vec![47, 40]);
    assert_eq!(v.dimensions, vec!["casts", "strnlensmall"]);

    let chars = v.get_strings(..).unwrap();
    assert_eq!(chars.len(), 47 * 40);

    // Joining the last axis gives the values `ncdump` prints.
    let width = *v.shape.last().unwrap() as usize;
    let joined: Vec<String> = chars
        .chunks(width)
        .map(|row| row.concat().trim_end_matches('\0').to_string())
        .collect();
    assert_eq!(joined.len(), 47);
    assert_eq!(joined[0], "GREAT BRITAIN");
}

#[test]
fn char_bytes_are_available_whole() {
    let Some(path) = corpus("wod_ctd_1964.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("country").unwrap();

    // The raw bytes are the simplest route for a caller that wants to build its
    // own string array.
    let values = v.read().unwrap();
    let bytes = values.as_bytes();
    assert_eq!(bytes.len(), 47 * 40);
    // A `char` variable pads with NUL, not with spaces.
    let first = String::from_utf8_lossy(&bytes[..40])
        .trim_end_matches('\0')
        .to_string();
    assert_eq!(first, "GREAT BRITAIN");
}

#[test]
fn a_rank_one_char_variable_is_one_string_across_its_axis() {
    let Some(path) = corpus("test_file.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("DATA_TYPE").unwrap();

    assert_eq!(v.shape, vec![16], "STRING16 is the string length");
    let chars = v.get_strings(..).unwrap();
    assert_eq!(chars.len(), 16);
    assert_eq!(chars.concat().trim_end(), "MO");
}

// ─── both engines agree ────────────────────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test]
async fn the_async_engine_reads_the_same_strings() {
    for name in ["vlen_strings.nc", "wod_ctd_1964.nc", "test_file.nc"] {
        let Some(path) = corpus(name) else { continue };
        let sync = oxcdf::open(&path).unwrap();
        let file = oxcdf::open_async(std::sync::Arc::new(oxcdf::SyncAsAsync(
            oxcdf::FileSource::open(&path).unwrap(),
        )))
        .await
        .unwrap();

        for want in sync.variables() {
            if !want.dtype().is_text() {
                continue;
            }
            let got = file.variable(&want.path).unwrap();
            assert_eq!(
                got.get_strings(..).await.unwrap(),
                want.get_strings(..).unwrap(),
                "{} in {name}",
                want.path
            );
        }
    }
}

/// A `string` variable in the global heap must survive an asynchronous read.
///
/// The heap is a second dependent read, so the async engine resolves it through
/// the replay driver rather than the data path.
#[cfg(feature = "async")]
#[tokio::test]
async fn the_async_engine_follows_the_global_heap() {
    let Some(path) = corpus("vlen_strings.nc") else {
        return;
    };
    let file = oxcdf::open_async(std::sync::Arc::new(oxcdf::SyncAsAsync(
        oxcdf::FileSource::open(&path).unwrap(),
    )))
    .await
    .unwrap();

    let v = file.variable("station_name").unwrap();
    let names = v.get_strings(..).await.unwrap();
    assert_eq!(names.len(), 4);
    assert_eq!(names[0], "Ålesund");
    assert_eq!(names[2], "");
    assert_eq!(
        v.get_string([3]).await.unwrap(),
        "a much longer station name than the others"
    );
}

// ─── parity with the netcdf crate ──────────────────────────────────────────

#[cfg(feature = "diff-tests")]
#[test]
fn netcdf_string_parity() {
    let Some(path) = corpus("vlen_strings.nc") else {
        return;
    };
    let ours = oxcdf::open(&path).unwrap();
    let theirs = netcdf::open(&path).unwrap();

    for v in ours.variables() {
        if v.dtype() != DType::String {
            continue;
        }
        let Some(other) = theirs.variable(&v.name) else {
            continue;
        };
        assert_eq!(
            v.get_strings(..).unwrap(),
            other.get_strings(netcdf::Extents::All).unwrap(),
            "strings of {}",
            v.path
        );
    }
}
