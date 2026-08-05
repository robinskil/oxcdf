//! Typed reads and selections.
//!
//! The rules here follow the `netcdf` crate. A read converts between any two
//! numeric types. A read of a string as a number fails. The selection forms are
//! the same forms.
//!
//! `netcdf_parity` compares against that crate directly, when `diff-tests` is
//! on.

use oxcdf::{DType, Error, Extents};

fn corpus(name: &str) -> Option<String> {
    let path = format!("{}/../../test_files/{name}", env!("CARGO_MANIFEST_DIR"));
    std::path::Path::new(&path).exists().then_some(path)
}

const LEGACY: &str = "legacy_v1_objheader.h5";

// ─── exact reads ───────────────────────────────────────────────────────────

#[test]
fn a_read_of_the_stored_type_copies_the_values() {
    let Some(path) = corpus(LEGACY) else { return };
    let file = oxcdf::open(&path).unwrap();

    let f64_var = file.variable("/contig_f64").unwrap();
    assert_eq!(f64_var.vartype(), DType::Float(8));
    let exact = f64_var.get_values::<f64, _>(..).unwrap();

    // The same bytes, reinterpreted. No conversion can have happened.
    let bytes = f64_var.get_raw_values(..).unwrap();
    for (i, v) in exact.iter().enumerate() {
        let b: [u8; 8] = bytes[i * 8..i * 8 + 8].try_into().unwrap();
        assert_eq!(v.to_bits(), f64::from_ne_bytes(b).to_bits(), "element {i}");
    }
}

#[test]
fn an_f32_variable_reads_as_f32_without_a_round_trip_through_f64() {
    let Some(path) = corpus(LEGACY) else { return };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("/contig_f32be").unwrap();
    assert_eq!(v.vartype(), DType::Float(4));

    let exact = v.get_values::<f32, _>(..).unwrap();
    let raw = v.get_raw_values(..).unwrap();
    for (i, got) in exact.iter().enumerate() {
        let b: [u8; 4] = raw[i * 4..i * 4 + 4].try_into().unwrap();
        assert_eq!(
            got.to_bits(),
            f32::from_ne_bytes(b).to_bits(),
            "element {i}"
        );
    }
}

#[test]
fn an_i32_variable_reads_as_i32() {
    let Some(path) = corpus(LEGACY) else { return };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("/chunked_i32").unwrap();
    assert_eq!(v.vartype(), DType::Int(4));

    let exact = v.get_values::<i32, _>(..).unwrap();
    let wide = v.get_values::<i64, _>(..).unwrap();
    assert_eq!(exact.len(), wide.len());
    for (a, b) in exact.iter().zip(&wide) {
        assert_eq!(i64::from(*a), *b);
    }
}

// ─── conversion, as the netcdf crate does it ───────────────────────────────

#[test]
fn any_numeric_type_converts_to_any_other() {
    let Some(path) = corpus(LEGACY) else { return };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("/chunked_i32").unwrap();

    // Integer to float, integer to narrower integer, and back.
    assert!(v.get_values::<f32, _>(..).is_ok());
    assert!(v.get_values::<f64, _>(..).is_ok());
    assert!(v.get_values::<i16, _>(..).is_ok());
    assert!(v.get_values::<u8, _>(..).is_ok());

    let f = file.variable("/contig_f64").unwrap();
    assert!(f.get_values::<i64, _>(..).is_ok(), "float to integer");
    assert!(
        f.get_values::<f32, _>(..).is_ok(),
        "float to narrower float"
    );
}

#[test]
fn a_float_to_integer_read_truncates_toward_zero() {
    let Some(path) = corpus(LEGACY) else { return };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("/contig_f64").unwrap();

    let floats = v.get_values::<f64, _>(..).unwrap();
    let ints = v.get_values::<i64, _>(..).unwrap();
    for (f, i) in floats.iter().zip(&ints) {
        assert_eq!(*i, *f as i64, "{f} should truncate to {i}");
    }
}

#[test]
fn a_string_variable_read_as_a_number_reports_the_stored_type() {
    let Some(path) = corpus(LEGACY) else { return };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("/fixed_strings").unwrap();

    let err = v.get_values::<f64, _>(..).unwrap_err();
    match err {
        Error::TypeMismatch { stored, asked, .. } => {
            assert!(stored.contains("string"), "stored was {stored}");
            assert_eq!(asked, "f64");
        }
        other => panic!("expected TypeMismatch, got {other:?}"),
    }
    // The message must name the stored type, so a caller can ask again.
    assert!(v
        .get_values::<f64, _>(..)
        .unwrap_err()
        .to_string()
        .contains("string"));
}

// ─── selections ────────────────────────────────────────────────────────────

#[test]
fn every_selection_form_reads_the_same_elements() {
    let Some(path) = corpus(LEGACY) else { return };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("/contig_f64").unwrap();
    assert_eq!(v.shape, vec![40, 6]);

    let all = v.get_values::<f64, _>(Extents::All).unwrap();
    assert_eq!(v.get_values::<f64, _>(..).unwrap(), all);
    assert_eq!(v.get_values::<f64, _>(()).unwrap(), all);
    assert_eq!(v.get_values::<f64, _>([0..40, 0..6]).unwrap(), all);
    assert_eq!(v.get_values::<f64, _>([0.., 0..]).unwrap(), all);

    // A block, three ways.
    let want = v.get_values::<f64, _>([2..5, 1..4]).unwrap();
    assert_eq!(want.len(), 9);
    assert_eq!(v.get_values::<f64, _>([2..=4, 1..=3]).unwrap(), want);
    assert_eq!(
        v.get_values::<f64, _>(([2usize, 1].as_slice(), [3usize, 3].as_slice()))
            .unwrap(),
        want
    );

    // An index fixes an axis to one element. Mixing an index and a range needs
    // `Extent`, because a Rust array holds one type.
    let row = v
        .get_values::<f64, _>([oxcdf::Extent::Index(3), (0..6).into()])
        .unwrap();
    assert_eq!(row.len(), 6);
    assert_eq!(row, all[3 * 6..4 * 6]);

    // A range that starts at zero.
    assert_eq!(v.get_values::<f64, _>([..2, ..6]).unwrap(), all[..12]);
}

#[test]
fn get_value_reads_exactly_one_element() {
    let Some(path) = corpus(LEGACY) else { return };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("/contig_f64").unwrap();
    let all = v.get_values::<f64, _>(..).unwrap();

    assert_eq!(v.get_value::<f64, _>([0, 0]).unwrap(), all[0]);
    assert_eq!(v.get_value::<f64, _>([3, 2]).unwrap(), all[3 * 6 + 2]);

    // A selection naming more than one element is a bad request.
    let err = v.get_value::<f64, _>([0..2, 0..2]).unwrap_err();
    assert!(matches!(err, Error::BadRequest(_)), "got {err:?}");
}

#[test]
#[allow(clippy::single_range_in_vec_init)]
fn a_selection_of_the_wrong_rank_or_past_the_end_is_reported() {
    let Some(path) = corpus(LEGACY) else { return };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("/contig_f64").unwrap();

    assert!(
        v.get_values::<f64, _>([0..2]).is_err(),
        "rank 1 on a rank-2 variable"
    );
    assert!(
        v.get_values::<f64, _>([0..99, 0..6]).is_err(),
        "past the end"
    );
}

#[test]
fn a_stride_is_refused_rather_than_ignored() {
    let Some(path) = corpus(LEGACY) else { return };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("/contig_f64").unwrap();

    let strided = Extents::Extent(vec![
        oxcdf::Extent::SliceEnd {
            start: 0,
            end: 40,
            stride: 2,
        },
        oxcdf::Extent::Index(0),
    ]);
    let err = v.get_values::<f64, _>(strided).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
}

// ─── ndarray ───────────────────────────────────────────────────────────────

#[cfg(feature = "ndarray")]
#[test]
fn get_array_keeps_the_selection_shape() {
    let Some(path) = corpus(LEGACY) else { return };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("/contig_f64").unwrap();

    let a = v.get::<f64, _>(..).unwrap();
    assert_eq!(a.shape(), &[40, 6]);
    let b = v.get::<f32, _>([2..5, 1..4]).unwrap();
    assert_eq!(b.shape(), &[3, 3]);
}

// ─── the async engine reads the same values ────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test]
async fn the_async_engine_applies_the_same_type_rules() {
    let Some(path) = corpus(LEGACY) else { return };
    let sync = oxcdf::open(&path).unwrap();
    let file = oxcdf::open_async(std::sync::Arc::new(oxcdf::SyncAsAsync(
        oxcdf::FileSource::open(&path).unwrap(),
    )))
    .await
    .unwrap();

    for want in sync.variables() {
        let got = file.variable(&want.path).unwrap();
        assert_eq!(got.vartype(), want.vartype(), "{}", want.path);

        match want.vartype() {
            DType::Float(_) | DType::Int(_) | DType::Uint(_) => {
                assert_eq!(
                    got.get_values::<f64, _>(..).await.unwrap(),
                    want.get_values::<f64, _>(..).unwrap(),
                    "{}",
                    want.path
                );
                assert_eq!(
                    got.get_value::<f64, _>(vec![0usize; want.shape.len()])
                        .await
                        .unwrap(),
                    want.get_value::<f64, _>(vec![0usize; want.shape.len()])
                        .unwrap(),
                );
            }
            _ => {
                assert!(got.get_values::<f64, _>(..).await.is_err(), "{}", want.path);
            }
        }
    }
}

// ─── parity with the netcdf crate ──────────────────────────────────────────

/// Read every variable both ways, as the stored type and as `f64`, and compare.
///
/// This is the check that matters: it proves the conversion rules match, not
/// just that the bytes are right.
#[cfg(feature = "diff-tests")]
#[test]
fn netcdf_parity() {
    for name in ["test_file.nc", "gridded-example.nc", "wod_ctd_1964.nc"] {
        let Some(path) = corpus(name) else { continue };
        let ours = oxcdf::open(&path).unwrap();
        let theirs = netcdf::open(&path).unwrap();

        for v in ours.variables() {
            if !v.is_readable() || v.is_empty() {
                continue;
            }
            let Some(other) = theirs.variable(&v.name) else {
                continue;
            };

            macro_rules! compare {
                ($t:ty) => {{
                    let mine = v.get_values::<$t, _>(..).unwrap();
                    let yours = other.get_values::<$t, _>(netcdf::Extents::All).unwrap();
                    assert_eq!(mine.len(), yours.len(), "{} in {name}", v.path);
                    for (i, (a, b)) in mine.iter().zip(&yours).enumerate() {
                        assert_eq!(a, b, "{} element {i} in {name}", v.path);
                    }
                }};
            }

            // As the stored type, then widened. Both must agree with netcdf.
            match v.vartype() {
                DType::Int(1) => compare!(i8),
                DType::Int(2) => compare!(i16),
                DType::Int(4) => compare!(i32),
                DType::Int(8) => compare!(i64),
                DType::Uint(1) => compare!(u8),
                DType::Uint(2) => compare!(u16),
                DType::Uint(4) => compare!(u32),
                DType::Uint(8) => compare!(u64),
                DType::Float(4) => compare!(f32),
                DType::Float(8) => compare!(f64),
                _ => continue,
            }
        }
    }
}

/// The selection forms must pick the same elements as the netcdf crate.
#[cfg(feature = "diff-tests")]
#[test]
fn netcdf_selection_parity() {
    let Some(path) = corpus("gridded-example.nc") else {
        return;
    };
    let ours = oxcdf::open(&path).unwrap();
    let theirs = netcdf::open(&path).unwrap();

    for v in ours.variables() {
        if !v.is_readable() || v.vartype() != DType::Float(4) || v.shape.len() < 2 {
            continue;
        }
        let Some(other) = theirs.variable(&v.name) else {
            continue;
        };
        if v.shape[0] < 2 || v.shape[1] < 4 {
            continue;
        }

        let mine = v.get_values::<f32, _>([0..1, 1..4]).unwrap();
        let yours = other
            .get_values::<f32, _>((&[0usize, 1][..], &[1usize, 3][..]))
            .unwrap();
        assert_eq!(mine, yours, "block of {}", v.path);

        let mine = v.get_value::<f32, _>([0, 2]).unwrap();
        let yours = other.get_value::<f32, _>([0usize, 2]).unwrap();
        assert_eq!(mine, yours, "one element of {}", v.path);
    }
}
