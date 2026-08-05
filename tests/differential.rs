//! Differential tests against netcdf-c.
//!
//! Every other test in this crate checks the reader against a specification, a
//! fixture whose contents are known, or `ncdump`'s *metadata*. This suite is the
//! one that checks **values**, element by element, against the reference
//! implementation reading the same bytes.
//!
//! That distinction matters. A structural bug shows up as a parse failure or a
//! wrong shape, and the other suites catch it. A decoding bug — a byte order
//! flipped, a chunk assembled at the wrong offset, a filter undone in the wrong
//! order — produces plausible numbers that no structural check would question.
//! Only comparing against a second implementation finds those.
//!
//! Floats are compared by their bit patterns, not with a tolerance. The reader
//! does not compute anything, it relocates bytes, so any difference at all is a
//! bug. Comparing bits also catches `NaN` payloads and negative zero, which
//! `==` would quietly accept.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p oxcdf --features diff-tests --test differential
//! ```
#![cfg(feature = "diff-tests")]

use oxcdf::netcdf::{NetcdfFile, Variable};
use netcdf::types::{FloatType, IntType, NcVariableType};

fn corpus() -> Vec<(&'static str, String)> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/test_files");
    [
        ("test_file.nc", "test_file.nc"),
        (
            "gridded-example.nc",
            "gridded-example.nc",
        ),
        (
            "wod_ctd_1964.nc",
            "wod_ctd_1964.nc",
        ),
    ]
    .iter()
    .map(|(name, p)| (*name, format!("{root}/{p}")))
    .filter(|(_, p)| std::path::Path::new(p).exists())
    .collect()
}

/// What a comparison did, so the suite can report coverage honestly instead of
/// passing silently because everything was skipped.
#[derive(Default, Debug)]
struct Tally {
    integer: usize,
    float: usize,
    skipped_char: usize,
    skipped_other: usize,
}

impl Tally {
    fn compared(&self) -> usize {
        self.integer + self.float
    }
}

/// Compare one integer variable, whole.
fn compare_integers(
    label: &str,
    reference: &netcdf::Variable<'_>,
    mine: &Variable<'_>,
    int_type: IntType,
) {
    // Read through netcdf-c at its stored width, then widen, so no conversion
    // happens inside the C library that could mask a difference.
    let expected: Vec<i64> = match int_type {
        IntType::I8 => read_ref::<i8>(reference, label)
            .into_iter()
            .map(i64::from)
            .collect(),
        IntType::U8 => read_ref::<u8>(reference, label)
            .into_iter()
            .map(i64::from)
            .collect(),
        IntType::I16 => read_ref::<i16>(reference, label)
            .into_iter()
            .map(i64::from)
            .collect(),
        IntType::U16 => read_ref::<u16>(reference, label)
            .into_iter()
            .map(i64::from)
            .collect(),
        IntType::I32 => read_ref::<i32>(reference, label)
            .into_iter()
            .map(i64::from)
            .collect(),
        IntType::U32 => read_ref::<u32>(reference, label)
            .into_iter()
            .map(i64::from)
            .collect(),
        IntType::I64 => read_ref::<i64>(reference, label),
        IntType::U64 => read_ref::<u64>(reference, label)
            .into_iter()
            .map(|v| v as i64)
            .collect(),
    };

    let got = mine
        .read()
        .unwrap_or_else(|e| panic!("{label}: native read failed: {e}"))
        .get::<i64>()
        .unwrap_or_else(|e| panic!("{label}: native decode failed: {e}"));

    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: element count differs from netcdf-c"
    );
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, b, "{label}: element {i} differs from netcdf-c");
    }
}

/// Compare one floating-point variable, whole, by bit pattern.
fn compare_floats(
    label: &str,
    reference: &netcdf::Variable<'_>,
    mine: &Variable<'_>,
    float_type: FloatType,
) {
    let expected: Vec<f64> = match float_type {
        FloatType::F32 => read_ref::<f32>(reference, label)
            .into_iter()
            .map(f64::from)
            .collect(),
        FloatType::F64 => read_ref::<f64>(reference, label),
    };

    let got = mine
        .read()
        .unwrap_or_else(|e| panic!("{label}: native read failed: {e}"))
        .get::<f64>()
        .unwrap_or_else(|e| panic!("{label}: native decode failed: {e}"));

    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: element count differs from netcdf-c"
    );
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}: element {i} differs from netcdf-c ({a} vs {b})"
        );
    }
}

fn read_ref<T: netcdf::NcTypeDescriptor + Copy>(
    variable: &netcdf::Variable<'_>,
    label: &str,
) -> Vec<T> {
    variable
        .get_values::<T, _>(netcdf::Extents::All)
        .unwrap_or_else(|e| panic!("{label}: netcdf-c read failed: {e}"))
}

/// Every numeric variable of every corpus file must decode identically.
///
/// This is the headline check. It covers chunked and contiguous storage, the
/// shuffle and deflate filters, and both byte orders, because the corpus
/// contains all of them.
#[test]
fn every_numeric_variable_matches_netcdf_c() {
    let files = corpus();
    assert!(!files.is_empty(), "the corpus should not be empty");

    for (name, path) in files {
        let reference = netcdf::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let native = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut tally = Tally::default();

        for ref_var in reference.variables() {
            let var_name = ref_var.name();
            let label = format!("{name}:{var_name}");

            let Some(mine) = native.variable(&format!("/{var_name}")) else {
                panic!("{label}: the native reader did not find this variable");
            };

            // Shape must agree before values can mean anything.
            let ref_shape: Vec<u64> =
                ref_var.dimensions().iter().map(|d| d.len() as u64).collect();
            assert_eq!(
                mine.shape, ref_shape,
                "{label}: shape differs from netcdf-c"
            );

            match ref_var.vartype() {
                NcVariableType::Int(t) => {
                    compare_integers(&label, &ref_var, &mine, t);
                    tally.integer += 1;
                }
                NcVariableType::Float(t) => {
                    compare_floats(&label, &ref_var, &mine, t);
                    tally.float += 1;
                }
                // netcdf-c refuses to hand back NC_CHAR data as a numeric type,
                // so these are covered by the shape check above and by the
                // string tests elsewhere rather than compared element by element.
                NcVariableType::Char | NcVariableType::String => tally.skipped_char += 1,
                _ => tally.skipped_other += 1,
            }
        }

        eprintln!(
            "{name}: {} variables compared ({} int, {} float), \
             {} char/string skipped, {} other skipped",
            tally.compared(),
            tally.integer,
            tally.float,
            tally.skipped_char,
            tally.skipped_other
        );
        assert!(
            tally.compared() > 0,
            "{name}: no variable was actually compared, so this proved nothing"
        );
    }
}

/// Hyperslab reads must match netcdf-c too, not just whole-variable reads.
///
/// Whole reads would not catch an offset error in the chunk-to-output copy when
/// the selection starts partway into a chunk.
#[test]
fn hyperslab_reads_match_netcdf_c() {
    for (name, path) in corpus() {
        let reference = netcdf::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let native = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut checked = 0;

        for ref_var in reference.variables() {
            let var_name = ref_var.name();
            let label = format!("{name}:{var_name}");
            let Some(mine) = native.variable(&format!("/{var_name}")) else {
                continue;
            };
            if mine.shape.is_empty() || mine.shape.iter().any(|&d| d < 2) {
                continue; // too small for an interesting interior slab
            }
            let float_type = match ref_var.vartype() {
                NcVariableType::Float(t) => t,
                _ => continue,
            };

            // Start one element in on every axis, so the selection begins
            // partway through the first chunk.
            let ranges: Vec<std::ops::Range<usize>> = mine
                .shape
                .iter()
                .map(|&d| 1usize..(d as usize).clamp(2, 4))
                .collect();

            let expected: Vec<f64> = match float_type {
                FloatType::F32 => ref_var
                    .get_values::<f32, _>(ranges.as_slice())
                    .unwrap_or_else(|e| panic!("{label}: netcdf-c slab read failed: {e}"))
                    .into_iter()
                    .map(f64::from)
                    .collect(),
                FloatType::F64 => ref_var
                    .get_values::<f64, _>(ranges.as_slice())
                    .unwrap_or_else(|e| panic!("{label}: netcdf-c slab read failed: {e}")),
            };

            // The same selection value drives both crates, which is the point:
            // the two `get_values` calls above and below differ only in which
            // crate they call.
            let got = mine
                .get_values::<f64, _>(ranges.as_slice())
                .unwrap_or_else(|e| panic!("{label}: native slab read failed: {e}"));

            assert_eq!(
                got.len(),
                expected.len(),
                "{label}: slab element count differs"
            );
            for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{label}: slab element {i} differs from netcdf-c"
                );
            }
            checked += 1;
        }

        eprintln!("{name}: {checked} hyperslab reads match netcdf-c");
    }
}

/// Reading a variable chunk by chunk must reconstruct exactly what netcdf-c
/// returns for the whole variable.
///
/// This is the check that the chunk grid, the clipping of edge chunks and the
/// per-chunk filter pipeline are all right together.
#[test]
fn chunkwise_reads_reassemble_to_what_netcdf_c_returns() {
    for (name, path) in corpus() {
        let reference = netcdf::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let native = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut checked = 0;

        for ref_var in reference.variables() {
            let var_name = ref_var.name();
            let label = format!("{name}:{var_name}");
            let Some(mine) = native.variable(&format!("/{var_name}")) else {
                continue;
            };
            let NcVariableType::Float(FloatType::F32) = ref_var.vartype() else {
                continue;
            };
            if mine.chunk_shape().is_none() || mine.shape.len() != 2 {
                continue;
            }

            let whole: Vec<f64> = read_ref::<f32>(&ref_var, &label)
                .into_iter()
                .map(f64::from)
                .collect();

            // Rebuild the variable from its chunks and compare against the
            // reference's flat, row-major result.
            let cols = mine.shape[1] as usize;
            let mut rebuilt = vec![f64::NAN; whole.len()];
            for chunk in mine.chunks() {
                let block = mine
                    .read_chunk(&chunk)
                    .unwrap_or_else(|e| panic!("{label}: chunk read failed: {e}"))
                    .get::<f64>()
                    .unwrap();
                for row in 0..chunk.shape[0] as usize {
                    for col in 0..chunk.shape[1] as usize {
                        let global = (chunk.offset[0] as usize + row) * cols
                            + chunk.offset[1] as usize
                            + col;
                        rebuilt[global] = block[row * chunk.shape[1] as usize + col];
                    }
                }
            }

            for (i, (a, b)) in rebuilt.iter().zip(whole.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{label}: element {i} differs after chunkwise reassembly"
                );
            }
            checked += 1;
        }

        eprintln!("{name}: {checked} variables reassembled correctly from chunks");
    }
}

/// Variable and global attribute values must match netcdf-c.
///
/// Only attributes both readers surface are compared. Where this reader knows
/// its attribute list is short, it says so through `attributes_complete`, and
/// those objects are reported rather than silently passed.
#[test]
fn attribute_values_match_netcdf_c() {
    for (name, path) in corpus() {
        let reference = netcdf::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let native = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));

        let mut compared = 0;
        let mut incomplete = 0;

        for ref_var in reference.variables() {
            let var_name = ref_var.name();
            let Some(mine) = native.variable(&format!("/{var_name}")) else {
                continue;
            };
            if !mine.attributes_complete {
                incomplete += 1;
            }

            for ref_attr in ref_var.attributes() {
                let attr_name = ref_attr.name().to_string();
                let Some(got) = mine.attribute(&attr_name) else {
                    // Either a bookkeeping attribute this layer hides, or one
                    // hidden by a gapped heap, which `attributes_complete`
                    // already reports.
                    continue;
                };
                let label = format!("{name}:{var_name}:{attr_name}");

                match ref_attr.value() {
                    Ok(netcdf::AttributeValue::Str(s)) => {
                        assert_eq!(
                            got.value.as_text(),
                            Some(s.as_str()),
                            "{label}: text attribute differs"
                        );
                        compared += 1;
                    }
                    Ok(netcdf::AttributeValue::Float(f)) => {
                        assert_eq!(
                            got.value.as_f64().map(|v| v as f32),
                            Some(f),
                            "{label}: float attribute differs"
                        );
                        compared += 1;
                    }
                    Ok(netcdf::AttributeValue::Double(d)) => {
                        assert_eq!(got.value.as_f64(), Some(d), "{label}: double differs");
                        compared += 1;
                    }
                    Ok(netcdf::AttributeValue::Int(i)) => {
                        assert_eq!(
                            got.value.as_f64(),
                            Some(i as f64),
                            "{label}: int attribute differs"
                        );
                        compared += 1;
                    }
                    Ok(netcdf::AttributeValue::Short(i)) => {
                        assert_eq!(
                            got.value.as_f64(),
                            Some(i as f64),
                            "{label}: short attribute differs"
                        );
                        compared += 1;
                    }
                    // Other shapes are not modelled by `AttributeValue`'s
                    // scalar accessors; the value tests above carry the weight.
                    _ => {}
                }
            }
        }

        eprintln!(
            "{name}: {compared} attribute values match netcdf-c \
             ({incomplete} variables have a known-short attribute list)"
        );
        assert!(compared > 0, "{name}: no attribute was compared");
    }
}
