//! Attribute values keep the type the file stores.
//!
//! `AttributeValue` mirrors `netcdf::AttributeValue`, variant for variant.
//! `netcdf_attribute_parity` compares them directly when `diff-tests` is on.

use oxcdf::{AttributeValue, DType};

fn corpus(name: &str) -> Option<String> {
    let path = format!("{}/../../test_files/{name}", env!("CARGO_MANIFEST_DIR"));
    std::path::Path::new(&path).exists().then_some(path)
}

const FILES: [&str; 4] = [
    "test_file.nc",
    "gridded-example.nc",
    "wod_ctd_1964.nc",
    "legacy_v1_objheader.h5",
];

// ─── the value keeps its stored type ───────────────────────────────────────

#[test]
fn a_text_attribute_is_one_string() {
    let Some(path) = corpus("legacy_v1_objheader.h5") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();

    let title = file.attribute("title").expect("a global attribute");
    assert!(
        matches!(&title.value, AttributeValue::Str(s) if s == "legacy fixture"),
        "a text attribute is one Str, not a list of characters: {:?}",
        title.value
    );
    assert_eq!(title.value.vartype(), DType::String);
    assert_eq!(title.value.len(), 1);
}

#[test]
fn a_float_attribute_keeps_its_width() {
    let Some(path) = corpus("test_file.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();

    // Every numeric attribute must report a width, and it must be the width the
    // variable's own type would use. Nothing may silently widen to f64.
    let mut seen_narrow = false;
    for v in file.variables() {
        for a in v.attributes() {
            match &a.value {
                AttributeValue::Float(_) | AttributeValue::Floats(_) => {
                    assert_eq!(a.value.vartype(), DType::Float(4));
                    seen_narrow = true;
                }
                AttributeValue::Double(_) | AttributeValue::Doubles(_) => {
                    assert_eq!(a.value.vartype(), DType::Float(8));
                }
                _ => {}
            }
        }
    }
    assert!(
        seen_narrow,
        "the Argo file has f32 attributes; they must not arrive as f64"
    );
}

#[test]
fn an_integer_attribute_keeps_its_width_and_sign() {
    for name in FILES {
        let Some(path) = corpus(name) else { continue };
        let file = oxcdf::open(&path).unwrap();

        for v in file.variables() {
            for a in v.attributes() {
                let want = match &a.value {
                    AttributeValue::Schar(_) | AttributeValue::Schars(_) => DType::Int(1),
                    AttributeValue::Short(_) | AttributeValue::Shorts(_) => DType::Int(2),
                    AttributeValue::Int(_) | AttributeValue::Ints(_) => DType::Int(4),
                    AttributeValue::Longlong(_) | AttributeValue::Longlongs(_) => DType::Int(8),
                    AttributeValue::Uchar(_) | AttributeValue::Uchars(_) => DType::Uint(1),
                    AttributeValue::Ushort(_) | AttributeValue::Ushorts(_) => DType::Uint(2),
                    AttributeValue::Uint(_) | AttributeValue::Uints(_) => DType::Uint(4),
                    AttributeValue::Ulonglong(_) | AttributeValue::Ulonglongs(_) => DType::Uint(8),
                    _ => continue,
                };
                assert_eq!(a.value.vartype(), want, "{} in {name}", a.name);
            }
        }
    }
}

#[test]
fn one_value_is_singular_and_several_are_plural() {
    for name in FILES {
        let Some(path) = corpus(name) else { continue };
        let file = oxcdf::open(&path).unwrap();

        for v in file.variables() {
            for a in v.attributes() {
                let singular = matches!(
                    &a.value,
                    AttributeValue::Uchar(_)
                        | AttributeValue::Schar(_)
                        | AttributeValue::Ushort(_)
                        | AttributeValue::Short(_)
                        | AttributeValue::Uint(_)
                        | AttributeValue::Int(_)
                        | AttributeValue::Ulonglong(_)
                        | AttributeValue::Longlong(_)
                        | AttributeValue::Float(_)
                        | AttributeValue::Double(_)
                        | AttributeValue::Str(_)
                );
                if singular {
                    assert_eq!(a.value.len(), 1, "{} in {name} is singular", a.name);
                }
            }
        }
    }
}

#[test]
fn a_fill_value_attribute_matches_its_variable_type() {
    for name in FILES {
        let Some(path) = corpus(name) else { continue };
        let file = oxcdf::open(&path).unwrap();

        for v in file.variables() {
            let Some(fill) = v.attribute("_FillValue") else {
                continue;
            };
            // netcdf-c writes `_FillValue` in the variable's own type. A reader
            // that widened it would produce a value that never compares equal.
            //
            // A `char` variable is the one exception: netCDF has no `char`
            // attribute value, so its fill value arrives as text. The `netcdf`
            // crate does the same.
            let want = match v.vartype() {
                DType::Char => DType::String,
                other => other,
            };
            assert_eq!(
                fill.value.vartype(),
                want,
                "_FillValue of {} in {name}",
                v.path
            );
        }
    }
}

#[test]
fn as_f64_still_reaches_any_number() {
    for name in FILES {
        let Some(path) = corpus(name) else { continue };
        let file = oxcdf::open(&path).unwrap();

        for v in file.variables() {
            for a in v.attributes() {
                let numeric = a.value.vartype().is_integer() || a.value.vartype().is_float();
                assert_eq!(
                    a.value.as_f64().is_some(),
                    numeric && !a.value.is_empty(),
                    "as_f64 on {} in {name}: {:?}",
                    a.name,
                    a.value
                );
            }
        }
    }
}

// ─── both engines agree ────────────────────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test]
async fn the_async_engine_decodes_the_same_attributes() {
    for name in FILES {
        let Some(path) = corpus(name) else { continue };
        let sync = oxcdf::open(&path).unwrap();
        let file = oxcdf::open_async(std::sync::Arc::new(oxcdf::SyncAsAsync(
            oxcdf::FileSource::open(&path).unwrap(),
        )))
        .await
        .unwrap();

        let want: Vec<_> = sync.attributes().iter().map(|a| &a.value).collect();
        let got: Vec<_> = file.attributes().iter().map(|a| &a.value).collect();
        assert_eq!(got, want, "global attributes of {name}");

        for w in sync.variables() {
            let g = file.variable(&w.path).unwrap();
            let want: Vec<_> = w.attributes.iter().map(|a| (&a.name, &a.value)).collect();
            let got: Vec<_> = g.attributes().iter().map(|a| (&a.name, &a.value)).collect();
            assert_eq!(got, want, "attributes of {} in {name}", w.path);
        }
    }
}

// ─── parity with the netcdf crate ──────────────────────────────────────────

/// Render an attribute value as a tag plus its contents.
///
/// Both enums use the same variant names, so one macro serves both and the
/// comparison is exact rather than approximate.
#[cfg(feature = "diff-tests")]
macro_rules! canonical {
    ($value:expr, $enum:path) => {{
        use $enum as A;
        match $value {
            A::Uchar(x) => format!("Uchar {x:?}"),
            A::Uchars(x) => format!("Uchars {x:?}"),
            A::Schar(x) => format!("Schar {x:?}"),
            A::Schars(x) => format!("Schars {x:?}"),
            A::Ushort(x) => format!("Ushort {x:?}"),
            A::Ushorts(x) => format!("Ushorts {x:?}"),
            A::Short(x) => format!("Short {x:?}"),
            A::Shorts(x) => format!("Shorts {x:?}"),
            A::Uint(x) => format!("Uint {x:?}"),
            A::Uints(x) => format!("Uints {x:?}"),
            A::Int(x) => format!("Int {x:?}"),
            A::Ints(x) => format!("Ints {x:?}"),
            A::Ulonglong(x) => format!("Ulonglong {x:?}"),
            A::Ulonglongs(x) => format!("Ulonglongs {x:?}"),
            A::Longlong(x) => format!("Longlong {x:?}"),
            A::Longlongs(x) => format!("Longlongs {x:?}"),
            // Compare floats by bit pattern, as the value tests do.
            A::Float(x) => format!("Float {:?}", x.to_bits()),
            A::Floats(x) => format!(
                "Floats {:?}",
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            ),
            A::Double(x) => format!("Double {:?}", x.to_bits()),
            A::Doubles(x) => format!(
                "Doubles {:?}",
                x.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            ),
            A::Str(x) => format!("Str {x:?}"),
            A::Strs(x) => format!("Strs {x:?}"),
            #[allow(unreachable_patterns)]
            other => format!("{other:?}"),
        }
    }};
}

#[cfg(feature = "diff-tests")]
#[test]
fn netcdf_attribute_parity() {
    let mut compared = 0usize;

    for name in ["test_file.nc", "gridded-example.nc", "wod_ctd_1964.nc"] {
        let Some(path) = corpus(name) else { continue };
        let ours = oxcdf::open(&path).unwrap();
        let theirs = netcdf::open(&path).unwrap();

        // Global attributes.
        for a in ours.attributes() {
            let Some(other) = theirs.attribute(&a.name) else {
                continue;
            };
            let want = canonical!(&other.value().unwrap(), netcdf::AttributeValue);
            let got = canonical!(&a.value, oxcdf::AttributeValue);
            assert_eq!(got, want, "global attribute {} of {name}", a.name);
            compared += 1;
        }

        // Variable attributes.
        for v in ours.variables() {
            let Some(other_var) = theirs.variable(&v.name) else {
                continue;
            };
            for a in v.attributes() {
                let Some(other) = other_var.attribute(&a.name) else {
                    continue;
                };
                let want = canonical!(&other.value().unwrap(), netcdf::AttributeValue);
                let got = canonical!(&a.value, oxcdf::AttributeValue);
                assert_eq!(got, want, "attribute {} of {} in {name}", a.name, v.path);
                compared += 1;
            }
        }
    }

    assert!(compared > 50, "only {compared} attributes were compared");
    println!("compared {compared} attributes against netcdf-c");
}
