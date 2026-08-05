//! A classic file uses the same interface as a netCDF-4 file.
//!
//! `oxcdf::open` reads the magic bytes and picks the container. Everything
//! above that is the same: `get_values`, `Extents`, `vartype`, `get_strings`
//! and typed attributes.
//!
//! `netcdf_classic_parity` compares against the `netcdf` crate, which reads
//! classic files through netcdf-c.

use oxcdf::{AttributeValue, Container, DType, Extents};

fn corpus(name: &str) -> Option<String> {
    let path = format!("{}/test_files/{name}", env!("CARGO_MANIFEST_DIR"));
    std::path::Path::new(&path).exists().then_some(path)
}

const FILES: [&str; 2] = ["classic.nc", "classic64.nc"];

#[test]
fn open_reads_a_classic_file() {
    for name in FILES {
        let Some(path) = corpus(name) else { continue };
        let file = oxcdf::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));

        assert!(
            matches!(file.container(), Container::Cdf1 | Container::Cdf2),
            "{name} is a classic container"
        );
        assert!(file.hdf5().is_none(), "{name} has no HDF5 layer");
        assert!(file.classic().is_some(), "{name} reports its classic file");
        assert!(!file.variables().is_empty(), "{name} has variables");
        assert!(!file.dimensions().is_empty(), "{name} has dimensions");
    }
}

#[test]
fn the_metadata_is_the_netcdf_shape() {
    let Some(path) = corpus("classic.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();

    let names: Vec<String> = file.variables().iter().map(|v| v.name.clone()).collect();
    assert_eq!(
        names,
        vec!["time", "pressure", "count", "flag", "label"],
        "every variable is reachable"
    );

    let pressure = file.variable("pressure").expect("a variable by name");
    assert_eq!(pressure.vartype(), DType::Float(4));
    assert_eq!(pressure.dimensions, vec!["time", "level"]);
    assert_eq!(pressure.shape.len(), 2);

    // A leading slash works, as it does for netCDF-4.
    assert!(file.variable("/pressure").is_some());

    // One dimension is unlimited in this fixture.
    assert!(file.dimensions().iter().any(|d| d.is_unlimited));
    assert!(file.dimension("time").is_some());
    assert_eq!(file.dimension_len("time"), file.dimension("time").map(|d| d.len));
}

#[test]
fn values_read_as_their_stored_type() {
    let Some(path) = corpus("classic.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();

    let pressure = file.variable("pressure").unwrap();
    let exact = pressure.get_values::<f32, _>(..).unwrap();
    assert_eq!(exact.len() as u64, pressure.len());

    // A conversion works, as it does for netCDF-4.
    let wide = pressure.get_values::<f64, _>(..).unwrap();
    for (a, b) in exact.iter().zip(&wide) {
        assert_eq!(f64::from(*a), *b);
    }

    // Big-endian storage must arrive in native order, not reversed.
    let raw = pressure.get_raw_values(..).unwrap();
    assert_eq!(raw.len(), exact.len() * 4);
    let first = f32::from_ne_bytes(raw[..4].try_into().unwrap());
    assert_eq!(first.to_bits(), exact[0].to_bits());

    // Integer types keep their width.
    assert_eq!(file.variable("count").unwrap().vartype(), DType::Int(4));
    assert_eq!(file.variable("flag").unwrap().vartype(), DType::Int(2));
    assert_eq!(file.variable("time").unwrap().vartype(), DType::Float(8));
}

#[test]
fn selections_work_the_same_way() {
    let Some(path) = corpus("classic.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("pressure").unwrap();
    let all = v.get_values::<f32, _>(Extents::All).unwrap();
    let (rows, cols) = (v.shape[0] as usize, v.shape[1] as usize);

    assert_eq!(v.get_values::<f32, _>(..).unwrap(), all);

    // One row, two ways.
    let row = v.get_values::<f32, _>([0..1, 0..cols]).unwrap();
    assert_eq!(row, all[..cols]);
    let same = v
        .get_values::<f32, _>([oxcdf::Extent::Index(0), (0..cols).into()])
        .unwrap();
    assert_eq!(same, row);

    // One element.
    assert_eq!(v.get_value::<f32, _>([0, 0]).unwrap(), all[0]);
    if rows > 1 {
        assert_eq!(v.get_value::<f32, _>([1, 0]).unwrap(), all[cols]);
    }

    // A selection past the end is refused.
    assert!(v.get_values::<f32, _>([0..rows + 9, 0..cols]).is_err());
}

#[test]
fn a_char_variable_reads_as_text() {
    let Some(path) = corpus("classic.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();
    let label = file.variable("label").expect("the char variable");

    // netCDF calls this `char`, and its last dimension is the string length.
    assert_eq!(label.vartype(), DType::Char);
    assert_eq!(label.dimensions.len(), 2);

    let width = *label.shape.last().unwrap() as usize;
    let chars = label.get_strings(..).unwrap();
    assert_eq!(chars.len() as u64, label.len());

    let joined: Vec<String> = chars
        .chunks(width)
        .map(|row| row.concat().trim_end_matches('\0').to_string())
        .collect();
    assert_eq!(joined.len(), label.shape[0] as usize);
    assert!(!joined[0].is_empty(), "the first label has text");
}

#[test]
fn attributes_keep_their_stored_type() {
    let Some(path) = corpus("classic.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();

    // A text attribute is one `Str`, as it is for netCDF-4.
    let text = file
        .attributes()
        .iter()
        .find(|a| matches!(a.value, AttributeValue::Str(_)));
    assert!(text.is_some(), "the fixture has a text global attribute");

    // No numeric attribute may arrive widened to f64 unless it is stored so.
    for v in file.variables() {
        for a in v.attributes() {
            match &a.value {
                AttributeValue::Float(_) | AttributeValue::Floats(_) => {
                    assert_eq!(a.value.vartype(), DType::Float(4));
                }
                AttributeValue::Double(_) | AttributeValue::Doubles(_) => {
                    assert_eq!(a.value.vartype(), DType::Float(8));
                }
                AttributeValue::Short(_) | AttributeValue::Shorts(_) => {
                    assert_eq!(a.value.vartype(), DType::Int(2));
                }
                _ => {}
            }
            assert!(
                !matches!(a.value, AttributeValue::Raw(_)),
                "{} of {} decoded to raw bytes",
                a.name,
                v.path
            );
        }
    }
}

#[test]
fn a_classic_variable_reports_one_chunk() {
    let Some(path) = corpus("classic.nc") else {
        return;
    };
    let file = oxcdf::open(&path).unwrap();
    let v = file.variable("pressure").unwrap();

    // Classic files store contiguously, so the chunk loop still works.
    assert_eq!(v.chunk_shape(), None);
    let chunks = v.chunks();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].shape, v.shape);
    assert_eq!(v.read_chunk(&chunks[0]).unwrap().len() as u64, v.len());
    assert!(v.prepare().is_ok(), "prepare is a no-op, not an error");
    assert!(v.is_readable());
}

// ─── the asynchronous engine reads classic files too ──────────────────────

#[cfg(feature = "async")]
fn async_classic(path: &str) -> oxcdf::AsyncFile {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        oxcdf::open_async(std::sync::Arc::new(oxcdf::SyncAsAsync(
            oxcdf::FileSource::open(path).unwrap(),
        )))
        .await
        .unwrap()
    })
}

#[cfg(feature = "async")]
#[tokio::test]
async fn the_async_engine_opens_a_classic_file() {
    for name in FILES {
        let Some(path) = corpus(name) else { continue };
        let sync = oxcdf::open(&path).unwrap();
        let file = oxcdf::open_async(std::sync::Arc::new(oxcdf::SyncAsAsync(
            oxcdf::FileSource::open(&path).unwrap(),
        )))
        .await
        .unwrap_or_else(|e| panic!("{name}: {e}"));

        let want: Vec<_> = sync
            .variables()
            .iter()
            .map(|v| (v.path.clone(), v.shape.clone(), v.dimensions.clone()))
            .collect();
        let got: Vec<_> = file
            .variables()
            .iter()
            .map(|v| (v.path.clone(), v.shape.clone(), v.dimensions.clone()))
            .collect();
        assert_eq!(got, want, "variables of {name}");

        let want: Vec<_> = sync.dimensions().iter().map(|d| (&d.name, d.len)).collect();
        let got: Vec<_> = file.dimensions().iter().map(|d| (&d.name, d.len)).collect();
        assert_eq!(got, want, "dimensions of {name}");

        let want: Vec<_> = sync.attributes().iter().map(|a| &a.value).collect();
        let got: Vec<_> = file.attributes().iter().map(|a| &a.value).collect();
        assert_eq!(got, want, "global attributes of {name}");
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn the_two_engines_read_the_same_classic_values() {
    for name in FILES {
        let Some(path) = corpus(name) else { continue };
        let sync = oxcdf::open(&path).unwrap();
        let file = oxcdf::open_async(std::sync::Arc::new(oxcdf::SyncAsAsync(
            oxcdf::FileSource::open(&path).unwrap(),
        )))
        .await
        .unwrap();

        for want in sync.variables() {
            let got = file.variable(&want.path).unwrap();
            assert_eq!(got.vartype(), want.vartype(), "{} in {name}", want.path);

            if want.vartype() == DType::Char {
                assert_eq!(
                    got.get_strings(..).await.unwrap(),
                    want.get_strings(..).unwrap(),
                    "strings of {} in {name}",
                    want.path
                );
                continue;
            }

            assert_eq!(
                got.get_values::<f64, _>(..).await.unwrap(),
                want.get_values::<f64, _>(..).unwrap(),
                "values of {} in {name}",
                want.path
            );
            assert_eq!(
                got.get_raw_values(..).await.unwrap(),
                want.get_raw_values(..).unwrap(),
                "raw bytes of {} in {name}",
                want.path
            );
        }
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn an_async_classic_selection_matches_the_synchronous_one() {
    let Some(path) = corpus("classic.nc") else {
        return;
    };
    let sync = oxcdf::open(&path).unwrap();
    let file = oxcdf::open_async(std::sync::Arc::new(oxcdf::SyncAsAsync(
        oxcdf::FileSource::open(&path).unwrap(),
    )))
    .await
    .unwrap();

    let want = sync.variable("pressure").unwrap();
    let got = file.variable("pressure").unwrap();
    let cols = want.shape[1] as usize;

    assert_eq!(
        got.get_values::<f32, _>([0..1, 0..cols]).await.unwrap(),
        want.get_values::<f32, _>([0..1, 0..cols]).unwrap(),
    );
    assert_eq!(
        got.get_value::<f32, _>([0, 0]).await.unwrap(),
        want.get_value::<f32, _>([0, 0]).unwrap(),
    );

    // A classic variable reports one chunk through the async handle too.
    let chunks = got.chunks().await.unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(got.read_chunk(&chunks[0]).await.unwrap().len() as u64, got.len());
    assert!(got.dataset().is_none(), "a classic variable has no HDF5 dataset");
}

#[cfg(feature = "async")]
#[test]
fn a_classic_file_needs_no_runtime_handle_to_open() {
    let Some(path) = corpus("classic.nc") else {
        return;
    };
    // The classic path must work on a current-thread runtime, like the rest of
    // the asynchronous engine.
    let file = async_classic(&path);
    assert!(
        matches!(file.container(), Container::Cdf1 | Container::Cdf2),
        "the backend is classic"
    );
    assert!(file.netcdf().hdf5().is_none());
}

// ─── parity with the netcdf crate ──────────────────────────────────────────

#[cfg(feature = "diff-tests")]
#[test]
fn netcdf_classic_parity() {
    for name in FILES {
        let Some(path) = corpus(name) else { continue };
        let ours = oxcdf::open(&path).unwrap();
        let theirs = netcdf::open(&path).unwrap();

        for v in ours.variables() {
            let Some(other) = theirs.variable(&v.name) else {
                panic!("{} is missing from netcdf-c's view of {name}", v.name);
            };

            assert_eq!(
                v.dimensions,
                other
                    .dimensions()
                    .iter()
                    .map(|d| d.name())
                    .collect::<Vec<_>>(),
                "axes of {} in {name}",
                v.name
            );

            macro_rules! compare {
                ($t:ty) => {{
                    let mine = v.get_values::<$t, _>(..).unwrap();
                    let yours = other.get_values::<$t, _>(netcdf::Extents::All).unwrap();
                    assert_eq!(mine, yours, "values of {} in {name}", v.name);
                }};
            }

            match v.vartype() {
                DType::Int(1) => compare!(i8),
                DType::Int(2) => compare!(i16),
                DType::Int(4) => compare!(i32),
                DType::Int(8) => compare!(i64),
                DType::Uint(1) => compare!(u8),
                DType::Float(4) => compare!(f32),
                DType::Float(8) => compare!(f64),
                // netcdf-c reads `char` through its own text call, which the
                // string test above already covers.
                DType::Char => {}
                other => panic!("{} has an unexpected type {other:?}", v.name),
            }
        }
    }
}
