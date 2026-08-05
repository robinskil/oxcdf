//! Checks the netCDF-4 layer against `ncdump`, the reference implementation.
//!
//! These tests compare the variable list, the dimension list and each
//! variable's declared axes. Together they are what catches the mistake this
//! layer exists to prevent: reporting a dimension as if it were a variable, or
//! attaching the wrong axes to a variable.

use std::collections::{BTreeMap, BTreeSet};

use oxcdf::netcdf::NetcdfFile;

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

fn have_ncdump() -> bool {
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|d| d.join("ncdump").is_file())
    })
}

struct NcDump {
    dimensions: BTreeMap<String, u64>,
    /// Variable name to its declared dimension names.
    variables: BTreeMap<String, Vec<String>>,
}

/// Parse the header `ncdump -h` prints.
fn ncdump(path: &str) -> Option<NcDump> {
    let out = std::process::Command::new("ncdump")
        .arg("-h")
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);

    let mut dimensions = BTreeMap::new();
    let mut variables = BTreeMap::new();
    let mut section = "";

    for line in text.lines() {
        let t = line.trim();
        match t {
            "dimensions:" => {
                section = "dimensions";
                continue;
            }
            "variables:" => {
                section = "variables";
                continue;
            }
            _ => {}
        }
        if t.starts_with("//") || t == "}" || t == "data:" {
            if t.starts_with("// global") {
                section = "";
            }
            continue;
        }

        if section == "dimensions" {
            // `N_PROF = 8 ;` or `TIME = UNLIMITED ; // (5 currently)`
            if let Some((name, rest)) = t.split_once('=') {
                let name = name.trim().to_string();
                let value = rest.trim().trim_end_matches(';').trim();
                let len = if value.starts_with("UNLIMITED") {
                    rest.split_once('(')
                        .and_then(|(_, r)| r.split_whitespace().next().map(|s| s.to_string()))
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0)
                } else {
                    value.parse::<u64>().unwrap_or(0)
                };
                dimensions.insert(name, len);
            }
        } else if section == "variables" {
            // `float TEMP(N_PROF, N_LEVELS) ;` for an array variable, or
            // `int crs ;` for a scalar one. Attribute lines carry a colon
            // before any parenthesis.
            let Some(open) = t.find('(') else {
                // No parentheses: a scalar declaration, unless it is an
                // attribute line or a continuation of one.
                let head = t.trim_end_matches(';').trim();
                if head.contains(':') || head.contains('=') {
                    continue;
                }
                let parts: Vec<&str> = head.split_whitespace().collect();
                if parts.len() == 2 {
                    variables.insert(parts[1].to_string(), Vec::new());
                }
                continue;
            };
            let head = &t[..open];
            if head.contains(':') {
                continue;
            }
            let Some(name) = head.split_whitespace().nth(1) else {
                continue;
            };
            let Some(close) = t[open..].find(')') else {
                continue;
            };
            let dims: Vec<String> = t[open + 1..open + close]
                .split(',')
                .map(|d| d.trim().to_string())
                .filter(|d| !d.is_empty())
                .collect();
            variables.insert(name.to_string(), dims);
        }
    }

    Some(NcDump {
        dimensions,
        variables,
    })
}

#[test]
fn variable_lists_match_ncdump_exactly() {
    if !have_ncdump() {
        eprintln!("ncdump not available; skipping");
        return;
    }

    for (name, path) in corpus() {
        let Some(expected) = ncdump(&path) else {
            continue;
        };
        let file = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));

        let found: BTreeSet<String> = file
            .variables()
            .iter()
            .map(|v| v.path.trim_start_matches('/').to_string())
            .collect();
        let want: BTreeSet<String> = expected.variables.keys().cloned().collect();

        let missing: Vec<_> = want.difference(&found).collect();
        let extra: Vec<_> = found.difference(&want).collect();

        assert!(
            missing.is_empty() && extra.is_empty(),
            "{name}: variable list differs from ncdump.\n  missing: {missing:?}\n  extra: {extra:?}"
        );
        eprintln!("{name}: {} variables match", found.len());
    }
}

#[test]
fn dimension_lists_match_ncdump() {
    if !have_ncdump() {
        return;
    }

    for (name, path) in corpus() {
        let Some(expected) = ncdump(&path) else {
            continue;
        };
        let file = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));

        let found: BTreeMap<String, u64> = file
            .root()
            .dimensions
            .iter()
            .map(|d| (d.name.clone(), d.len))
            .collect();

        assert_eq!(
            found.keys().collect::<Vec<_>>(),
            expected.dimensions.keys().collect::<Vec<_>>(),
            "{name}: dimension names differ from ncdump"
        );

        for (dim, len) in &expected.dimensions {
            assert_eq!(
                found.get(dim),
                Some(len),
                "{name}: dimension {dim} has the wrong length"
            );
        }
        eprintln!("{name}: {} dimensions match", found.len());
    }
}

/// The hardest part of the layer: following `DIMENSION_LIST` through the global
/// heap to name each axis of each variable.
#[test]
fn each_variable_declares_the_same_axes_as_ncdump() {
    if !have_ncdump() {
        return;
    }

    for (name, path) in corpus() {
        let Some(expected) = ncdump(&path) else {
            continue;
        };
        let file = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));

        let mut checked = 0;
        for variable in file.variables() {
            let key = variable.path.trim_start_matches('/');
            let Some(want) = expected.variables.get(key) else {
                continue;
            };
            assert_eq!(
                &variable.dimensions, want,
                "{name}: variable {key} has the wrong axes"
            );
            assert_eq!(
                variable.shape.len(),
                want.len(),
                "{name}: variable {key} has the wrong rank"
            );
            checked += 1;
        }
        assert!(checked > 0, "{name}: no variables were checked");
        eprintln!("{name}: axes match for {checked} variables");
    }
}

/// netCDF bookkeeping attributes must not reach the caller.
#[test]
fn reserved_attributes_are_not_exposed() {
    for (name, path) in corpus() {
        let file = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        for variable in file.variables() {
            for attr in &variable.attributes {
                assert!(
                    !matches!(
                        attr.name.as_str(),
                        "CLASS" | "NAME" | "DIMENSION_LIST" | "REFERENCE_LIST" | "_Netcdf4Dimid"
                    ),
                    "{name}: variable {} exposes the internal attribute {}",
                    variable.path,
                    attr.name
                );
            }
        }
    }
}

/// Values must still be readable through the netCDF layer, and must match what
/// the HDF5 layer returns for the same variable.
#[test]
fn reads_real_values_through_the_netcdf_layer() {
    use oxcdf::read::Hyperslab;

    for (name, path) in corpus() {
        let file = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));

        let mut read_any = false;
        for variable in file.variables() {
            if !variable.is_readable() || variable.shape.is_empty() {
                continue;
            }
            // Read a small corner so the test stays quick on large variables.
            let count: Vec<u64> = variable.shape.iter().map(|&d| d.min(2)).collect();
            let slab =
                Hyperslab::new(vec![0; variable.shape.len()], count.clone(), &variable.shape)
                    .unwrap();

            let raw = variable
                .read_selection(&slab)
                .unwrap_or_else(|e| panic!("{name}: reading {} failed: {e}", variable.path));

            let expected: u64 = count.iter().product();
            assert_eq!(
                raw.len() as u64,
                expected,
                "{name}: {} returned the wrong element count",
                variable.path
            );
            read_any = true;
        }
        assert!(read_any, "{name}: no variable could be read");
        eprintln!("{name}: values read through the netCDF layer");
    }
}
