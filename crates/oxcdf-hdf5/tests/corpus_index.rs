//! Indexes every netCDF-4 file in the repository and checks the result against
//! what `ncdump` reports, so the structural parse is measured against the
//! reference implementation rather than against itself.

use oxcdf_hdf5::index::Hdf5File;

fn corpus() -> Vec<(&'static str, String)> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files");
    [
        ("test_file.nc", "test_file.nc"),
        ("gridded-example.nc", "gridded-example.nc"),
        ("wod_ctd_1964.nc", "wod_ctd_1964.nc"),
    ]
    .iter()
    .map(|(name, p)| (*name, format!("{root}/{p}")))
    .filter(|(_, p)| std::path::Path::new(p).exists())
    .collect()
}

#[test]
fn indexes_every_corpus_file() {
    let files = corpus();
    assert!(!files.is_empty(), "the corpus should not be empty");

    for (name, path) in files {
        let file = Hdf5File::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let datasets = file.datasets();
        assert!(
            !datasets.is_empty(),
            "{name} should contain datasets, found none"
        );

        for d in &datasets {
            assert!(
                !d.shape.is_empty() || d.element_count() == 1,
                "{name}: {} has an implausible shape {:?}",
                d.path,
                d.shape
            );
        }

        eprintln!(
            "{name}: {} datasets, {} root attributes",
            datasets.len(),
            file.root().attributes.len()
        );
    }
}

/// Every netCDF variable must appear as an HDF5 dataset.
///
/// This is the check that catches a group walk that silently stops early, which
/// is the failure mode that matters most for dense link storage: a missing link
/// would quietly hide a variable.
///
/// The reverse does not hold, and must not be asserted. A netCDF dimension is
/// stored as an HDF5 dataset too, so this reader legitimately reports more
/// datasets than `ncdump` reports variables. Telling the two apart is the job of
/// the netCDF layer, which reads the dimension-scale attributes.
#[test]
fn every_ncdump_variable_appears_as_a_dataset() {
    if which_ncdump().is_none() {
        eprintln!("ncdump not available; skipping the cross-check");
        return;
    }

    for (name, path) in corpus() {
        let expected = ncdump_variable_names(&path);
        if expected.is_empty() {
            continue;
        }

        let file = Hdf5File::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let found: std::collections::HashSet<String> = file
            .datasets()
            .iter()
            .map(|d| d.path.trim_start_matches('/').to_string())
            .collect();

        let missing: Vec<&String> = expected.iter().filter(|v| !found.contains(*v)).collect();
        assert!(
            missing.is_empty(),
            "{name}: ncdump reports variables the reader never found: {missing:?}"
        );

        eprintln!(
            "{name}: {} ncdump variables, all present among {} HDF5 datasets",
            expected.len(),
            found.len()
        );
    }
}

fn which_ncdump() -> Option<std::path::PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join("ncdump"))
            .find(|p| p.is_file())
    })
}

/// Variable names as `ncdump -h` reports them.
fn ncdump_variable_names(path: &str) -> Vec<String> {
    let out = match std::process::Command::new("ncdump")
        .arg("-h")
        .arg(path)
        .output()
    {
        Ok(o) if o.status.success() => o.stdout,
        _ => return Vec::new(),
    };
    let text = String::from_utf8_lossy(&out);

    let mut names = Vec::new();
    let mut in_variables = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed == "variables:" {
            in_variables = true;
            continue;
        }
        if in_variables
            && (trimmed.starts_with("// global") || trimmed == "}" || trimmed == "data:")
        {
            break;
        }
        if !in_variables {
            continue;
        }
        // A declaration looks like `float TEMP(N_PROF, N_LEVELS) ;`.
        // Attribute lines start with the variable name followed by a colon.
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        if let Some(paren) = trimmed.find('(') {
            let head = &trimmed[..paren];
            if head.contains(':') {
                continue; // an attribute, not a declaration
            }
            if let Some(name) = head.split_whitespace().nth(1) {
                names.push(name.to_string());
            }
        }
    }
    names
}
