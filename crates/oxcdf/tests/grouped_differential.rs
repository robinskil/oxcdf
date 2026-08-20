//! Differential tests over a grouped file, against netcdf-c.
//!
//! `differential.rs` walks the root group of the corpus. This suite walks a
//! whole tree, so it covers what a grouped file adds: a variable found by path
//! rather than by name, and a dimension this reader invented rather than read.
//!
//! Every variable is compared whole, element by element, at every depth. A
//! large one is compared in slabs so the comparison does not need the whole
//! variable twice in memory.
//!
//! The corpus fixture is small. Point `OXCDF_GROUPED_FILE` at any grouped file
//! to run the same comparison over that instead:
//!
//! ```bash
//! OXCDF_GROUPED_FILE=/path/to/file.h5 \
//!   cargo test -p oxcdf --features diff-tests --test grouped_differential -- --nocapture
//! ```
#![cfg(feature = "diff-tests")]

use netcdf::types::{FloatType, IntType, NcVariableType};
use oxcdf::netcdf::{NetcdfFile, Variable};

/// Elements to compare in one go. A larger variable is walked in slabs along
/// its first axis, so neither reader holds all of it at once.
const SLAB_ELEMENTS: u64 = 1 << 20;

fn files() -> Vec<(String, String)> {
    let mut out = vec![];

    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_files/nested_groups.h5"
    );
    if std::path::Path::new(fixture).exists() {
        out.push(("nested_groups.h5".to_string(), fixture.to_string()));
    }

    // An instrument file is far larger than anything this repository ships, so
    // it is opt in rather than committed.
    if let Some(path) = std::env::var_os("OXCDF_GROUPED_FILE") {
        let path = expand_home(&path.to_string_lossy());
        assert!(
            std::path::Path::new(&path).exists(),
            "OXCDF_GROUPED_FILE points at {path}, which does not exist"
        );
        let name = std::path::Path::new(&path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());
        out.push((name, path));
    }

    out
}

fn expand_home(path: &str) -> String {
    match path.strip_prefix("~/") {
        Some(rest) => match std::env::var("HOME") {
            Ok(home) => format!("{home}/{rest}"),
            Err(_) => path.to_string(),
        },
        None => path.to_string(),
    }
}

/// What the comparison did, so a pass reports coverage instead of passing
/// silently because everything was skipped.
#[derive(Default, Debug)]
struct Tally {
    integer: usize,
    float: usize,
    text: usize,
    /// Compound, enum and opaque: netcdf-c hands back no scalar type for one,
    /// so only the byte count is checked.
    opaque: usize,
    empty: usize,
    elements: u64,
    bytes: u64,
}

impl Tally {
    fn variables(&self) -> usize {
        self.integer + self.float + self.text + self.opaque + self.empty
    }
}

/// Walk the reference tree, handing each variable and its full path to `f`.
///
/// A netcdf-c group handle borrows from its parent, so a path lookup cannot
/// hand one back. The walk keeps the whole chain alive instead.
fn for_each_reference_variable(
    group: &netcdf::Group<'_>,
    prefix: &str,
    f: &mut impl FnMut(&str, &netcdf::Variable<'_>),
) {
    for v in group.variables() {
        f(&format!("{prefix}/{}", v.name()), &v);
    }
    for child in group.groups() {
        let name = child.name();
        for_each_reference_variable(&child, &format!("{prefix}/{name}"), f);
    }
}

/// The slabs to compare a variable in: whole if it is small, otherwise blocks
/// of rows along axis 0.
fn slabs(shape: &[u64]) -> Vec<Vec<std::ops::Range<usize>>> {
    let total: u64 = shape.iter().product();
    if shape.is_empty() || total <= SLAB_ELEMENTS {
        return vec![shape.iter().map(|&d| 0..d as usize).collect()];
    }

    let row: u64 = shape[1..].iter().product::<u64>().max(1);
    let rows_per_slab = (SLAB_ELEMENTS / row).max(1);

    let mut out = vec![];
    let mut start = 0u64;
    while start < shape[0] {
        let end = (start + rows_per_slab).min(shape[0]);
        // One block of rows on axis 0, the whole extent on the rest.
        let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(shape.len());
        ranges.push(start as usize..end as usize);
        ranges.extend(shape[1..].iter().map(|&d| 0..d as usize));
        out.push(ranges);
        start = end;
    }
    out
}

/// Compare one integer variable, slab by slab.
fn compare_integers(
    label: &str,
    reference: &netcdf::Variable<'_>,
    mine: &Variable<'_>,
    int_type: IntType,
    tally: &mut Tally,
) {
    for ranges in slabs(&mine.shape) {
        let r = ranges.as_slice();
        // Read through netcdf-c at the stored width, then widen, so no
        // conversion happens inside the C library that could mask a difference.
        let expected: Vec<i64> = match int_type {
            IntType::I8 => widen::<i8>(reference, r, label),
            IntType::U8 => widen::<u8>(reference, r, label),
            IntType::I16 => widen::<i16>(reference, r, label),
            IntType::U16 => widen::<u16>(reference, r, label),
            IntType::I32 => widen::<i32>(reference, r, label),
            IntType::U32 => widen::<u32>(reference, r, label),
            IntType::I64 => read_ref::<i64>(reference, r, label),
            IntType::U64 => read_ref::<u64>(reference, r, label)
                .into_iter()
                .map(|v| v as i64)
                .collect(),
        };

        let got = mine
            .get_values::<i64, _>(r)
            .unwrap_or_else(|e| panic!("{label}: native read failed: {e}"));

        assert_eq!(
            got.len(),
            expected.len(),
            "{label}: element count differs from netcdf-c"
        );
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, b, "{label}: element {i} of {ranges:?} differs");
        }
        tally.elements += got.len() as u64;
    }
}

/// Compare one floating-point variable by bit pattern, slab by slab.
fn compare_floats(
    label: &str,
    reference: &netcdf::Variable<'_>,
    mine: &Variable<'_>,
    float_type: FloatType,
    tally: &mut Tally,
) {
    for ranges in slabs(&mine.shape) {
        let r = ranges.as_slice();
        let expected: Vec<f64> = match float_type {
            FloatType::F32 => read_ref::<f32>(reference, r, label)
                .into_iter()
                .map(f64::from)
                .collect(),
            FloatType::F64 => read_ref::<f64>(reference, r, label),
        };

        let got = mine
            .get_values::<f64, _>(r)
            .unwrap_or_else(|e| panic!("{label}: native read failed: {e}"));

        assert_eq!(
            got.len(),
            expected.len(),
            "{label}: element count differs from netcdf-c"
        );
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}: element {i} of {ranges:?} differs ({a} vs {b})"
            );
        }
        tally.elements += got.len() as u64;
    }
}

fn read_ref<T: netcdf::NcTypeDescriptor + Copy>(
    variable: &netcdf::Variable<'_>,
    ranges: &[std::ops::Range<usize>],
    label: &str,
) -> Vec<T> {
    if ranges.is_empty() {
        return variable
            .get_values::<T, _>(netcdf::Extents::All)
            .unwrap_or_else(|e| panic!("{label}: netcdf-c read failed: {e}"));
    }
    variable
        .get_values::<T, _>(ranges)
        .unwrap_or_else(|e| panic!("{label}: netcdf-c read failed: {e}"))
}

fn widen<T>(
    variable: &netcdf::Variable<'_>,
    ranges: &[std::ops::Range<usize>],
    label: &str,
) -> Vec<i64>
where
    T: netcdf::NcTypeDescriptor + Copy + Into<i64>,
{
    read_ref::<T>(variable, ranges, label)
        .into_iter()
        .map(Into::into)
        .collect()
}

/// Every variable in the tree must decode identically to netcdf-c, at every
/// depth, whatever its type.
#[test]
fn every_variable_in_every_group_matches_netcdf_c() {
    let files = files();
    assert!(!files.is_empty(), "no file to compare");

    for (name, path) in files {
        let reference = netcdf::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let native = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));

        // The two readers must agree on which variables exist, before any
        // value can mean anything.
        let mut want = vec![];
        for_each_reference_variable(&reference.root().expect("a root group"), "", &mut |p, _| {
            want.push(p.to_string())
        });
        want.sort();
        let mut found: Vec<String> = native.variables().iter().map(|v| v.path.clone()).collect();
        found.sort();
        assert_eq!(found, want, "{name}: the variable trees differ");

        let mut tally = Tally::default();

        for_each_reference_variable(
            &reference.root().expect("a root group"),
            "",
            &mut |path, ref_var| {
                let label = format!("{name}:{path}");
                let mine = native
                    .variable(path)
                    .unwrap_or_else(|| panic!("{label}: this reader lost the variable"));

                let ref_shape: Vec<u64> = ref_var
                    .dimensions()
                    .iter()
                    .map(|d| d.len() as u64)
                    .collect();
                assert_eq!(mine.shape, ref_shape, "{label}: shape differs");

                // The raw bytes must be there whatever the type, so this covers
                // the types netcdf-c will not hand back as numbers.
                let raw = mine
                    .get_raw_values(..)
                    .unwrap_or_else(|e| panic!("{label}: raw read failed: {e}"));
                tally.bytes += raw.len() as u64;

                if mine.shape.contains(&0) {
                    assert!(raw.is_empty(), "{label}: an empty variable read bytes");
                    tally.empty += 1;
                    return;
                }

                match ref_var.vartype() {
                    NcVariableType::Int(t) => {
                        compare_integers(&label, ref_var, &mine, t, &mut tally);
                        tally.integer += 1;
                    }
                    NcVariableType::Float(t) => {
                        compare_floats(&label, ref_var, &mine, t, &mut tally);
                        tally.float += 1;
                    }
                    // netcdf-c refuses to hand back text as a numeric type, so
                    // these go through the string reader instead.
                    NcVariableType::Char | NcVariableType::String => {
                        let got = mine
                            .get_strings(..)
                            .unwrap_or_else(|e| panic!("{label}: string read failed: {e}"));
                        let want = ref_var
                            .get_strings(netcdf::Extents::All)
                            .unwrap_or_else(|e| {
                                panic!("{label}: netcdf-c string read failed: {e}")
                            });

                        let count: u64 = mine.shape.iter().product::<u64>().max(1);
                        assert_eq!(got.len() as u64, count, "{label}: one string per element");
                        assert_eq!(got.len(), want.len(), "{label}: string count differs");
                        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                            assert_eq!(a, b, "{label}: string {i} differs from netcdf-c");
                        }
                        tally.elements += got.len() as u64;
                        tally.text += 1;
                    }
                    // A compound, enum or opaque type. netcdf-c gives no scalar
                    // type for one, so the byte count above is the check.
                    _ => {
                        let width = ref_var.vartype().size() as u64;
                        let expected: u64 = mine.shape.iter().product::<u64>().max(1);
                        assert_eq!(
                            raw.len() as u64,
                            expected * width,
                            "{label}: raw byte count differs from netcdf-c's element size"
                        );
                        tally.opaque += 1;
                    }
                }
            },
        );

        eprintln!(
            "{name}: {} variables compared at every depth \
             ({} int, {} float, {} text, {} compound, {} empty); \
             {} elements, {} MiB read",
            tally.variables(),
            tally.integer,
            tally.float,
            tally.text,
            tally.opaque,
            tally.empty,
            tally.elements,
            tally.bytes / (1 << 20),
        );
        assert_eq!(
            tally.variables(),
            want.len(),
            "{name}: some variable was not accounted for"
        );
        assert!(tally.elements > 0, "{name}: nothing was compared");
    }
}

/// A selection that starts partway into a chunk must match too. A whole read
/// would not catch an offset error in the chunk-to-output copy.
#[test]
fn interior_selections_match_netcdf_c() {
    for (name, path) in files() {
        let reference = netcdf::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let native = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut checked = 0;

        for_each_reference_variable(
            &reference.root().expect("a root group"),
            "",
            &mut |path, ref_var| {
                let label = format!("{name}:{path}");
                let Some(mine) = native.variable(path) else {
                    return;
                };
                if mine.shape.is_empty() || mine.shape.iter().any(|&d| d < 3) {
                    return; // too small for an interior slab
                }

                // Start one element in on every axis, so the selection begins
                // partway through the first chunk and ends before its last.
                let ranges: Vec<std::ops::Range<usize>> = mine
                    .shape
                    .iter()
                    .map(|&d| 1usize..(d as usize).clamp(2, 4))
                    .collect();
                let r = ranges.as_slice();

                // Floats are compared by bit pattern, carried as integers so
                // both arms share one comparison.
                let (expected, got): (Vec<i64>, Vec<i64>) = match ref_var.vartype() {
                    NcVariableType::Int(IntType::I16) => (
                        widen::<i16>(ref_var, r, &label),
                        mine.get_values::<i64, _>(r).unwrap(),
                    ),
                    NcVariableType::Int(IntType::I32) => (
                        widen::<i32>(ref_var, r, &label),
                        mine.get_values::<i64, _>(r).unwrap(),
                    ),
                    NcVariableType::Int(IntType::I64) => (
                        read_ref::<i64>(ref_var, r, &label),
                        mine.get_values::<i64, _>(r).unwrap(),
                    ),
                    NcVariableType::Float(FloatType::F64) => (
                        read_ref::<f64>(ref_var, r, &label)
                            .into_iter()
                            .map(|v| v.to_bits() as i64)
                            .collect(),
                        mine.get_values::<f64, _>(r)
                            .unwrap()
                            .into_iter()
                            .map(|v| v.to_bits() as i64)
                            .collect(),
                    ),
                    _ => return,
                };

                assert_eq!(got.len(), expected.len(), "{label}: slab length differs");
                assert_eq!(got, expected, "{label}: slab values differ from netcdf-c");
                checked += 1;
            },
        );

        eprintln!("{name}: {checked} interior selections match netcdf-c");
    }
}

/// Many selection shapes over the same variable, all against netcdf-c.
///
/// `interior_selections_match_netcdf_c` takes one slab per variable. This takes
/// a spread of them over the largest variables: a single element, a partial
/// row, a selection that spans rows, and the far corner. An offset error in the
/// copy out of storage shows up in one of these even when a whole read is
/// clean, because a whole read never computes an offset.
#[test]
fn many_selection_shapes_match_netcdf_c() {
    for (name, path) in files() {
        let reference = netcdf::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let native = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut checked = 0;

        for_each_reference_variable(
            &reference.root().expect("a root group"),
            "",
            &mut |path, ref_var| {
                let label = format!("{name}:{path}");
                let Some(mine) = native.variable(path) else {
                    return;
                };
                // Only the numeric types both readers hand back as numbers.
                if !matches!(
                    ref_var.vartype(),
                    NcVariableType::Int(_) | NcVariableType::Float(_)
                ) {
                    return;
                }
                let shape = mine.shape.clone();
                if shape.is_empty() || shape.contains(&0) {
                    return;
                }

                for ranges in selection_shapes(&shape) {
                    let r = ranges.as_slice();
                    let want = read_ref::<f64>(ref_var, r, &label);
                    let got = mine.get_values::<f64, _>(r).unwrap_or_else(|e| {
                        panic!("{label}: native read of {ranges:?} failed: {e}")
                    });

                    assert_eq!(
                        got.len(),
                        want.len(),
                        "{label}: {ranges:?} returned the wrong count"
                    );
                    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{label}: element {i} of {ranges:?} differs ({a} vs {b})"
                        );
                    }
                    checked += 1;
                }

                // The singular reads, against the same reference.
                let first: Vec<std::ops::Range<usize>> = shape.iter().map(|_| 0..1).collect();
                let one = mine.get_value::<f64, _>(first.as_slice()).unwrap();
                let want = read_ref::<f64>(ref_var, first.as_slice(), &label)[0];
                assert_eq!(one.to_bits(), want.to_bits(), "{label}: get_value differs");
                checked += 1;
            },
        );

        eprintln!("{name}: {checked} selections match netcdf-c");
        assert!(checked > 0, "{name}: no selection was checked");
    }
}

/// A spread of selections over one shape: the near corner, the far corner, a
/// partial row, and a selection that spans rows.
fn selection_shapes(shape: &[u64]) -> Vec<Vec<std::ops::Range<usize>>> {
    let last = |axis: usize| -> std::ops::Range<usize> {
        let d = shape[axis] as usize;
        d - 1..d
    };
    let mut out = vec![
        // The near corner.
        shape.iter().map(|_| 0..1).collect::<Vec<_>>(),
        // The far corner, which is the last byte of the storage.
        (0..shape.len()).map(last).collect(),
    ];

    // A partial run along the last axis, starting off a boundary.
    if *shape.last().unwrap() >= 4 {
        let mut r: Vec<std::ops::Range<usize>> = shape.iter().map(|_| 0..1).collect();
        let d = *shape.last().unwrap() as usize;
        *r.last_mut().unwrap() = 1..(d - 1).min(1 + 1000);
        out.push(r);
    }

    // A selection that spans rows, so the copy has to skip storage between
    // them rather than run straight through.
    if shape.len() >= 2 && shape[0] >= 3 && shape[1] >= 4 {
        let mut r: Vec<std::ops::Range<usize>> = shape.iter().map(|&d| 0..d as usize).collect();
        r[0] = 1..3;
        r[1] = 2..(shape[1] as usize - 1).min(2 + 500);
        out.push(r);
    }

    // The tail of the first axis, whole rows.
    if shape[0] >= 2 {
        let mut r: Vec<std::ops::Range<usize>> = shape.iter().map(|&d| 0..d as usize).collect();
        r[0] = shape[0] as usize - 2..shape[0] as usize;
        // Keep the read bounded on a wide variable.
        if shape.len() >= 2 {
            r[1] = 0..(shape[1] as usize).min(2000);
        }
        out.push(r);
    }

    out
}

/// Attributes must match netcdf-c at every depth, not just in the root group.
///
/// `attributes.rs` covers the netCDF corpus, which is one group deep. This
/// covers a group attribute and a variable attribute inside a nested group.
#[test]
fn attributes_match_netcdf_c_at_every_depth() {
    for (name, path) in files() {
        let reference = netcdf::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let native = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut compared = 0;
        let mut incomplete = 0;

        // The groups first, root included.
        for_each_reference_group(
            &reference.root().expect("a root group"),
            "/",
            &mut |path, ref_group| {
                let label = format!("{name}:{path}");
                let ours = native
                    .group(path)
                    .unwrap_or_else(|| panic!("{label}: this reader lost the group"));
                for ref_attr in ref_group.attributes() {
                    let attr_name = ref_attr.name().to_string();
                    let Some(mine) = ours.attribute(&attr_name) else {
                        panic!("{label}: group attribute {attr_name} is missing");
                    };
                    compare_attribute(&label, &attr_name, &ref_attr, &mine.value);
                    compared += 1;
                }
            },
        );

        for_each_reference_variable(
            &reference.root().expect("a root group"),
            "",
            &mut |path, ref_var| {
                let label = format!("{name}:{path}");
                let Some(mine) = native.variable(path) else {
                    return;
                };
                if !mine.attributes_complete {
                    incomplete += 1;
                }

                for ref_attr in ref_var.attributes() {
                    let attr_name = ref_attr.name().to_string();
                    let Some(ours) = mine.attribute(&attr_name) else {
                        panic!("{label}: attribute {attr_name} is missing");
                    };
                    compare_attribute(&label, &attr_name, &ref_attr, &ours.value);
                    compared += 1;
                }
            },
        );

        eprintln!(
            "{name}: {compared} attribute values match netcdf-c at every depth \
             ({incomplete} objects report an incomplete attribute list)"
        );
        // A file may genuinely carry none. Saying so beats passing silently.
        if compared == 0 {
            eprintln!("{name}: this file holds no attribute, so nothing was proved here");
        }
    }
}

/// Walk the reference group tree, handing each group and its path to `f`.
fn for_each_reference_group(
    group: &netcdf::Group<'_>,
    path: &str,
    f: &mut impl FnMut(&str, &netcdf::Group<'_>),
) {
    f(path, group);
    for child in group.groups() {
        let name = child.name();
        let child_path = if path == "/" {
            format!("/{name}")
        } else {
            format!("{path}/{name}")
        };
        for_each_reference_group(&child, &child_path, f);
    }
}

/// One attribute value, compared as text where it is text and as numbers
/// otherwise, so the stored type is not lost on the way.
fn compare_attribute(
    label: &str,
    attr_name: &str,
    reference: &netcdf::Attribute<'_>,
    ours: &oxcdf::AttributeValue,
) {
    match reference.value() {
        Ok(netcdf::AttributeValue::Str(want)) => {
            assert_eq!(
                ours.as_text(),
                Some(want.as_str()),
                "{label}: attribute {attr_name} differs"
            );
        }
        Ok(other) => {
            let Some(want) = attribute_numbers(&other) else {
                return;
            };
            let got = ours
                .as_f64s()
                .unwrap_or_else(|| panic!("{label}: attribute {attr_name} is not numeric here"));
            assert_eq!(
                got.len(),
                want.len(),
                "{label}: attribute {attr_name} length differs"
            );
            for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{label}: attribute {attr_name}[{i}] differs"
                );
            }
        }
        Err(_) => {}
    }
}

/// A netcdf-c attribute value as `f64`s, or `None` when it is not numeric.
fn attribute_numbers(value: &netcdf::AttributeValue) -> Option<Vec<f64>> {
    use netcdf::AttributeValue as A;
    Some(match value {
        A::Uchar(v) => vec![*v as f64],
        A::Uchars(v) => v.iter().map(|&x| x as f64).collect(),
        A::Schar(v) => vec![*v as f64],
        A::Schars(v) => v.iter().map(|&x| x as f64).collect(),
        A::Ushort(v) => vec![*v as f64],
        A::Ushorts(v) => v.iter().map(|&x| x as f64).collect(),
        A::Short(v) => vec![*v as f64],
        A::Shorts(v) => v.iter().map(|&x| x as f64).collect(),
        A::Uint(v) => vec![*v as f64],
        A::Uints(v) => v.iter().map(|&x| x as f64).collect(),
        A::Int(v) => vec![*v as f64],
        A::Ints(v) => v.iter().map(|&x| x as f64).collect(),
        A::Ulonglong(v) => vec![*v as f64],
        A::Ulonglongs(v) => v.iter().map(|&x| x as f64).collect(),
        A::Longlong(v) => vec![*v as f64],
        A::Longlongs(v) => v.iter().map(|&x| x as f64).collect(),
        A::Float(v) => vec![*v as f64],
        A::Floats(v) => v.iter().map(|&x| x as f64).collect(),
        A::Double(v) => vec![*v],
        A::Doubles(v) => v.clone(),
        _ => return None,
    })
}

/// The asynchronous engine must return the same bytes as the synchronous one,
/// for every variable at every depth.
#[cfg(feature = "async")]
#[test]
fn the_asynchronous_engine_reads_the_same_values() {
    use oxcdf::{AsyncNetcdfFile, FileSource, SyncAsAsync};

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    for (name, path) in files() {
        let sync = NetcdfFile::open(&path).unwrap_or_else(|e| panic!("{name}: {e}"));

        runtime.block_on(async {
            let source = std::sync::Arc::new(SyncAsAsync(FileSource::open(&path).unwrap()));
            let file = AsyncNetcdfFile::open(source).await.unwrap();

            let mut checked = 0;
            for mine in sync.variables() {
                let path = &mine.info().path;
                let label = format!("{name}:{path}");
                let theirs = file
                    .variable(path)
                    .unwrap_or_else(|| panic!("{label}: the async engine lost the variable"));

                assert_eq!(theirs.info().shape, mine.info().shape, "{label}: shape");
                assert_eq!(
                    theirs.info().dimensions,
                    mine.info().dimensions,
                    "{label}: axes"
                );

                // Compare a bounded corner, so a large variable does not turn
                // this into a second full read of the file.
                let corner: Vec<std::ops::Range<usize>> = mine
                    .info()
                    .shape
                    .iter()
                    .map(|&d| 0..(d as usize).min(8))
                    .collect();
                let want = mine.get_raw_values(corner.as_slice()).unwrap();
                let got = theirs.get_raw_values(corner.as_slice()).await.unwrap();
                assert_eq!(got, want, "{label}: the two engines differ");
                checked += 1;
            }
            eprintln!("{name}: {checked} variables agree across both engines");
        });
    }
}
