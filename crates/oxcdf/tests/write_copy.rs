#![cfg(feature = "write")]
#![expect(
    clippy::single_range_in_vec_init,
    reason = "a selection on one axis is an array with one range"
)]
//! The netCDF writer copies a file through the public interface.
//!
//! The copy must read back equal to the source.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p oxcdf --features write --test write_copy
//! ```

use std::fmt::Debug;
use std::ops::Range;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use oxcdf::hdf5::message::filter::id as filter_id;
use oxcdf::netcdf::NcGroup;
use oxcdf::{
    AttributeValue, DType, Element, Error, FileMut, MemorySource, NcAttribute, NetcdfFile,
    Variable, VariableMut,
};

/// Filter ids that the writer can make.
const WRITABLE_FILTERS: [u16; 4] = [
    filter_id::DEFLATE,
    filter_id::SHUFFLE,
    filter_id::FLETCHER32,
    filter_id::ZSTD,
];

/// Read the bytes of a file in memory.
fn reopen(bytes: Vec<u8>) -> NetcdfFile {
    NetcdfFile::from_source(Arc::new(MemorySource::new(bytes))).unwrap()
}

/// Read `T` from `source` and write it to all elements of `target`.
fn copy_typed<T: Element>(
    source: &Variable<'_>,
    target: &mut VariableMut<'_>,
    ranges: Vec<Range<usize>>,
) -> oxcdf::Result<()> {
    let values = source.get_values::<T, _>(..)?;
    if ranges.is_empty() {
        target.put_values(&values, ..)
    } else {
        target.put_values(&values, ranges)
    }
}

/// Set the fill value of `target` with the type `T` of the variable.
fn copy_fill<T: Element>(source: &Variable<'_>, target: &mut VariableMut<'_>) -> oxcdf::Result<()> {
    match source.fill_value::<T>()? {
        Some(fill) => target.set_fill_value(fill),
        None => Ok(()),
    }
}

/// Set the fill value of a `char` variable from its attribute.
fn copy_char_fill(source: &Variable<'_>, target: &mut VariableMut<'_>) -> oxcdf::Result<()> {
    let Some(attribute) = source.attribute("_FillValue") else {
        return Ok(());
    };
    let byte = match &attribute.value {
        AttributeValue::Uchar(b) => Some(*b),
        AttributeValue::Uchars(b) => b.first().copied(),
        AttributeValue::Schar(b) => Some(*b as u8),
        AttributeValue::Str(s) => s.bytes().next(),
        _ => None,
    };
    match byte {
        Some(byte) => target.set_fill_value(byte),
        None => Err(Error::unsupported(format!(
            "the _FillValue {:?} of char variable {}",
            attribute.value, source.name
        ))),
    }
}

/// Set the fill value of `target` with the type of the variable.
fn copy_any_fill(source: &Variable<'_>, target: &mut VariableMut<'_>) -> oxcdf::Result<()> {
    match source.vartype() {
        DType::Int(1) => copy_fill::<i8>(source, target),
        DType::Int(2) => copy_fill::<i16>(source, target),
        DType::Int(4) => copy_fill::<i32>(source, target),
        DType::Int(8) => copy_fill::<i64>(source, target),
        DType::Uint(1) => copy_fill::<u8>(source, target),
        DType::Uint(2) => copy_fill::<u16>(source, target),
        DType::Uint(4) => copy_fill::<u32>(source, target),
        DType::Uint(8) => copy_fill::<u64>(source, target),
        DType::Float(4) => copy_fill::<f32>(source, target),
        DType::Float(8) => copy_fill::<f64>(source, target),
        DType::Char => copy_char_fill(source, target),
        other => Err(Error::unsupported(format!(
            "a _FillValue on variable {} of type {other:?}",
            source.name
        ))),
    }
}

/// Copy the definition, the attributes and the values of one variable.
fn copy_variable(source: &Variable<'_>, target: &mut VariableMut<'_>) -> oxcdf::Result<()> {
    let dtype = source.vartype();
    let dataset = source
        .dataset()
        .ok_or_else(|| Error::unsupported("a variable of a classic file"))?;
    if let Some(order) = source.datatype().byte_order() {
        target.set_endianness(order)?;
    }
    if let Some(chunks) = source.chunking()? {
        target.set_chunking(&chunks)?;
    }
    let (mut deflate, mut zstd, mut shuffle, mut fletcher) = (None, None, false, false);
    for filter in &dataset.pipeline.filters {
        let level = filter.client_data.first().map_or(0, |&l| l as i32);
        match filter.id {
            filter_id::DEFLATE => deflate = Some(level),
            filter_id::ZSTD => zstd = Some(level),
            filter_id::SHUFFLE => shuffle = true,
            filter_id::FLETCHER32 => fletcher = true,
            other => {
                return Err(Error::unsupported(format!("the filter {other}")));
            }
        }
    }
    match (deflate, zstd, shuffle) {
        (Some(level), None, _) => target.set_compression(level, shuffle)?,
        (None, Some(level), _) => {
            target.set_shuffle(shuffle)?;
            target.set_zstd(level)?;
        }
        (None, None, true) => target.set_shuffle(true)?,
        (None, None, false) => {}
        (Some(_), Some(_), _) => return Err(Error::unsupported("deflate with Zstandard")),
    }
    if fletcher {
        target.set_fletcher32(true)?;
    }
    // `_FillValue` goes in its own place, so the attribute order stays.
    for attribute in source.attributes() {
        if attribute.name == "_FillValue" {
            copy_any_fill(source, target)?;
        } else {
            target.put_attribute(&attribute.name, attribute.value.clone())?;
        }
    }

    // A selection that names each element lets an unlimited axis grow.
    let ranges: Vec<Range<usize>> = source.shape.iter().map(|&n| 0..n as usize).collect();
    if source.is_empty() {
        return Ok(());
    }
    match dtype {
        DType::Int(1) => copy_typed::<i8>(source, target, ranges),
        DType::Int(2) => copy_typed::<i16>(source, target, ranges),
        DType::Int(4) => copy_typed::<i32>(source, target, ranges),
        DType::Int(8) => copy_typed::<i64>(source, target, ranges),
        DType::Uint(1) => copy_typed::<u8>(source, target, ranges),
        DType::Uint(2) => copy_typed::<u16>(source, target, ranges),
        DType::Uint(4) => copy_typed::<u32>(source, target, ranges),
        DType::Uint(8) => copy_typed::<u64>(source, target, ranges),
        DType::Float(4) => copy_typed::<f32>(source, target, ranges),
        DType::Float(8) => copy_typed::<f64>(source, target, ranges),
        DType::Char => {
            let bytes = source.get_raw_values(..)?;
            if ranges.is_empty() {
                target.put_values(&bytes, ..)
            } else {
                target.put_values(&bytes, ranges)
            }
        }
        DType::String => {
            let strings = source.get_strings(..)?;
            if ranges.is_empty() {
                target.put_strings(&strings, ..)
            } else {
                target.put_strings(&strings, ranges)
            }
        }
        other => Err(Error::unsupported(format!(
            "variable {} of type {other:?}",
            source.name
        ))),
    }
}

/// The value of an attribute that the writer takes.
fn copy_attribute(attribute: &NcAttribute) -> oxcdf::Result<AttributeValue> {
    match &attribute.value {
        AttributeValue::Raw(_) => Err(Error::unsupported(format!(
            "the raw attribute {}",
            attribute.name
        ))),
        value => Ok(value.clone()),
    }
}

/// Copy the dimensions, the attributes and the variables of one group.
///
/// `$target` is a `FileMut` or a `GroupMut`, which have the same methods.
macro_rules! copy_members {
    ($source:expr, $group:expr, $target:expr) => {{
        let mut dimensions: Vec<_> = $group.dimensions.iter().collect();
        dimensions.sort_by_key(|d| d.id);
        for dimension in dimensions {
            if dimension.is_phony {
                return Err(Error::unsupported(format!(
                    "the phony dimension {}",
                    dimension.name
                )));
            }
            if dimension.is_unlimited {
                $target.add_unlimited_dimension(&dimension.name)?;
            } else if dimension.len == 0 {
                return Err(Error::unsupported(format!(
                    "the fixed dimension {} of length 0",
                    dimension.name
                )));
            } else {
                $target.add_dimension(&dimension.name, dimension.len as usize)?;
            }
        }
        for attribute in &$group.attributes {
            $target.add_attribute(&attribute.name, copy_attribute(attribute)?)?;
        }
        for info in &$group.variables {
            let source = $source
                .variable(&info.path)
                .ok_or_else(|| Error::not_found(format!("variable {}", info.path)))?;
            let names: Vec<&str> = info.dimensions.iter().map(String::as_str).collect();
            let mut variable =
                $target.add_variable_with_type(&info.name, &names, &source.vartype())?;
            copy_variable(&source, &mut variable)?;
        }
    }};
}

/// Copy a group and the groups below it to `target`.
fn copy_group(
    source: &NetcdfFile,
    group: &NcGroup,
    target: &mut oxcdf::GroupMut<'_>,
) -> oxcdf::Result<()> {
    copy_members!(source, group, target);
    for child in &group.groups {
        let mut child_target = target.add_group(&child.name)?;
        copy_group(source, child, &mut child_target)?;
    }
    Ok(())
}

/// Copy everything of `source` to `target` through the public interface.
///
/// Returns [`Error::Unsupported`] for a part that the writer cannot make.
pub fn copy_file(source: &NetcdfFile, target: &mut FileMut) -> oxcdf::Result<()> {
    let root = source.root();
    copy_members!(source, root, target);
    for child in &root.groups {
        let mut child_target = target.add_group(&child.name)?;
        copy_group(source, child, &mut child_target)?;
    }
    Ok(())
}

/// Panic with a message that names the file and the place of the difference.
fn differ(what: &str, place: &str, detail: &str) -> ! {
    panic!("{what}: {place}: {detail}")
}

/// Compare two values and panic at the first difference.
fn same<T: PartialEq + Debug>(what: &str, place: &str, a: &T, b: &T) {
    if a != b {
        differ(what, place, &format!("source {a:?}, copy {b:?}"));
    }
}

/// Compare the attributes of two objects.
fn same_attributes(what: &str, place: &str, a: &[NcAttribute], b: &[NcAttribute]) {
    let names = |list: &[NcAttribute]| list.iter().map(|x| x.name.clone()).collect::<Vec<_>>();
    same(
        what,
        &format!("{place} attribute names"),
        &names(a),
        &names(b),
    );
    for (x, y) in a.iter().zip(b) {
        same(
            what,
            &format!("{place} attribute {}", x.name),
            &format!("{:?}", x.value),
            &format!("{:?}", y.value),
        );
    }
}

/// Compare the values of one variable as `T`. `key` maps a value to a comparable form.
fn same_typed<T: Element, K: PartialEq + Debug>(
    what: &str,
    path: &str,
    a: &Variable<'_>,
    b: &Variable<'_>,
    key: fn(&T) -> K,
) {
    let x = a.get_values::<T, _>(..).unwrap();
    let y = b.get_values::<T, _>(..).unwrap();
    same(
        what,
        &format!("variable {path} value count"),
        &x.len(),
        &y.len(),
    );
    for (index, (p, q)) in x.iter().zip(&y).enumerate() {
        if key(p) != key(q) {
            differ(
                what,
                &format!("variable {path}"),
                &format!("index {index}: source {:?}, copy {:?}", key(p), key(q)),
            );
        }
    }
}

/// Compare the values of two variables of the same type.
fn same_values(what: &str, path: &str, a: &Variable<'_>, b: &Variable<'_>) {
    match a.vartype() {
        DType::Int(1) => same_typed::<i8, _>(what, path, a, b, |v| *v),
        DType::Int(2) => same_typed::<i16, _>(what, path, a, b, |v| *v),
        DType::Int(4) => same_typed::<i32, _>(what, path, a, b, |v| *v),
        DType::Int(8) => same_typed::<i64, _>(what, path, a, b, |v| *v),
        DType::Uint(1) => same_typed::<u8, _>(what, path, a, b, |v| *v),
        DType::Uint(2) => same_typed::<u16, _>(what, path, a, b, |v| *v),
        DType::Uint(4) => same_typed::<u32, _>(what, path, a, b, |v| *v),
        DType::Uint(8) => same_typed::<u64, _>(what, path, a, b, |v| *v),
        DType::Float(4) => same_typed::<f32, _>(what, path, a, b, |v| v.to_bits()),
        DType::Float(8) => same_typed::<f64, _>(what, path, a, b, |v| v.to_bits()),
        DType::Char => {
            let x = a.get_raw_values(..).unwrap();
            let y = b.get_raw_values(..).unwrap();
            same(
                what,
                &format!("variable {path} byte count"),
                &x.len(),
                &y.len(),
            );
            if let Some(index) = x.iter().zip(&y).position(|(p, q)| p != q) {
                let detail = format!("index {index}: source {}, copy {}", x[index], y[index]);
                differ(what, &format!("variable {path}"), &detail);
            }
        }
        DType::String => {
            let x = a.get_strings(..).unwrap();
            let y = b.get_strings(..).unwrap();
            same(
                what,
                &format!("variable {path} string count"),
                &x.len(),
                &y.len(),
            );
            if let Some(index) = x.iter().zip(&y).position(|(p, q)| p != q) {
                let detail = format!("index {index}: source {:?}, copy {:?}", x[index], y[index]);
                differ(what, &format!("variable {path}"), &detail);
            }
        }
        other => differ(what, path, &format!("the type {other:?} has no comparison")),
    }
}

/// Compare one group and the groups below it.
fn same_group(a: &NetcdfFile, b: &NetcdfFile, x: &NcGroup, y: &NcGroup, what: &str) {
    let place = format!("group {}", x.path);
    let dims = |g: &NcGroup| {
        g.dimensions
            .iter()
            .map(|d| (d.name.clone(), d.len, d.is_unlimited))
            .collect::<Vec<_>>()
    };
    same(what, &format!("{place} dimensions"), &dims(x), &dims(y));
    same_attributes(what, &place, &x.attributes, &y.attributes);
    let names = |g: &NcGroup| {
        g.variables
            .iter()
            .map(|v| v.name.clone())
            .collect::<Vec<_>>()
    };
    same(
        what,
        &format!("{place} variable names"),
        &names(x),
        &names(y),
    );
    for (vx, vy) in x.variables.iter().zip(&y.variables) {
        let path = vx.path.as_str();
        let va = a.variable(&vx.path).expect("variable of the source");
        let vb = b.variable(&vy.path).expect("variable of the copy");
        same(
            what,
            &format!("variable {path} type"),
            &va.vartype(),
            &vb.vartype(),
        );
        same(
            what,
            &format!("variable {path} dimensions"),
            &vx.dimensions,
            &vy.dimensions,
        );
        same(
            what,
            &format!("variable {path} shape"),
            &vx.shape,
            &vy.shape,
        );
        let order = |v: &Variable<'_>| format!("{:?}", v.datatype().byte_order());
        same(
            what,
            &format!("variable {path} byte order"),
            &order(&va),
            &order(&vb),
        );
        let chunking = (va.chunking().unwrap(), vb.chunking().unwrap());
        same(
            what,
            &format!("variable {path} chunking"),
            &chunking.0,
            &chunking.1,
        );
        let ids = |v: &Variable<'_>| {
            v.dataset()
                .map(|d| d.pipeline.filters.iter().map(|f| f.id).collect::<Vec<_>>())
        };
        same(
            what,
            &format!("variable {path} filters"),
            &ids(&va),
            &ids(&vb),
        );
        same_attributes(
            what,
            &format!("variable {path}"),
            &vx.attributes,
            &vy.attributes,
        );
        same_values(what, path, &va, &vb);
    }
    let groups = |g: &NcGroup| g.groups.iter().map(|c| c.name.clone()).collect::<Vec<_>>();
    same(
        what,
        &format!("{place} child groups"),
        &groups(x),
        &groups(y),
    );
    for (cx, cy) in x.groups.iter().zip(&y.groups) {
        same_group(a, b, cx, cy, what);
    }
}

/// Compare two files through the reader. Panic at the first difference.
pub fn assert_same(a: &NetcdfFile, b: &NetcdfFile, what: &str) {
    same_group(a, b, a.root(), b.root(), what);
}

/// Copy `source` to a file in memory and read it back.
fn copy_in_memory(source: &NetcdfFile) -> oxcdf::Result<NetcdfFile> {
    let mut target = oxcdf::create_in_memory()?;
    copy_file(source, &mut target)?;
    let bytes = target.into_bytes()?;
    NetcdfFile::from_source(Arc::new(MemorySource::new(bytes)))
}

/// The reason why the writer cannot copy `file`, when a check says so.
fn skip_reason(file: &NetcdfFile) -> Option<String> {
    if file.hdf5().is_none() {
        return Some("a classic file".to_owned());
    }
    skip_reason_group(file, file.root())
}

/// The reason why the writer cannot copy `group`, when a check says so.
fn skip_reason_group(file: &NetcdfFile, group: &NcGroup) -> Option<String> {
    if let Some(d) = group.dimensions.iter().find(|d| d.is_phony) {
        return Some(format!("the phony dimension {}", d.name));
    }
    if let Some(a) = group
        .attributes
        .iter()
        .find(|a| matches!(a.value, AttributeValue::Raw(_)))
    {
        return Some(format!("the raw attribute {}", a.name));
    }
    for info in &group.variables {
        let Some(variable) = file.variable(&info.path) else {
            return Some(format!("no dataset for {}", info.path));
        };
        if let Some(a) = info
            .attributes
            .iter()
            .find(|a| matches!(a.value, AttributeValue::Raw(_)))
        {
            return Some(format!("the raw attribute {} of {}", a.name, info.path));
        }
        match variable.vartype() {
            DType::Other | DType::Vlen(_) | DType::FixedString(_) => {
                return Some(format!(
                    "the type {:?} of {}",
                    variable.vartype(),
                    info.path
                ));
            }
            _ => {}
        }
        let filters = variable.dataset().map(|d| d.pipeline.filters.as_slice());
        if let Some(f) = filters
            .unwrap_or_default()
            .iter()
            .find(|f| !WRITABLE_FILTERS.contains(&f.id))
        {
            return Some(format!("the filter {} of {}", f.id, info.path));
        }
    }
    group
        .groups
        .iter()
        .find_map(|child| skip_reason_group(file, child))
}

/// The `.nc` files of a directory, sorted by name.
fn nc_files(directory: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(directory)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", directory.display()))
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "nc"))
        .collect();
    files.sort();
    files
}

/// Copy each file in `DIR/cases` to `DIR/copies`, and compare the copy.
#[test]
#[ignore = "run by scripts/check_copy.py"]
fn python_files_copy_through_the_writer() {
    let directory = PathBuf::from(
        std::env::var_os("OXCDF_COPY_DIR").expect("set OXCDF_COPY_DIR to the directory of cases"),
    );
    let copies = directory.join("copies");
    std::fs::create_dir_all(&copies).unwrap();
    let mut failures = Vec::new();
    for path in nc_files(&directory.join("cases")) {
        let name = path.file_name().unwrap().to_owned();
        let label = name.to_string_lossy().into_owned();
        let target_path = copies.join(&name);
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
            let source = oxcdf::open(&path).map_err(|e| format!("open failed: {e}"))?;
            let mut target =
                oxcdf::create(&target_path).map_err(|e| format!("create failed: {e}"))?;
            copy_file(&source, &mut target).map_err(|e| format!("copy failed: {e}"))?;
            target.close().map_err(|e| format!("close failed: {e}"))?;
            let copy = oxcdf::open(&target_path).map_err(|e| format!("reopen failed: {e}"))?;
            assert_same(&source, &copy, &label);
            Ok(())
        }));
        match result {
            Ok(Ok(())) => println!("ok   {label}"),
            Ok(Err(message)) => {
                println!("FAIL {label}: {message}");
                failures.push(format!("{label}: {message}"));
            }
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_else(|| "a panic".to_owned());
                println!("FAIL {label}: {message}");
                failures.push(message);
            }
        }
    }
    assert!(
        failures.is_empty(),
        "copy failures:\n{}",
        failures.join("\n")
    );
}

/// Copy each file of the corpus to memory, and compare the copy.
#[test]
fn corpus_files_copy_through_the_writer() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_files");
    if !directory.is_dir() {
        return;
    }
    let mut copied = 0;
    let mut failures = Vec::new();
    for path in nc_files(&directory) {
        let label = path.file_name().unwrap().to_string_lossy().into_owned();
        let source = match oxcdf::open(&path) {
            Ok(source) => source,
            Err(e) => {
                println!("skip {label}: the reader cannot open it: {e}");
                continue;
            }
        };
        if let Some(reason) = skip_reason(&source) {
            println!("skip {label}: {reason}");
            continue;
        }
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| match copy_in_memory(&source) {
            Ok(copy) => {
                assert_same(&source, &copy, &label);
                Ok(true)
            }
            Err(Error::Unsupported(reason)) => {
                println!("skip {label}: the writer says: {reason}");
                Ok(false)
            }
            Err(e) => Err(format!("{label}: copy failed: {e}")),
        }));
        match result {
            Ok(Ok(true)) => {
                println!("ok   {label}");
                copied += 1;
            }
            Ok(Ok(false)) => {}
            Ok(Err(message)) => failures.push(message),
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_else(|| "a panic".to_owned());
                failures.push(message);
            }
        }
    }
    assert!(
        failures.is_empty(),
        "copy failures:\n{}",
        failures.join("\n")
    );
    assert!(copied >= 4, "only {copied} corpus files copied");
}

/// A file from the writer copies to an equal file.
#[test]
fn a_file_from_the_writer_copies_to_an_equal_file() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_unlimited_dimension("time").unwrap();
    file.add_dimension("lat", 3).unwrap();
    file.add_dimension("len", 4).unwrap();
    file.add_attribute("title", "copy test").unwrap();
    file.add_attribute("levels", vec![1i32, 2, 3]).unwrap();
    file.add_attribute("scale", 0.5f64).unwrap();
    let mut lat = file.add_variable::<f32>("lat", &["lat"]).unwrap();
    lat.put_values(&[-10.0f32, 0.0, 10.0], ..).unwrap();
    let mut temp = file.add_variable::<f32>("TEMP", &["time", "lat"]).unwrap();
    temp.set_chunking(&[2, 3]).unwrap();
    temp.set_compression(4, true).unwrap();
    temp.set_fill_value(-999.0f32).unwrap();
    temp.put_attribute("units", "degC").unwrap();
    temp.put_values(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [0..2, 0..3])
        .unwrap();
    let mut count = file.add_variable::<u16>("count", &["time"]).unwrap();
    count.set_fletcher32(true).unwrap();
    count
        .set_endianness(oxcdf::hdf5::message::ByteOrder::Big)
        .unwrap();
    count.put_values(&[7u16, 8], [0..2]).unwrap();
    let mut packed = file.add_variable::<i64>("packed", &["lat"]).unwrap();
    packed.set_zstd(3).unwrap();
    packed.put_values(&[1i64, 2, 3], ..).unwrap();
    let mut name = file.add_string_variable("name", &["lat"]).unwrap();
    name.put_strings(&["south", "equator", "north"], ..)
        .unwrap();
    let mut code = file
        .add_variable_with_type("code", &["lat", "len"], &DType::Char)
        .unwrap();
    code.put_values(b"abcdefghijkl".as_slice(), ..).unwrap();
    let mut scalar = file.add_variable::<f64>("scalar", &[]).unwrap();
    scalar.put_values(&[2.5f64], ..).unwrap();
    let mut sub = file.add_group("sub").unwrap();
    sub.add_dimension("z", 2).unwrap();
    sub.add_attribute("note", "in a group").unwrap();
    let mut deep = sub.add_variable::<i32>("deep", &["z", "lat"]).unwrap();
    deep.put_values(&[1, 2, 3, 4, 5, 6], ..).unwrap();
    let mut inner = sub.add_group("inner").unwrap();
    let mut bytes = inner.add_variable::<i8>("bytes", &["z"]).unwrap();
    bytes.put_values(&[-1i8, 1], ..).unwrap();

    let source = reopen(file.into_bytes().unwrap());
    let copy = copy_in_memory(&source).unwrap();
    assert_same(&source, &copy, "a file from the writer");
}
