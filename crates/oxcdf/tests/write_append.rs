#![cfg(feature = "write")]
#![expect(
    clippy::single_range_in_vec_init,
    reason = "a selection on one axis is an array with one range"
)]
//! The netCDF writer opens a file that exists and changes it.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p oxcdf --features write --test write_append
//! ```

use std::sync::Arc;

use oxcdf::{AttributeValue, Error, FileMut, MemorySource, NetcdfFile};

fn read(bytes: Vec<u8>) -> NetcdfFile {
    NetcdfFile::from_source(Arc::new(MemorySource::new(bytes))).unwrap()
}

fn dimensions(file: &NetcdfFile) -> Vec<(String, u64, bool)> {
    file.dimensions()
        .iter()
        .map(|d| (d.name.clone(), d.len, d.is_unlimited))
        .collect()
}

/// A file with an unlimited dimension, a coordinate variable, filters, strings and a group.
fn first_file() -> FileMut {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_unlimited_dimension("time").unwrap();
    file.add_dimension("lat", 3).unwrap();
    file.add_attribute("title", "first").unwrap();
    let mut lat = file.add_variable::<f32>("lat", &["lat"]).unwrap();
    lat.put_values(&[-10.0f32, 0.0, 10.0], ..).unwrap();
    let mut temp = file.add_variable::<f32>("TEMP", &["time", "lat"]).unwrap();
    temp.set_chunking(&[2, 3]).unwrap();
    temp.set_compression(4, true).unwrap();
    temp.set_fill_value(-999.0f32).unwrap();
    temp.put_attribute("units", "degC").unwrap();
    temp.put_values(&[1.0f32, 2.0, 3.0], [0..1, 0..3]).unwrap();
    let mut name = file.add_string_variable("name", &["lat"]).unwrap();
    name.put_strings(&["south", "equator"], [0..2]).unwrap();
    let mut sub = file.add_group("sub").unwrap();
    sub.add_dimension("z", 2).unwrap();
    let mut deep = sub.add_variable::<i32>("deep", &["z", "lat"]).unwrap();
    deep.put_values(&[1, 2, 3, 4, 5, 6], ..).unwrap();
    file
}

#[test]
fn an_append_with_no_change_keeps_the_file() {
    let file = oxcdf::append_in_memory(first_file().into_bytes().unwrap()).unwrap();
    assert_eq!(file.dimension_len("time"), Some(1));
    let file = read(file.into_bytes().unwrap());
    assert_eq!(
        dimensions(&file),
        [("time".to_owned(), 1, true), ("lat".to_owned(), 3, false)]
    );
    let names: Vec<String> = file.variables().iter().map(|v| v.name.clone()).collect();
    // The list holds the variables of each group.
    assert_eq!(names, ["lat", "TEMP", "name", "deep"]);
    let temp = file.variable("TEMP").unwrap();
    assert_eq!(temp.dimensions, ["time", "lat"]);
    assert_eq!(temp.get_values::<f32, _>(..).unwrap(), [1.0, 2.0, 3.0]);
    assert_eq!(temp.attributes().len(), 2);
    assert!(file.variable("lat").unwrap().is_coordinate);
    let deep = file.variable("/sub/deep").unwrap();
    assert_eq!(deep.dimensions, ["z", "lat"]);
    assert_eq!(deep.get_values::<i32, _>(..).unwrap(), [1, 2, 3, 4, 5, 6]);
    assert_eq!(file.attributes().len(), 1);
}

#[test]
fn an_append_adds_records_along_the_unlimited_dimension() {
    let mut file = oxcdf::append_in_memory(first_file().into_bytes().unwrap()).unwrap();
    let mut temp = file.variable_mut("TEMP").unwrap();
    // Record 1 is in the compressed chunk of record 0. Record 2 is in a new chunk.
    temp.put_values(&[4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0], [1..3, 0..3])
        .unwrap();
    temp.put_values(&[20.0f32], [0..1, 1..2]).unwrap();
    assert_eq!(file.dimension_len("time"), Some(3));

    let file = read(file.into_bytes().unwrap());
    assert_eq!(file.dimension_len("time"), Some(3));
    let temp = file.variable("TEMP").unwrap();
    assert_eq!(
        temp.get_values::<f32, _>(..).unwrap(),
        [1.0, 20.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
    assert_eq!(temp.fill_value::<f32>().unwrap(), Some(-999.0));
}

#[test]
fn an_append_adds_objects_on_dimensions_of_the_file() {
    let mut file = oxcdf::append_in_memory(first_file().into_bytes().unwrap()).unwrap();
    file.add_attribute("title", "second").unwrap();
    file.add_attribute("history", "appended").unwrap();
    file.add_dimension("depth", 2).unwrap();
    let mut psal = file
        .add_variable::<f64>("PSAL", &["time", "lat", "depth"])
        .unwrap();
    psal.set_compression(1, false).unwrap();
    psal.put_values(&[1.5f64; 6], [0..1, 0..3, 0..2]).unwrap();
    let mut name = file.variable_mut("name").unwrap();
    name.put_string("north", [2]).unwrap();
    name.put_attribute("long_name", "station name").unwrap();
    let mut sub = file.group_mut("sub").unwrap();
    let mut more = sub.add_variable::<i16>("more", &["z"]).unwrap();
    more.put_values(&[7i16, 8], ..).unwrap();
    file.add_group("new").unwrap();

    let file = read(file.into_bytes().unwrap());
    assert_eq!(
        dimensions(&file),
        [
            ("time".to_owned(), 1, true),
            ("lat".to_owned(), 3, false),
            ("depth".to_owned(), 2, false)
        ]
    );
    let value = |name: &str| {
        file.attribute(name)
            .unwrap()
            .value
            .as_text()
            .map(str::to_owned)
    };
    assert_eq!(value("title").as_deref(), Some("second"));
    assert_eq!(value("history").as_deref(), Some("appended"));
    let psal = file.variable("PSAL").unwrap();
    assert_eq!(psal.dimensions, ["time", "lat", "depth"]);
    assert_eq!(psal.get_values::<f64, _>(..).unwrap(), [1.5; 6]);
    let name = file.variable("name").unwrap();
    assert_eq!(name.get_strings(..).unwrap(), ["south", "equator", "north"]);
    assert_eq!(name.attributes().len(), 1);
    let more = file.variable("/sub/more").unwrap();
    assert_eq!(more.dimensions, ["z"]);
    assert_eq!(more.get_values::<i16, _>(..).unwrap(), [7, 8]);
    assert!(file.group("/new").is_some());
    assert_eq!(file.variables().len(), 6);
}

#[test]
fn a_variable_of_the_file_keeps_its_definition() {
    let mut file = oxcdf::append_in_memory(first_file().into_bytes().unwrap()).unwrap();
    let mut temp = file.variable_mut("TEMP").unwrap();
    let bad = |result: Result<(), Error>| matches!(result, Err(Error::BadRequest(_)));
    assert!(bad(temp.set_chunking(&[1, 3])));
    assert!(bad(temp.set_fill_value(0.0f32)));
    assert!(bad(temp.set_compression(1, false)));
    assert!(bad(file.add_dimension("lat", 4)));
    assert!(bad(file.add_variable::<i32>("TEMP", &["lat"]).map(|_| ())));
    assert!(bad(file.add_group("sub").map(|_| ())));
    let file = read(file.into_bytes().unwrap());
    assert_eq!(file.variables().len(), 4);
}

#[test]
fn a_file_takes_many_append_sessions() {
    let mut bytes = first_file().into_bytes().unwrap();
    for record in 1..6usize {
        let mut file = oxcdf::append_in_memory(bytes).unwrap();
        let mut temp = file.variable_mut("TEMP").unwrap();
        let values = [record as f32; 3];
        temp.put_values(&values, [record..record + 1, 0..3])
            .unwrap();
        bytes = file.into_bytes().unwrap();
    }
    let file = read(bytes);
    let temp = file.variable("TEMP").unwrap();
    assert_eq!(temp.shape, [6, 3]);
    let values = temp.get_values::<f32, _>(..).unwrap();
    assert_eq!(values[..3], [1.0, 2.0, 3.0]);
    assert_eq!(values[15..], [5.0; 3]);
}

#[test]
fn a_file_on_disk_takes_an_append_and_one_writer() {
    let dir = std::env::temp_dir().join(format!("oxcdf-append-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("out.nc");
    std::fs::write(&path, first_file().into_bytes().unwrap()).unwrap();

    let mut file = oxcdf::append(&path).unwrap();
    assert!(matches!(oxcdf::append(&path), Err(Error::Io(_))));
    let mut temp = file.variable_mut("TEMP").unwrap();
    temp.put_values(&[4.0f32, 5.0, 6.0], [1..2, 0..3]).unwrap();
    file.add_attribute("history", "appended").unwrap();
    file.close().unwrap();

    let file = oxcdf::open(&path).unwrap();
    let temp = file.variable("TEMP").unwrap();
    assert_eq!(
        temp.get_values::<f32, _>(..).unwrap(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    let history = file.attribute("history").unwrap();
    assert!(matches!(&history.value, AttributeValue::Str(s) if s == "appended"));
    drop(file);
    // A drop writes the file, as a close does.
    let mut file = oxcdf::append(&path).unwrap();
    file.add_attribute("history", "dropped").unwrap();
    drop(file);
    let file = oxcdf::open(&path).unwrap();
    assert_eq!(
        file.attribute("history").unwrap().value.as_text(),
        Some("dropped")
    );
    drop(file);
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A new directory for the test `tag` with the file of [`first_file`] in it.
fn disk_file(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let name = format!("oxcdf-append-{tag}-{}", std::process::id());
    let dir = std::env::temp_dir().join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("out.nc");
    std::fs::write(&path, first_file().into_bytes().unwrap()).unwrap();
    (dir, path)
}

/// The text of the attribute `history` of the file at `path`.
fn history(path: &std::path::Path) -> Option<String> {
    let file = oxcdf::open(path).unwrap();
    let attribute = file.attribute("history")?;
    attribute.value.as_text().map(str::to_owned)
}

#[test]
fn a_reader_opens_the_file_during_an_append() {
    let (dir, path) = disk_file("reader");
    let mut file = oxcdf::append(&path).unwrap();
    let mut temp = file.variable_mut("TEMP").unwrap();
    temp.put_values(&[4.0f32, 5.0, 6.0], [1..2, 0..3]).unwrap();
    file.add_attribute("history", "appended").unwrap();

    // Until the close, a reader sees the structure and values before the append.
    let reader = oxcdf::open(&path).unwrap();
    let temp = reader.variable("TEMP").unwrap();
    assert_eq!(temp.shape, [1, 3]);
    assert_eq!(temp.get_values::<f32, _>(..).unwrap(), [1.0, 2.0, 3.0]);
    assert!(reader.attribute("history").is_none());
    file.close().unwrap();
    drop(reader);

    let reader = oxcdf::open(&path).unwrap();
    let temp = reader.variable("TEMP").unwrap();
    assert_eq!(
        temp.get_values::<f32, _>(..).unwrap(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    drop(reader);
    assert_eq!(history(&path).as_deref(), Some("appended"));
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_new_coordinate_variable_for_a_dimension_of_the_file_is_unsupported() {
    let mut file = oxcdf::create_in_memory().unwrap();
    file.add_unlimited_dimension("time").unwrap();
    let mut v = file.add_variable::<i32>("v", &["time"]).unwrap();
    v.put_values(&[1, 2], [0..2]).unwrap();
    let bytes = file.into_bytes().unwrap();

    let mut file = oxcdf::append_in_memory(bytes.clone()).unwrap();
    let error = file.add_variable::<f64>("time", &["time"]).unwrap_err();
    assert!(error.is_fallback_worthy(), "{error}");
    assert!(error.to_string().contains("\"time\""), "{error}");
    // The refused call changes nothing.
    let after = read(file.into_bytes().unwrap());
    assert_eq!(content(&after), content(&read(bytes)));
}

#[test]
fn into_bytes_writes_a_file_from_append_in_place() {
    let (dir, path) = disk_file("into-bytes");
    let mut file = oxcdf::append(&path).unwrap();
    let mut temp = file.variable_mut("TEMP").unwrap();
    temp.put_values(&[4.0f32, 5.0, 6.0], [1..2, 0..3]).unwrap();
    file.add_attribute("history", "kept").unwrap();
    let error = file.into_bytes().unwrap_err();
    assert!(
        matches!(&error, Error::BadRequest(m) if m.contains("use close")),
        "{error}"
    );

    assert_eq!(history(&path).as_deref(), Some("kept"));
    let reader = oxcdf::open(&path).unwrap();
    assert_eq!(reader.dimension_len("time"), Some(2));
    drop(reader);
    // The refused call frees the file for the next writer.
    oxcdf::append(&path).unwrap().close().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}

#[cfg(feature = "object-store")]
#[tokio::test]
async fn close_to_store_writes_a_file_from_append_in_place() {
    use object_store::ObjectStoreExt;
    let (dir, path) = disk_file("store");
    let store = object_store::memory::InMemory::new();
    let location = object_store::path::Path::from("out.nc");
    let mut file = oxcdf::append(&path).unwrap();
    file.add_attribute("history", "kept").unwrap();
    let error = file.close_to_store(&store, &location).await.unwrap_err();
    assert!(
        matches!(&error, Error::BadRequest(m) if m.contains("use close")),
        "{error}"
    );

    assert!(store.get(&location).await.is_err());
    assert_eq!(history(&path).as_deref(), Some("kept"));
    oxcdf::append(&path).unwrap().close().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn bytes_that_are_no_netcdf_4_file_are_rejected() {
    assert!(oxcdf::append_in_memory(b"not a file".to_vec()).is_err());
    // A classic file starts with `CDF`.
    let classic = b"CDF\x01\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0".to_vec();
    assert!(oxcdf::append_in_memory(classic).is_err());
}

/// Each value of a variable as text, or the error of the read.
fn values(variable: &oxcdf::Variable<'_>) -> String {
    match variable.get_values::<f64, _>(..) {
        Ok(numbers) => format!(
            "{:?}",
            numbers.iter().map(|n| n.to_bits()).collect::<Vec<_>>()
        ),
        Err(_) => format!("{:?}", variable.get_strings(..).map_err(|e| e.to_string())),
    }
}

/// The dimensions, attributes and variables of a file as text, with each value.
fn content(file: &NetcdfFile) -> Vec<String> {
    fn walk(file: &NetcdfFile, group: &oxcdf::netcdf::NcGroup, out: &mut Vec<String>) {
        for d in &group.dimensions {
            out.push(format!(
                "{} dim {} {} {}",
                group.path, d.name, d.len, d.is_unlimited
            ));
        }
        for a in &group.attributes {
            out.push(format!("{} att {} {:?}", group.path, a.name, a.value));
        }
        for info in &group.variables {
            let v = file.variable(&info.path).expect("the variable opens");
            let head = format!("{} {:?} {:?}", info.path, info.dimensions, info.shape);
            out.push(format!("{head} {:?} {:?}", v.vartype(), v.attributes()));
            out.push(format!("{} = {}", info.path, values(&v)));
        }
        for child in &group.groups {
            walk(file, child, out);
        }
    }
    let mut out = Vec::new();
    walk(file, file.root(), &mut out);
    out
}

#[test]
fn an_append_keeps_the_content_of_each_corpus_file() {
    let dir = format!("{}/../../test_files", env!("CARGO_MANIFEST_DIR"));
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let mut paths: Vec<_> = entries.map(|e| e.unwrap().path()).collect();
    paths.sort();
    let mut appended = Vec::new();
    for path in paths {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !name.ends_with(".nc") && !name.ends_with(".h5") {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let mut file = match oxcdf::append_in_memory(bytes.clone()) {
            Ok(file) => file,
            Err(Error::Unsupported(reason)) => {
                println!("{name}: no append: {reason}");
                continue;
            }
            Err(other) => {
                println!("{name}: no append: {other}");
                continue;
            }
        };
        file.add_attribute("oxcdf_append", "yes").unwrap();
        let after = read(file.into_bytes().unwrap());
        let mut expected = content(&read(bytes));
        let added = "/ att oxcdf_append Str(\"yes\")".to_owned();
        let at = expected.iter().rposition(|line| line.starts_with("/ att "));
        let at = at.map_or_else(
            || {
                expected
                    .iter()
                    .take_while(|l| l.starts_with("/ dim "))
                    .count()
            },
            |at| at + 1,
        );
        expected.insert(at, added);
        assert_eq!(content(&after), expected, "{name}");
        appended.push(name);
    }
    println!("appended: {appended:?}");
}

/// Append to the file that `scripts/check_writer.py` makes with netcdf-c.
///
/// The script sets `OXCDF_EXPORT_DIR` and checks the result with other libraries.
#[test]
#[ignore = "run by scripts/check_writer.py"]
fn a_python_file_takes_an_append() {
    let dir = std::env::var("OXCDF_EXPORT_DIR").expect("OXCDF_EXPORT_DIR names a directory");
    let mut file = oxcdf::append(std::path::Path::new(&dir).join("python.nc")).unwrap();
    assert_eq!(file.dimension_len("time"), Some(1));
    let mut temp = file.variable_mut("TEMP").unwrap();
    temp.put_values(&[4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0], [1..3, 0..3])
        .unwrap();
    temp.put_attribute("long_name", "temperature").unwrap();
    let mut name = file.variable_mut("name").unwrap();
    name.put_string("north", [2]).unwrap();
    file.add_attribute("history", "appended by oxcdf").unwrap();
    file.add_attribute("attr_03", 33i32).unwrap();
    file.add_dimension("depth", 2).unwrap();
    let mut psal = file
        .add_variable::<f64>("PSAL", &["time", "depth"])
        .unwrap();
    psal.put_values(&[0.5f64, 1.5, 2.5, 3.5, 4.5, 5.5], ..)
        .unwrap();
    let mut sub = file.group_mut("sub").unwrap();
    let mut more = sub.add_variable::<i16>("more", &["z"]).unwrap();
    more.put_values(&[7i16, 8], ..).unwrap();
    file.close().unwrap();
}
