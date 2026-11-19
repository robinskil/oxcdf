#![cfg(feature = "write")]
//! The writer opens a file that exists, changes it, and the reader reads the result.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p oxcdf-hdf5 --features write --test write_edit
//! ```

use std::sync::{Arc, Mutex};

use oxcdf_hdf5::checksum;
use oxcdf_hdf5::message::{Attribute, ByteOrder};
use oxcdf_hdf5::read::{read_hyperslab, resolve_vlen_strings, resolve_vlen_strings_of};
use oxcdf_hdf5::write::{ByteSink, DatasetSpec, Filter, Hdf5Writer, MemorySink, WriteOptions};
use oxcdf_hdf5::{DType, Element, Error, Hdf5File, Hyperslab, MemorySource, RawData, Result};

fn reopen(bytes: Vec<u8>) -> Hdf5File {
    Hdf5File::from_source(Arc::new(MemorySource::new(bytes))).unwrap()
}

fn read_all<T: Element>(file: &Hdf5File, path: &str) -> Vec<T> {
    let d = file.dataset(path).unwrap();
    d.prepare(file.ctx()).unwrap();
    read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
        .unwrap()
        .get(d)
        .unwrap()
}

fn read_strings(file: &Hdf5File, path: &str) -> Vec<String> {
    let d = file.dataset(path).unwrap();
    d.prepare(file.ctx()).unwrap();
    let raw = read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape)).unwrap();
    resolve_vlen_strings(file.ctx(), d, &raw).unwrap()
}

fn int4(shape: &[u64]) -> DatasetSpec {
    DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, shape).unwrap()
}

/// Open the file `bytes` for edits.
fn edit(bytes: Vec<u8>) -> Hdf5Writer<MemorySink> {
    let file = reopen(bytes.clone());
    Hdf5Writer::open(
        MemorySink::from_bytes(bytes),
        &file,
        WriteOptions::default(),
    )
    .unwrap()
}

/// A file with each kind of object that the writer makes.
fn first_file() -> Vec<u8> {
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
    let root = w.root();
    w.put_text_attribute(root, "title", "first").unwrap();
    let fixed = w.add_dataset(root, "fixed", int4(&[4])).unwrap();
    w.write(fixed, &Hyperslab::all(&[4]), &[1i32, 2, 3, 4])
        .unwrap();
    w.put_text_attribute(fixed, "units", "m").unwrap();
    let grow = int4(&[2])
        .chunked(&[2])
        .with_max_shape(&[None])
        .with_filters(&[Filter::Shuffle, Filter::Deflate(4)])
        .with_fill_value((-1i32).to_le_bytes().to_vec());
    let grow = w.add_dataset(root, "grow", grow).unwrap();
    w.write(grow, &Hyperslab::all(&[2]), &[10i32, 20]).unwrap();
    let group = w.add_group(root, "g").unwrap();
    let names = w
        .add_dataset(group, "names", DatasetSpec::strings(&[3]))
        .unwrap();
    w.write_strings(names, &Hyperslab::all(&[3]), &["a", "", "ccc"])
        .unwrap();
    w.put_string_attribute(group, "tags", &["x", "yz"]).unwrap();
    let scale = w.add_dataset(root, "scale", int4(&[4])).unwrap();
    w.attach_scale(fixed, 0, scale).unwrap();
    w.close().unwrap().into_inner()
}

#[test]
fn an_edit_with_no_change_keeps_the_content() {
    let bytes = edit(first_file()).close().unwrap().into_inner();
    let file = reopen(bytes);
    assert_eq!(read_all::<i32>(&file, "/fixed"), [1, 2, 3, 4]);
    assert_eq!(read_all::<i32>(&file, "/grow"), [10, 20]);
    assert_eq!(read_strings(&file, "/g/names"), ["a", "", "ccc"]);
    assert_eq!(file.root().attributes.len(), 1);
    let fixed = file.dataset("/fixed").unwrap();
    let names: Vec<&str> = fixed.attributes.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names, ["units", "DIMENSION_LIST"]);
    let scale = file.dataset("/scale").unwrap();
    let back = scale.attribute("REFERENCE_LIST").unwrap();
    assert_eq!(back.data[..8], fixed.address.to_le_bytes());
    assert_eq!(file.root().group("g").unwrap().attributes.len(), 1);
}

#[test]
fn an_edit_changes_values_and_adds_objects() {
    let mut w = edit(first_file());
    let root = w.root();
    let fixed = w.find_dataset(root, "fixed").unwrap();
    let part = Hyperslab::new(vec![1], vec![2], &[4]).unwrap();
    w.write(fixed, &part, &[20i32, 30]).unwrap();
    w.put_text_attribute(fixed, "units", "km").unwrap();
    w.put_numeric_attribute(fixed, "scale", &[0.5f64]).unwrap();

    let grow = w.find_dataset(root, "grow").unwrap();
    w.resize(grow, &[5]).unwrap();
    // The chunk at 0 is in the file with its filters. The chunk at 4 is new.
    let tail = Hyperslab::new(vec![1], vec![4], &[5]).unwrap();
    w.write(grow, &tail, &[21i32, 30, 40, 50]).unwrap();

    let group = w.find_group(root, "g").unwrap();
    let names = w.find_dataset(group, "names").unwrap();
    let one = Hyperslab::new(vec![1], vec![1], &[3]).unwrap();
    w.write_strings(names, &one, &["bb"]).unwrap();
    let new = w.add_dataset(group, "new", int4(&[2])).unwrap();
    w.write(new, &Hyperslab::all(&[2]), &[7i32, 8]).unwrap();
    let scale = w.find_dataset(root, "scale").unwrap();
    w.attach_scale(new, 0, scale).unwrap();
    w.add_group(root, "h").unwrap();
    assert!(w.find_dataset(root, "none").is_none());
    assert!(w.find_group(root, "fixed").is_none());

    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(read_all::<i32>(&file, "/fixed"), [1, 20, 30, 4]);
    assert_eq!(read_all::<i32>(&file, "/grow"), [10, 21, 30, 40, 50]);
    assert_eq!(read_strings(&file, "/g/names"), ["a", "bb", "ccc"]);
    assert_eq!(read_all::<i32>(&file, "/g/new"), [7, 8]);
    assert!(file.root().group("h").is_some());
    let fixed = file.dataset("/fixed").unwrap();
    let names: Vec<&str> = fixed.attributes.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names, ["units", "scale", "DIMENSION_LIST"]);
    assert_eq!(fixed.attribute("units").unwrap().data, b"km");
    let scale = file.dataset("/scale").unwrap();
    assert_eq!(
        scale.attribute("REFERENCE_LIST").unwrap().dataspace.dims,
        [2]
    );
    assert!(file
        .dataset("/g/new")
        .unwrap()
        .attribute("DIMENSION_LIST")
        .is_some());
}

#[test]
fn a_file_takes_three_edit_sessions() {
    let mut bytes = first_file();
    for session in 0..3u64 {
        let mut w = edit(bytes);
        let root = w.root();
        let grow = w.find_dataset(root, "grow").unwrap();
        let len = 2 + session;
        w.resize(grow, &[len + 1]).unwrap();
        let last = Hyperslab::new(vec![len], vec![1], &[len + 1]).unwrap();
        w.write(grow, &last, &[100 + session as i32]).unwrap();
        bytes = w.close().unwrap().into_inner();
    }
    let file = reopen(bytes);
    assert_eq!(read_all::<i32>(&file, "/grow"), [10, 20, 100, 101, 102]);
    assert_eq!(read_all::<i32>(&file, "/fixed"), [1, 2, 3, 4]);
}

/// A sink that loses each write to the superblock, as a failure before the last write does.
#[derive(Debug)]
struct NoSuperblock(MemorySink);

impl ByteSink for NoSuperblock {
    fn write_all_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        match offset {
            0 => Ok(()),
            _ => self.0.write_all_at(offset, data),
        }
    }

    fn set_len(&mut self, len: u64) -> Result<()> {
        self.0.set_len(len)
    }

    fn sync(&mut self) -> Result<()> {
        self.0.sync()
    }
}

#[test]
fn the_earlier_file_is_valid_until_the_superblock_write() {
    let bytes = first_file();
    let file = reopen(bytes.clone());
    let sink = NoSuperblock(MemorySink::from_bytes(bytes));
    let mut w = Hdf5Writer::open(sink, &file, WriteOptions::default()).unwrap();
    let root = w.root();
    let new = w.add_dataset(root, "new", int4(&[2])).unwrap();
    w.write(new, &Hyperslab::all(&[2]), &[7i32, 8]).unwrap();
    w.put_text_attribute(root, "title", "second").unwrap();
    let grow = w.find_dataset(root, "grow").unwrap();
    w.resize(grow, &[4]).unwrap();
    let tail = Hyperslab::new(vec![2], vec![2], &[4]).unwrap();
    w.write(grow, &tail, &[30i32, 40]).unwrap();

    let file = reopen(w.close().unwrap().0.into_inner());
    assert!(file.dataset("/new").is_none());
    assert_eq!(file.root().attributes[0].data, b"first");
    assert_eq!(read_all::<i32>(&file, "/grow"), [10, 20]);
    assert_eq!(read_all::<i32>(&file, "/fixed"), [1, 2, 3, 4]);
    assert_eq!(read_strings(&file, "/g/names"), ["a", "", "ccc"]);
}

#[test]
fn an_edited_dataset_keeps_its_definition() {
    let mut w = edit(first_file());
    let root = w.root();
    let fixed = w.find_dataset(root, "fixed").unwrap();
    assert!(matches!(
        w.redefine_dataset(fixed, int4(&[8])),
        Err(Error::BadRequest(_))
    ));
    assert!(matches!(
        w.add_dataset(root, "fixed", int4(&[1])),
        Err(Error::BadRequest(_))
    ));
    let file = reopen(w.close().unwrap().into_inner());
    assert_eq!(read_all::<i32>(&file, "/fixed"), [1, 2, 3, 4]);
}

/// The datasets of a file as sorted text, with each attribute and the bytes of each value.
fn content(file: &Hdf5File) -> Vec<String> {
    let mut out = Vec::new();
    for d in file.datasets() {
        out.push(format!(
            "{} {:?} {:?} {:?}",
            d.path, d.shape, d.max_shape, d.datatype
        ));
        for a in &d.attributes {
            // The scale lists hold addresses, and each object header moves at an edit.
            let data = match a.name.as_str() {
                "DIMENSION_LIST" | "REFERENCE_LIST" => a.data.len().to_string(),
                _ => {
                    // The reader keeps the pad bytes of a version 1 attribute message.
                    let bytes = a.dataspace.element_count() * u64::from(a.datatype.size);
                    format!("{:?}", &a.data[..(bytes as usize).min(a.data.len())])
                }
            };
            out.push(format!(
                "{} @{} {:?} {data}",
                d.path, a.name, a.dataspace.dims
            ));
        }
        let values = d
            .prepare(file.ctx())
            .and_then(|()| read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape)));
        // A reference to the heap stays the same, because no heap object moves.
        out.push(match values {
            Ok(raw) => format!("{} = {:?}", d.path, raw.bytes),
            Err(error) => format!("{} = {error}", d.path),
        });
    }
    // The close puts the scale lists after the other attributes.
    out.sort();
    out
}

#[test]
fn an_edit_with_no_change_keeps_each_corpus_file() {
    let dir = format!("{}/../../test_files", env!("CARGO_MANIFEST_DIR"));
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let mut paths: Vec<_> = entries.map(|e| e.unwrap().path()).collect();
    paths.sort();
    let mut edited = Vec::new();
    for path in paths {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let bytes = std::fs::read(&path).unwrap();
        if !bytes.starts_with(b"\x89HDF") {
            continue;
        }
        let before = reopen(bytes.clone());
        let sink = MemorySink::from_bytes(bytes);
        let writer = match Hdf5Writer::open(sink, &before, WriteOptions::default()) {
            Ok(writer) => writer,
            Err(error) => {
                println!(
                    "{name}: no edit: {}",
                    error.to_string().chars().take(120).collect::<String>()
                );
                continue;
            }
        };
        let after = reopen(writer.close().unwrap().into_inner());
        let (found, expected) = (content(&after), content(&before));
        for (found, expected) in found.iter().zip(&expected) {
            assert_eq!(found, expected, "{name}");
        }
        assert_eq!(found.len(), expected.len(), "{name}");
        assert_eq!(
            after.root().attributes.len(),
            before.root().attributes.len(),
            "{name}"
        );
        edited.push(name);
    }
    println!("edited: {edited:?}");
    assert!(edited.len() >= 11, "{edited:?}");
}

/// One message frame of a version 2 object header.
struct Frame {
    /// Offset of the frame in the file.
    at: usize,
    /// Message type code.
    kind: u8,
    /// Body size.
    size: usize,
    /// Offset of the body in the file.
    body: usize,
}

/// The frames of the version 2 object header at `address`, and the end of its message area.
fn frames(bytes: &[u8], address: u64) -> (Vec<Frame>, usize) {
    let start = address as usize;
    assert_eq!(&bytes[start..start + 4], b"OHDR");
    let flags = bytes[start + 5];
    let mut at = start + 6;
    if flags & 0x20 != 0 {
        at += 16;
    }
    if flags & 0x10 != 0 {
        at += 4;
    }
    let width = 1usize << (flags & 0x03);
    let mut area = [0u8; 8];
    area[..width].copy_from_slice(&bytes[at..at + width]);
    at += width;
    let end = at + u64::from_le_bytes(area) as usize;
    let frame_bytes = if flags & 0x04 != 0 { 6 } else { 4 };
    let mut out = Vec::new();
    while at + frame_bytes <= end {
        let size = usize::from(u16::from_le_bytes([bytes[at + 1], bytes[at + 2]]));
        out.push(Frame {
            at,
            kind: bytes[at],
            size,
            body: at + frame_bytes,
        });
        at += frame_bytes + size;
    }
    (out, end)
}

/// Write the checksum of the object header at `address` again.
fn reseal(bytes: &mut [u8], address: u64) {
    let (_, end) = frames(bytes, address);
    let sum = checksum::metadata(&bytes[address as usize..end]);
    bytes[end..end + 4].copy_from_slice(&sum.to_le_bytes());
}

/// The body offset of the first message of type `kind` in the header at `address`.
fn body_of(bytes: &[u8], address: u64, kind: u8) -> usize {
    let (frames, _) = frames(bytes, address);
    frames.iter().find(|f| f.kind == kind).unwrap().body
}

/// Put a message in the slack of the header at `address`.
///
/// The rest of the slack holds zeros, which read as empty `Nil` messages.
fn put_message(bytes: &mut [u8], address: u64, kind: u8, flags: u8, body: &[u8]) {
    let (frames, _) = frames(bytes, address);
    let slack = frames
        .iter()
        .find(|f| f.kind == 0 && f.size >= body.len())
        .unwrap();
    let (at, start, size) = (slack.at, slack.body, slack.size);
    bytes[at] = kind;
    bytes[at + 1..at + 3].copy_from_slice(&(body.len() as u16).to_le_bytes());
    bytes[at + 3] = flags;
    bytes[start..start + size].fill(0);
    bytes[start..start + body.len()].copy_from_slice(body);
    reseal(bytes, address);
}

/// The object header address of the group or dataset `path`.
fn address_of(file: &Hdf5File, path: &str) -> u64 {
    if let Some(d) = file.dataset(path) {
        return d.address;
    }
    let mut group = file.root();
    for part in path.split('/').filter(|p| !p.is_empty()) {
        group = group.group(part).unwrap();
    }
    group.address
}

/// The error of an edit of `bytes`.
fn edit_error(bytes: Vec<u8>) -> Error {
    let file = reopen(bytes.clone());
    let sink = MemorySink::from_bytes(bytes);
    match Hdf5Writer::open(sink, &file, WriteOptions::default()) {
        Ok(_) => panic!("the edit opens"),
        Err(error) => error,
    }
}

/// Assert that `error` is unsupported and names `path`.
fn assert_unsupported_at(error: &Error, path: &str) {
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
    assert!(error.to_string().contains(path), "{error}");
}

#[test]
fn an_attribute_that_the_reader_skips_stops_the_edit() {
    let mut bytes = first_file();
    let address = address_of(&reopen(bytes.clone()), "/fixed");
    // The attribute message version 9 does not exist.
    let at = body_of(&bytes, address, 0x0C);
    bytes[at] = 9;
    reseal(&mut bytes, address);
    let file = reopen(bytes.clone());
    assert!(!file.dataset("/fixed").unwrap().attributes_complete);
    assert!(file.root().attributes_complete);
    assert_unsupported_at(&edit_error(bytes), "/fixed");
}

#[test]
fn a_filter_level_that_the_writer_refuses_stops_the_edit_as_unsupported() {
    let mut bytes = first_file();
    let address = address_of(&reopen(bytes.clone()), "/grow");
    // Version 2: version and count, shuffle in 10 bytes, then the deflate id, flags and count.
    let level = body_of(&bytes, address, 0x0B) + 2 + 10 + 6;
    assert_eq!(bytes[level..level + 4], 4u32.to_le_bytes());
    bytes[level..level + 4].copy_from_slice(&10u32.to_le_bytes());
    reseal(&mut bytes, address);
    assert_unsupported_at(&edit_error(bytes), "/grow");
}

#[test]
fn an_attribute_with_too_few_bytes_stops_the_edit_as_unsupported() {
    let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
    let x = w.add_dataset(w.root(), "x", int4(&[2])).unwrap();
    w.put_numeric_attribute(x, "range", &[1i32, 2]).unwrap();
    let mut bytes = w.close().unwrap().into_inner();
    let address = address_of(&reopen(bytes.clone()), "/x");
    let at = body_of(&bytes, address, 0x0C);
    let size = |at: usize| usize::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
    // Version 3: 9 fixed bytes, the name, the datatype, then 4 bytes before the first dimension.
    let dim = at + 9 + size(at + 2) + size(at + 4) + 4;
    assert_eq!(bytes[dim..dim + 8], 2u64.to_le_bytes());
    bytes[dim..dim + 8].copy_from_slice(&3u64.to_le_bytes());
    reseal(&mut bytes, address);
    let file = reopen(bytes.clone());
    assert_eq!(file.dataset("/x").unwrap().attributes[0].data.len(), 8);
    assert_unsupported_at(&edit_error(bytes), "/x");
}

#[test]
fn a_header_message_that_the_close_drops_stops_the_edit() {
    let time = [1, 0, 0, 0, 0x40, 0x42, 0x0F, 0x00];
    let cases: [(&str, u8, u8, &[u8], &str); 5] = [
        ("/fixed", 0x0D, 0, b"a note\0", "ObjectComment"),
        ("/fixed", 0x07, 0, &[0; 24], "ExternalDataFiles"),
        ("/", 0x09, 0, &[0, 0, 0, 0], "Bogus"),
        ("/g", 0x30, 0, &[1, 2, 3, 4], "Unknown"),
        ("/scale", 0x12, 0x02, &time, "shared"),
    ];
    for (path, kind, flags, body, name) in cases {
        let mut bytes = first_file();
        let address = address_of(&reopen(bytes.clone()), path);
        put_message(&mut bytes, address, kind, flags, body);
        let error = edit_error(bytes);
        assert_unsupported_at(&error, path);
        assert!(error.to_string().contains(name), "{error}");
    }
}

#[test]
fn a_modification_time_does_not_stop_the_edit() {
    let mut bytes = first_file();
    let address = address_of(&reopen(bytes.clone()), "/fixed");
    let time = [1, 0, 0, 0, 0x40, 0x42, 0x0F, 0x00];
    put_message(&mut bytes, address, 0x12, 0, &time);
    let file = reopen(edit(bytes).close().unwrap().into_inner());
    assert_eq!(read_all::<i32>(&file, "/fixed"), [1, 2, 3, 4]);
}

#[test]
fn a_link_that_the_index_does_not_show_stops_the_edit() {
    // Version 1 with a link type and a 1-byte name length, the name, then a 2-byte value length.
    let soft = [&[1, 0x08, 1, 5][..], b"alias", &[2, 0], b"/g"].concat();
    let external = [&[1, 0x08, 64, 4][..], b"away", &[9, 0], b"\0f.h5\0/x\0"].concat();
    for (path, body) in [("/", soft), ("/g", external)] {
        let mut bytes = first_file();
        let address = address_of(&reopen(bytes.clone()), path);
        put_message(&mut bytes, address, 0x06, 0, &body);
        assert_eq!(reopen(bytes.clone()).datasets().len(), 4);
        assert_unsupported_at(&edit_error(bytes), path);
    }
}

/// The state that a [`FailOnce`] sink shares with the test.
#[derive(Debug, Default)]
struct Shared {
    /// The bytes of the file.
    bytes: Vec<u8>,
    /// The number of sink calls.
    calls: usize,
    /// Whether a write to the superblock occurs.
    superblock: bool,
}

/// A sink that fails one time, at call number `fail_at`.
#[derive(Debug)]
struct FailOnce {
    shared: Arc<Mutex<Shared>>,
    fail_at: usize,
}

impl FailOnce {
    fn step(&mut self) -> Result<std::sync::MutexGuard<'_, Shared>> {
        let mut shared = self.shared.lock().unwrap();
        shared.calls += 1;
        if shared.calls - 1 == self.fail_at {
            return Err(Error::Io(std::io::ErrorKind::Other.into()));
        }
        Ok(shared)
    }
}

impl ByteSink for FailOnce {
    fn write_all_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        let mut shared = self.step()?;
        let end = offset as usize + data.len();
        if shared.bytes.len() < end {
            shared.bytes.resize(end, 0);
        }
        shared.bytes[offset as usize..end].copy_from_slice(data);
        shared.superblock |= offset == 0 && !data.is_empty();
        Ok(())
    }

    fn set_len(&mut self, len: u64) -> Result<()> {
        self.step()?.bytes.resize(len as usize, 0);
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.step().map(|_| ())
    }

    fn can_read(&self) -> bool {
        true
    }

    fn read_exact_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let shared = self.step()?;
        let range = offset as usize..offset as usize + buf.len();
        let bytes = shared
            .bytes
            .get(range)
            .ok_or_else(|| Error::Io(std::io::ErrorKind::UnexpectedEof.into()))?;
        buf.copy_from_slice(bytes);
        Ok(())
    }
}

/// The long strings of the edit session. Each one fills a heap collection.
fn long_strings() -> [String; 2] {
    ["p".repeat(40_000), "q".repeat(70_000)]
}

/// Run a call. A failed call gets one more try, which must succeed.
macro_rules! retry {
    ($failed:ident, $call:expr) => {
        match $call {
            Ok(value) => value,
            Err(_) => {
                $failed += 1;
                $call.unwrap()
            }
        }
    };
}

/// Run the edit session of the sink test on `first` with a failure at call `fail_at`.
///
/// Return the shared state, the number of failed calls before the close, and the close result.
fn edit_session(first: &[u8], fail_at: usize) -> (Arc<Mutex<Shared>>, usize, bool) {
    let shared = Arc::new(Mutex::new(Shared {
        bytes: first.to_vec(),
        ..Shared::default()
    }));
    let sink = FailOnce {
        shared: Arc::clone(&shared),
        fail_at,
    };
    let file = reopen(first.to_vec());
    let mut w = Hdf5Writer::open(sink, &file, WriteOptions::default()).unwrap();
    let mut failed = 0;
    let root = w.root();
    let fixed = w.find_dataset(root, "fixed").unwrap();
    let part = Hyperslab::new(vec![1], vec![2], &[4]).unwrap();
    retry!(failed, w.write(fixed, &part, &[20i32, 30]));
    retry!(failed, w.put_text_attribute(fixed, "units", "km"));
    let grow = w.find_dataset(root, "grow").unwrap();
    retry!(failed, w.resize(grow, &[5]));
    let tail = Hyperslab::new(vec![1], vec![4], &[5]).unwrap();
    retry!(failed, w.write(grow, &tail, &[21i32, 30, 40, 50]));
    let group = w.find_group(root, "g").unwrap();
    let names = w.find_dataset(group, "names").unwrap();
    let one = Hyperslab::new(vec![1], vec![1], &[3]).unwrap();
    retry!(failed, w.write_strings(names, &one, &[&long_strings()[0]]));
    retry!(
        failed,
        w.put_string_attribute(group, "tags", &long_strings())
    );
    let new = retry!(failed, w.add_dataset(group, "new", int4(&[4])));
    retry!(
        failed,
        w.write(new, &Hyperslab::all(&[4]), &[7i32, 8, 9, 10])
    );
    let scale = w.find_dataset(root, "scale").unwrap();
    retry!(failed, w.attach_scale(new, 0, scale));
    retry!(failed, w.put_numeric_attribute(new, "factor", &[0.5f64]));
    // Before the close, a reader sees the earlier structure.
    let before = shared.lock().unwrap().bytes.clone();
    check_earlier(
        before,
        &format!("failure at call {fail_at}, before the close"),
    );
    let closed = w.close().is_ok();
    (shared, failed, closed)
}

/// The value of each attribute of `attributes`, with the length of each string from the heap.
fn values_of(file: &Hdf5File, attributes: &[Attribute]) -> Vec<(String, String)> {
    let value = |a: &Attribute| match a.name.as_str() {
        "DIMENSION_LIST" | "REFERENCE_LIST" => format!("{:?}", a.dataspace.dims),
        _ if a.datatype.size == 16 => {
            let raw = RawData {
                bytes: a.data.clone(),
                element_size: 16,
                shape: a.dataspace.dims.clone(),
            };
            let strings = resolve_vlen_strings_of(file.ctx(), &a.name, &raw).unwrap();
            let lens: Vec<usize> = strings.iter().map(String::len).collect();
            format!("{lens:?}")
        }
        _ => format!("{:?}", a.data),
    };
    attributes
        .iter()
        .map(|a| (a.name.clone(), value(a)))
        .collect()
}

/// Check that `bytes` read as the file of [`first_file`].
///
/// A write to values of the file changes them at once, so `/fixed` and `/g/names` can hold the new values.
fn check_earlier(bytes: Vec<u8>, case: &str) {
    let earlier = reopen(first_file());
    let file = Hdf5File::from_source(Arc::new(MemorySource::new(bytes))).expect(case);
    let paths = |file: &Hdf5File| -> Vec<String> {
        let datasets = file.datasets().into_iter();
        datasets
            .map(|d| format!("{} {:?}", d.path, d.shape))
            .collect()
    };
    assert_eq!(paths(&file), paths(&earlier), "{case}");
    for d in earlier.datasets() {
        let got = file.dataset(&d.path).unwrap();
        let got = values_of(&file, &got.attributes);
        assert_eq!(got, values_of(&earlier, &d.attributes), "{case} {}", d.path);
    }
    let groups = |file: &Hdf5File| {
        let g = file.root().group("g").unwrap();
        let root = values_of(file, &file.root().attributes);
        (root, values_of(file, &g.attributes))
    };
    assert_eq!(groups(&file), groups(&earlier), "{case}");
    let fixed = read_all::<i32>(&file, "/fixed");
    assert!(
        fixed == [1, 2, 3, 4] || fixed == [1, 20, 30, 4],
        "{case} {fixed:?}"
    );
    assert_eq!(read_all::<i32>(&file, "/grow"), [10, 20], "{case}");
    let names = read_strings(&file, "/g/names");
    let lens: Vec<usize> = names.iter().map(String::len).collect();
    assert!(
        lens == [1, 0, 3] || lens == [1, 40_000, 3],
        "{case} {lens:?}"
    );
}

/// Check that `bytes` read as the file after the edit session.
fn check_edited(bytes: Vec<u8>, case: &str) {
    let file = Hdf5File::from_source(Arc::new(MemorySource::new(bytes))).expect(case);
    assert_eq!(read_all::<i32>(&file, "/fixed"), [1, 20, 30, 4], "{case}");
    let grow = read_all::<i32>(&file, "/grow");
    assert_eq!(grow, [10, 21, 30, 40, 50], "{case}");
    assert_eq!(read_all::<i32>(&file, "/g/new"), [7, 8, 9, 10], "{case}");
    let names = read_strings(&file, "/g/names");
    assert_eq!(names, ["a", &long_strings()[0], "ccc"], "{case}");
    let fixed = file.dataset("/fixed").unwrap();
    assert_eq!(fixed.attribute("units").unwrap().data, b"km", "{case}");
    let new = file.dataset("/g/new").unwrap();
    let factor = format!("{:?}", 0.5f64.to_le_bytes());
    let want = vec![
        ("factor".to_string(), factor),
        ("DIMENSION_LIST".to_string(), "[1]".to_string()),
    ];
    assert_eq!(values_of(&file, &new.attributes), want, "{case}");
    let scale = file.dataset("/scale").unwrap();
    let list = scale.attribute("REFERENCE_LIST").unwrap();
    assert_eq!(list.dataspace.dims, [2], "{case}");
    let g = file.root().group("g").unwrap();
    let want = vec![("tags".to_string(), "[40000, 70000]".to_string())];
    assert_eq!(values_of(&file, &g.attributes), want, "{case}");
}

#[test]
fn a_sink_failure_at_each_call_of_an_edit_leaves_a_correct_file_or_an_error() {
    let first = first_file();
    let (shared, failed, closed) = edit_session(&first, usize::MAX);
    assert_eq!((failed, closed), (0, true));
    let (calls, bytes) = {
        let shared = shared.lock().unwrap();
        (shared.calls, shared.bytes.clone())
    };
    check_edited(bytes, "no failure");

    let (mut retried, mut lost) = (0, 0);
    for fail_at in 0..calls {
        let case = format!("failure at call {fail_at}");
        let (shared, failed, closed) = edit_session(&first, fail_at);
        retried += failed;
        let shared = shared.lock().unwrap();
        if closed {
            check_edited(shared.bytes.clone(), &case);
        } else {
            lost += 1;
            if !shared.superblock {
                check_earlier(shared.bytes.clone(), &case);
            }
        }
    }
    println!("{calls} calls, {retried} retried, {lost} lost");
    // Writes of numbers, chunks, strings and heap collections each fail in some run.
    assert!(retried >= 6, "{retried} of {calls}");
    assert!(lost >= 3, "{lost} of {calls}");
}
