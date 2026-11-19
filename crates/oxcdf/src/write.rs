//! The netCDF-4 writer. It makes a new file with no netcdf-c.
//!
//! The interface follows the write half of the `netcdf` crate.
//!
//! ```no_run
//! let mut file = oxcdf::create("out.nc")?;
//! file.add_unlimited_dimension("time")?;
//! file.add_dimension("lat", 180)?;
//! file.add_attribute("title", "Sea surface temperature")?;
//!
//! let mut temp = file.add_variable::<f32>("TEMP", &["time", "lat"])?;
//! temp.set_chunking(&[1, 180])?;
//! temp.set_compression(4, true)?;
//! temp.set_fill_value(-999.0f32)?;
//! temp.put_attribute("units", "degC")?;
//! temp.put_values(&vec![12.5f32; 180], [0..1, 0..180])?;
//! file.close()?;
//! # Ok::<(), oxcdf::Error>(())
//! ```
//!
//! # Rules
//!
//! * Define a dimension before a variable that uses it.
//! * Set the chunk shape, the filters, the fill value and the byte order of
//!   a variable before its first write. A later change is
//!   [`Error::BadRequest`].
//! * A write past the length of an unlimited dimension grows the variable
//!   and the dimension. Other variables on that dimension read as their fill
//!   value there.
//! * A write of another numeric type converts each value as an `as` cast
//!   does.
//! * A write that fails with [`Error::BadRequest`] changes nothing.
//! * [`create`] writes to `name.tmp`. [`FileMut::close`] moves that file to
//!   `name`. A [`FileMut`] that is dropped does the same and ignores errors.
//!   A drop in a panic removes `name.tmp` and puts no file at `name`.
//!
//! # Limits
//!
//! [`append`] changes a file in place. The writer does not remove or rename
//! an object. It has no user-defined types. One
//! attribute holds 64 KiB at most. See [`oxcdf_hdf5::write`] for the limits
//! of the container.

use std::path::{Path, PathBuf};

use oxcdf_hdf5::message::{
    Attribute, ByteOrder, CharSet, Dataspace, DataspaceKind, Datatype, DatatypeClass, StringPad,
};
use oxcdf_hdf5::read::Hyperslab;
use oxcdf_hdf5::write::{
    numeric_datatype, vlen_string_datatype, ByteSink, DatasetId, DatasetSpec, FileSink, Filter,
    GroupId, Hdf5Writer, MemorySink, ObjectId, WriteOptions,
};

use std::sync::Arc;

use oxcdf_hdf5::source::ByteSource;
use oxcdf_hdf5::Hdf5File;

use crate::extent::Extents;
use crate::netcdf::{
    AttributeValue, NON_COORDINATE_PREFIX, PHANTOM_DIMENSION_PREFIX, RESERVED_ATTRIBUTES,
};
use crate::netcdf::{NcGroup, NetcdfFile};
use crate::{DType, Element, Error, Result};

/// Index of the root group in the group list.
const ROOT: usize = 0;

/// Longest name of a dimension, a variable or a group in bytes, as in netcdf-c.
const MAX_NAME_BYTES: usize = 256;

/// Target size of one default chunk in bytes, as in netcdf-c.
const DEFAULT_CHUNK_BYTES: u64 = 4 << 20;

/// Size in bytes of the default chunk of a variable with one unlimited axis only.
const DEFAULT_RECORD_CHUNK_BYTES: u64 = 4096;

/// Chunk length of the dataset of an unlimited dimension with no variable.
const SCALE_CHUNK: u64 = 1024;

/// Size of one part of an upload to an object store.
///
/// S3 needs 5 MiB at least for each part but the last one.
#[cfg(feature = "object-store")]
const UPLOAD_PART_BYTES: usize = 8 << 20;

/// Most parts of one upload that go to the store at the same time.
///
/// Each part in flight from a file on disk holds [`UPLOAD_PART_BYTES`] of memory.
#[cfg(feature = "object-store")]
const UPLOAD_PARTS_IN_FLIGHT: usize = 4;

/// Create the netCDF-4 file `path`.
///
/// The writer uses `path` with `.tmp` at its end until the close. An
/// existing file at `path` is replaced at the close.
///
/// # Errors
///
/// Returns [`Error::Io`] if the temporary file cannot be created.
pub fn create(path: impl AsRef<Path>) -> Result<FileMut> {
    create_with(path, WriteOptions::default())
}

/// Create the netCDF-4 file `path` with explicit options.
///
/// # Errors
///
/// Returns [`Error::BadRequest`] for options above their limits. Returns
/// [`Error::Io`] if the temporary file cannot be created. An error leaves no
/// file.
pub fn create_with(path: impl AsRef<Path>, options: WriteOptions) -> Result<FileMut> {
    let path = path.as_ref().to_path_buf();
    let mut name = path.clone().into_os_string();
    name.push(".tmp");
    let temporary = PathBuf::from(name);
    let sink = Sink::File(FileSink::create(&temporary)?);
    let target = Target::File {
        temporary: temporary.clone(),
        path,
    };
    FileMut::new(sink, options, target).inspect_err(|_| {
        // The first error is the one to report.
        let _ = std::fs::remove_file(&temporary);
    })
}

/// Create a netCDF-4 file in memory. [`FileMut::into_bytes`] returns it.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// let mut file = oxcdf::create_in_memory()?;
/// file.add_dimension("x", 3)?;
/// let mut v = file.add_variable::<i32>("v", &["x"])?;
/// v.put_values(&[1, 2, 3], ..)?;
/// let bytes = file.into_bytes()?;
///
/// let file = oxcdf::NetcdfFile::from_source(Arc::new(oxcdf::MemorySource::new(bytes)))?;
/// let v = file.variable("v").unwrap();
/// assert_eq!(v.get_values::<i32, _>(..)?, [1, 2, 3]);
/// # Ok::<(), oxcdf::Error>(())
/// ```
///
/// # Errors
///
/// None with the default options. The signature matches
/// [`create_in_memory_with`].
pub fn create_in_memory() -> Result<FileMut> {
    create_in_memory_with(WriteOptions::default())
}

/// Create a netCDF-4 file in memory with explicit options.
///
/// # Errors
///
/// Returns [`Error::BadRequest`] for options above their limits.
pub fn create_in_memory_with(options: WriteOptions) -> Result<FileMut> {
    FileMut::new(
        Sink::Memory(MemoryBuffer::default()),
        options,
        Target::Memory,
    )
}

/// Open the netCDF-4 file `path` to change it in place.
///
/// The file can come from this writer or from netcdf-c. [`FileMut::close`]
/// writes the changes. A [`FileMut`] that is dropped does the same and
/// ignores errors.
///
/// The writer is the only writer of the file until the close. A second
/// [`append`] of the file fails. On Windows, the handle of the writer
/// shares the file for reads only. On Unix, the writer takes an exclusive
/// advisory lock (`flock`) on the file. The OS frees the file when the
/// process stops, so a crash leaves no lock.
///
/// A reader such as [`crate::open`] can open the file during the session.
/// Until the close, it sees the earlier structure and the earlier values.
///
/// Each variable of the file keeps its definition. A write to values of
/// the file changes them at once. New groups, dimensions, variables and
/// attributes appear at the close. Each append session adds the size of
/// all object headers to the file.
///
/// # Errors
///
/// Returns [`Error::Io`] when the file cannot be opened, or when another
/// handle has it open for writes or holds a lock on it. Returns
/// [`Error::Unsupported`] for a classic file and for a file that the writer
/// cannot describe again, such as a file with user-defined types or with no
/// dimension scales. An error changes nothing.
///
/// [`FileMut::add_variable`] in a session returns [`Error::Unsupported`]
/// for a new coordinate variable of a dimension of the file.
pub fn append(path: impl AsRef<Path>) -> Result<FileMut> {
    append_with(path, WriteOptions::default())
}

/// Open the netCDF-4 file `path` to change it in place, with explicit options.
///
/// # Errors
///
/// The errors of [`append`]. Returns [`Error::BadRequest`] for options above
/// their limits.
pub fn append_with(path: impl AsRef<Path>, options: WriteOptions) -> Result<FileMut> {
    let sink = FileSink::open(path)?;
    let file = NetcdfFile::from_source(sink.source()?)?;
    FileMut::open(Sink::File(sink), &file, options, Target::InPlace)
}

/// Open the netCDF-4 file `bytes` in memory to change it.
///
/// [`FileMut::into_bytes`] returns the changed file.
///
/// # Errors
///
/// The errors of [`append`], apart from [`Error::Io`].
pub fn append_in_memory(bytes: Vec<u8>) -> Result<FileMut> {
    append_shared(Arc::new(bytes))
}

/// Open the file `bytes` in memory. The reader and the sink share the one buffer.
fn append_shared(bytes: Arc<Vec<u8>>) -> Result<FileMut> {
    let source = Arc::new(SharedSource(Arc::clone(&bytes)));
    let file = NetcdfFile::from_source(source)?;
    let sink = Sink::Memory(MemoryBuffer::shared(bytes));
    FileMut::open(sink, &file, WriteOptions::default(), Target::Memory)
}

/// A byte source over a buffer that a [`MemoryBuffer`] also holds.
#[derive(Debug)]
struct SharedSource(Arc<Vec<u8>>);

impl ByteSource for SharedSource {
    fn size(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let range = usize::try_from(offset)
            .ok()
            .and_then(|start| Some(start..start.checked_add(buf.len())?));
        match range.and_then(|range| self.0.get(range)) {
            Some(bytes) => buf.copy_from_slice(bytes),
            None => {
                return Err(Error::OutOfBounds {
                    what: "buffer",
                    offset,
                    len: buf.len() as u64,
                    available: self.size().saturating_sub(offset),
                })
            }
        }
        Ok(())
    }
}

/// A memory sink that can start from a buffer that a reader shares.
#[derive(Debug, Default)]
struct MemoryBuffer {
    /// The bytes of the file until the first access to the sink.
    shared: Option<Arc<Vec<u8>>>,
    sink: MemorySink,
}

impl MemoryBuffer {
    /// A sink with the bytes of `bytes`.
    fn shared(bytes: Arc<Vec<u8>>) -> Self {
        Self {
            shared: Some(bytes),
            sink: MemorySink::new(),
        }
    }

    /// The sink, with the shared bytes moved in at the first call.
    fn sink(&mut self) -> &mut MemorySink {
        if let Some(bytes) = self.shared.take() {
            // With no reader left, the move takes the buffer with no copy.
            let bytes = Arc::try_unwrap(bytes).unwrap_or_else(|bytes| bytes.as_ref().clone());
            self.sink = MemorySink::from_bytes(bytes);
        }
        &mut self.sink
    }

    /// Take the bytes.
    fn into_inner(mut self) -> Vec<u8> {
        self.sink();
        self.sink.into_inner()
    }
}

impl ByteSink for MemoryBuffer {
    fn write_all_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        self.sink().write_all_at(offset, data)
    }

    fn set_len(&mut self, len: u64) -> Result<()> {
        self.sink().set_len(len)
    }

    fn sync(&mut self) -> Result<()> {
        self.sink().sync()
    }

    fn can_read(&self) -> bool {
        self.sink.can_read()
    }

    fn read_exact_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.sink().read_exact_at(offset, buf)
    }
}

/// The place that holds the bytes of the file.
#[derive(Debug)]
enum Sink {
    File(FileSink),
    Memory(MemoryBuffer),
}

impl ByteSink for Sink {
    fn write_all_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        match self {
            Self::File(sink) => sink.write_all_at(offset, data),
            Self::Memory(sink) => sink.write_all_at(offset, data),
        }
    }

    fn set_len(&mut self, len: u64) -> Result<()> {
        match self {
            Self::File(sink) => sink.set_len(len),
            Self::Memory(sink) => sink.set_len(len),
        }
    }

    fn sync(&mut self) -> Result<()> {
        match self {
            Self::File(sink) => sink.sync(),
            Self::Memory(sink) => sink.sync(),
        }
    }

    fn can_read(&self) -> bool {
        match self {
            Self::File(sink) => sink.can_read(),
            Self::Memory(sink) => sink.can_read(),
        }
    }

    fn read_exact_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        match self {
            Self::File(sink) => sink.read_exact_at(offset, buf),
            Self::Memory(sink) => sink.read_exact_at(offset, buf),
        }
    }
}

/// Where the file goes at the close.
#[derive(Debug)]
enum Target {
    Memory,
    File {
        temporary: PathBuf,
        path: PathBuf,
    },
    /// A file from [`append`]. The sink is the file at its path.
    InPlace,
}

/// One group of the file.
#[derive(Debug)]
struct Group {
    id: GroupId,
    parent: Option<usize>,
    /// Indices of the dimensions that this group defines.
    dimensions: Vec<usize>,
    /// Indices of the variables of this group.
    variables: Vec<usize>,
    /// Name and index of each child group.
    groups: Vec<(String, usize)>,
}

/// One dimension. Its index in the list is its netCDF dimension id.
#[derive(Debug)]
struct Dimension {
    name: String,
    group: usize,
    len: u64,
    unlimited: bool,
    /// The variable of the same name that has this dimension as its first axis.
    coordinate: Option<usize>,
    /// The dataset of a dimension of the file with no variable, and its length.
    scale: Option<(DatasetId, u64)>,
}

/// The part of a variable that can change before its first write.
#[derive(Debug, Clone)]
struct Definition {
    dtype: DType,
    order: ByteOrder,
    chunks: Option<Vec<u64>>,
    shuffle: bool,
    deflate: Option<u8>,
    zstd: Option<u8>,
    fletcher32: bool,
    /// The fill value from the caller, as one element in native byte order.
    fill: Option<Vec<u8>>,
}

/// One variable.
#[derive(Debug)]
struct Variable {
    name: String,
    dataset: DatasetId,
    definition: Definition,
    /// The dimension index of each axis.
    dimensions: Vec<usize>,
    /// The shape of the HDF5 dataset.
    shape: Vec<u64>,
    /// True after the first write.
    written: bool,
    /// True for a variable of the file, which has its scales.
    attached: bool,
}

/// The model of the file and the HDF5 writer below it.
#[derive(Debug)]
struct State {
    writer: Hdf5Writer<Sink>,
    groups: Vec<Group>,
    dimensions: Vec<Dimension>,
    variables: Vec<Variable>,
}

/// A netCDF-4 file that is open for writes.
///
/// [`create`] and [`create_in_memory`] make one. The methods of the file
/// act on the root group.
#[derive(Debug)]
pub struct FileMut {
    /// `None` after the close.
    state: Option<State>,
    target: Target,
}

/// A group of a [`FileMut`].
#[derive(Debug)]
pub struct GroupMut<'f> {
    state: &'f mut State,
    group: usize,
}

/// A variable of a [`FileMut`].
#[derive(Debug)]
pub struct VariableMut<'f> {
    state: &'f mut State,
    variable: usize,
}

impl FileMut {
    fn new(sink: Sink, options: WriteOptions, target: Target) -> Result<Self> {
        let mut writer = Hdf5Writer::create(sink, options)?;
        let root = writer.root();
        let properties = format!("version=2,oxcdf={}", env!("CARGO_PKG_VERSION"));
        writer.put_text_attribute(root, "_NCProperties", &properties)?;
        let root = Group {
            id: root,
            parent: None,
            dimensions: Vec::new(),
            variables: Vec::new(),
            groups: Vec::new(),
        };
        let state = State {
            writer,
            groups: vec![root],
            dimensions: Vec::new(),
            variables: Vec::new(),
        };
        Ok(Self {
            state: Some(state),
            target,
        })
    }

    /// Open the file in `sink` for changes. `file` is the view of that file.
    fn open(sink: Sink, file: &NetcdfFile, options: WriteOptions, target: Target) -> Result<Self> {
        let hdf5 = file
            .hdf5()
            .ok_or_else(|| Error::unsupported("an append to a netCDF classic file"))?;
        let writer = Hdf5Writer::open(sink, hdf5, options)?;
        let state = State::from_file(writer, file, hdf5)?;
        Ok(Self {
            state: Some(state),
            target,
        })
    }

    fn state(&mut self) -> Result<&mut State> {
        self.state.as_mut().ok_or_else(state_defect)
    }

    /// Add a dimension of `len` elements to the root group.
    ///
    /// A length of 0 gives an unlimited dimension, as in netcdf-c.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] for a wrong name, a name that the group
    /// uses, or the name of a variable of the group.
    pub fn add_dimension(&mut self, name: &str, len: usize) -> Result<()> {
        self.state()?
            .add_dimension(ROOT, name, len as u64, len == 0)
    }

    /// Add an unlimited dimension to the root group.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::add_dimension`].
    pub fn add_unlimited_dimension(&mut self, name: &str) -> Result<()> {
        self.state()?.add_dimension(ROOT, name, 0, true)
    }

    /// The current length of dimension `name` of the root group.
    pub fn dimension_len(&self, name: &str) -> Option<u64> {
        let state = self.state.as_ref()?;
        let dimension = state.find_dimension(ROOT, name)?;
        Some(state.dimensions[dimension].len)
    }

    /// Put an attribute on the root group.
    ///
    /// A second put with the same name replaces the value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] for a wrong name or a name that netCDF
    /// reserves. Returns [`Error::Unsupported`] for [`AttributeValue::Raw`].
    pub fn add_attribute<T: Into<AttributeValue>>(&mut self, name: &str, value: T) -> Result<()> {
        let state = self.state()?;
        let object = state.groups[ROOT].id.into();
        put_attribute(&mut state.writer, object, name, value.into())
    }

    /// Add a variable of type `T` on the dimensions `dimensions` to the root group.
    ///
    /// An empty list gives a scalar variable.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] for a wrong name, a name that the group
    /// uses, or an unknown dimension. Returns [`Error::Unsupported`] for a
    /// new coordinate variable of a dimension of a file from [`append`]. That
    /// dimension has no coordinate variable in the file. This error changes
    /// nothing.
    pub fn add_variable<T: Element>(
        &mut self,
        name: &str,
        dimensions: &[&str],
    ) -> Result<VariableMut<'_>> {
        self.add_variable_with_type(name, dimensions, &T::DTYPE)
    }

    /// Add a variable of the netCDF type `dtype` to the root group.
    ///
    /// Use this for [`DType::Char`].
    ///
    /// # Errors
    ///
    /// The errors of [`Self::add_variable`]. Returns [`Error::Unsupported`]
    /// for a type that is not a netCDF atomic type.
    pub fn add_variable_with_type(
        &mut self,
        name: &str,
        dimensions: &[&str],
        dtype: &DType,
    ) -> Result<VariableMut<'_>> {
        let state = self.state()?;
        let variable = state.add_variable(ROOT, name, dimensions, dtype)?;
        Ok(VariableMut { state, variable })
    }

    /// Add a variable of netCDF strings to the root group.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::add_variable`].
    pub fn add_string_variable(
        &mut self,
        name: &str,
        dimensions: &[&str],
    ) -> Result<VariableMut<'_>> {
        self.add_variable_with_type(name, dimensions, &DType::String)
    }

    /// Add a group to the root group.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] for a wrong name or a name that the
    /// group uses.
    pub fn add_group(&mut self, name: &str) -> Result<GroupMut<'_>> {
        let state = self.state()?;
        let group = state.add_group(ROOT, name)?;
        Ok(GroupMut { state, group })
    }

    /// The variable `name` of the root group.
    pub fn variable_mut(&mut self, name: &str) -> Option<VariableMut<'_>> {
        let state = self.state.as_mut()?;
        let variable = state.find_variable(ROOT, name)?;
        Some(VariableMut { state, variable })
    }

    /// The group `name` of the root group.
    pub fn group_mut(&mut self, name: &str) -> Option<GroupMut<'_>> {
        let state = self.state.as_mut()?;
        let group = state.find_group(ROOT, name)?;
        Some(GroupMut { state, group })
    }

    /// Write the file and move it to its path.
    ///
    /// A file in memory is lost: use [`Self::into_bytes`] for it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if a write or the move fails. A close can also
    /// return an error of the HDF5 writer for a definition that it cannot
    /// encode. After an error, no file is at the path of [`create`] from this
    /// writer.
    pub fn close(mut self) -> Result<()> {
        self.finish_file()
    }

    /// Write a file in memory and return its bytes.
    ///
    /// A file from [`append`] gets the in-place close of [`Self::close`] and
    /// then an error, so the session keeps its changes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] for a file from [`create`], and the
    /// writer removes it. Returns [`Error::BadRequest`] for a file from
    /// [`append`] after the writer writes it in place. When that write
    /// fails, returns its error. The other errors are those of
    /// [`Self::close`].
    pub fn into_bytes(mut self) -> Result<Vec<u8>> {
        match self.target {
            Target::Memory => {}
            Target::InPlace => {
                self.finish_file()?;
                return Err(in_place_close("read the file"));
            }
            Target::File { .. } => {
                self.discard();
                return Err(Error::bad_request(
                    "the file is on disk: use close, then read the file",
                ));
            }
        }
        let state = self.state.take().ok_or_else(state_defect)?;
        match state.finish()? {
            Sink::Memory(sink) => Ok(sink.into_inner()),
            Sink::File(_) => Err(sink_defect()),
        }
    }

    /// Write the file and upload it to `location` in `store`.
    ///
    /// A file in memory goes up from memory. A file from [`create`] goes up
    /// from its temporary file in parts, and the writer removes that file.
    /// No file appears at the path of [`create`]. The upload is atomic: a
    /// reader sees the earlier object or the full new object.
    ///
    /// The close before the upload blocks the thread, as [`Self::close`]
    /// does. The read of each part of a temporary file also blocks it.
    ///
    /// A file from [`append`] gets the in-place close of [`Self::close`] and
    /// then an error, with no upload. The session keeps its changes.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::close`]. Returns [`Error::Io`] if the upload or
    /// the removal of the temporary file fails. A failed upload leaves the
    /// earlier object at `location`. Returns [`Error::BadRequest`] for a
    /// file from [`append`] after the writer writes it in place. When that
    /// write fails, returns its error.
    #[cfg(feature = "object-store")]
    #[cfg_attr(docsrs, doc(cfg(feature = "object-store")))]
    pub async fn close_to_store(
        mut self,
        store: &dyn object_store::ObjectStore,
        location: &object_store::path::Path,
    ) -> Result<()> {
        use object_store::ObjectStoreExt;
        if matches!(self.target, Target::InPlace) {
            self.finish_file()?;
            return Err(in_place_close("upload the file"));
        }
        let state = self.state.take().ok_or_else(state_defect)?;
        let finished = state.finish();
        let Target::File { temporary, .. } = &self.target else {
            let Sink::Memory(sink) = finished? else {
                return Err(sink_defect());
            };
            let bytes = bytes::Bytes::from(sink.into_inner());
            if bytes.len() <= UPLOAD_PART_BYTES {
                store
                    .put(location, bytes.into())
                    .await
                    .map_err(upload_error)?;
                return Ok(());
            }
            let upload = store.put_multipart(location).await.map_err(upload_error)?;
            // Each part is a view of the one buffer, with no copy.
            let parts = (0..bytes.len()).step_by(UPLOAD_PART_BYTES).map(|start| {
                let end = bytes.len().min(start + UPLOAD_PART_BYTES);
                Ok(bytes.slice(start..end))
            });
            return upload_parts(upload, parts).await;
        };
        let result = match finished {
            Ok(sink) => {
                // Windows keeps a file that is open for writes locked.
                drop(sink);
                upload_file(store, location, temporary).await
            }
            Err(e) => Err(e),
        };
        let removed = std::fs::remove_file(temporary);
        result?;
        Ok(removed?)
    }

    /// Drop the file and remove its temporary file, if it has one.
    fn discard(&mut self) {
        // Windows cannot remove a file that is open.
        drop(self.state.take());
        if let Target::File { temporary, .. } = &self.target {
            // The caller reports its own error, or has no place for one.
            let _ = std::fs::remove_file(temporary);
        }
    }

    /// Write a file on disk and move it to its path.
    fn finish_file(&mut self) -> Result<()> {
        let Some(state) = self.state.take() else {
            return Ok(());
        };
        let finished = state.finish();
        let Target::File { temporary, path } = &self.target else {
            return finished.map(drop);
        };
        let moved = finished.and_then(|sink| {
            // Windows cannot move a file that is open.
            drop(sink);
            Ok(std::fs::rename(temporary, path)?)
        });
        if moved.is_err() {
            // The first error is the one to report.
            let _ = std::fs::remove_file(temporary);
        }
        moved
    }
}

impl Drop for FileMut {
    fn drop(&mut self) {
        if std::thread::panicking() {
            // The panic can stop the writes part way, so no file goes to its path.
            self.discard();
            return;
        }
        // A drop has no place to report an error. Use `close` to get it.
        let _ = self.finish_file();
    }
}

/// Upload the file `path` to `location` in parts.
#[cfg(feature = "object-store")]
async fn upload_file(
    store: &dyn object_store::ObjectStore,
    location: &object_store::path::Path,
    path: &Path,
) -> Result<()> {
    use object_store::ObjectStoreExt;
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let upload = store.put_multipart(location).await.map_err(upload_error)?;
    let parts = std::iter::from_fn(move || {
        let mut part = Vec::with_capacity(UPLOAD_PART_BYTES);
        let limit = UPLOAD_PART_BYTES as u64;
        match file.by_ref().take(limit).read_to_end(&mut part) {
            Ok(0) => None,
            Ok(_) => Some(Ok(part.into())),
            Err(e) => Some(Err(Error::from(e))),
        }
    });
    upload_parts(upload, parts).await
}

/// Send each part to `upload` and complete it. An error stops the upload.
#[cfg(feature = "object-store")]
async fn upload_parts(
    upload: Box<dyn object_store::MultipartUpload>,
    parts: impl Iterator<Item = Result<bytes::Bytes>>,
) -> Result<()> {
    let mut upload = object_store::WriteMultipart::new_with_chunk_size(upload, UPLOAD_PART_BYTES);
    for part in parts {
        let sent = match part {
            Ok(part) => {
                upload.put(part);
                upload
                    .wait_for_capacity(UPLOAD_PARTS_IN_FLIGHT)
                    .await
                    .map_err(upload_error)
            }
            Err(e) => Err(e),
        };
        if let Err(e) = sent {
            // The first error is the one to report.
            let _ = upload.abort().await;
            return Err(e);
        }
    }
    upload.finish().await.map_err(upload_error)?;
    Ok(())
}

/// The error for a failed upload.
#[cfg(feature = "object-store")]
fn upload_error(error: object_store::Error) -> Error {
    Error::Io(std::io::Error::other(error))
}

impl GroupMut<'_> {
    /// Add a dimension of `len` elements to this group.
    ///
    /// A length of 0 gives an unlimited dimension, as in netcdf-c.
    ///
    /// # Errors
    ///
    /// The errors of [`FileMut::add_dimension`].
    pub fn add_dimension(&mut self, name: &str, len: usize) -> Result<()> {
        self.state
            .add_dimension(self.group, name, len as u64, len == 0)
    }

    /// Add an unlimited dimension to this group.
    ///
    /// # Errors
    ///
    /// The errors of [`FileMut::add_dimension`].
    pub fn add_unlimited_dimension(&mut self, name: &str) -> Result<()> {
        self.state.add_dimension(self.group, name, 0, true)
    }

    /// Put an attribute on this group.
    ///
    /// # Errors
    ///
    /// The errors of [`FileMut::add_attribute`].
    pub fn add_attribute<T: Into<AttributeValue>>(&mut self, name: &str, value: T) -> Result<()> {
        let object = self.state.groups[self.group].id.into();
        put_attribute(&mut self.state.writer, object, name, value.into())
    }

    /// Add a variable of type `T` to this group.
    ///
    /// A dimension name can be one of this group or of a parent group.
    ///
    /// # Errors
    ///
    /// The errors of [`FileMut::add_variable`].
    pub fn add_variable<T: Element>(
        &mut self,
        name: &str,
        dimensions: &[&str],
    ) -> Result<VariableMut<'_>> {
        self.add_variable_with_type(name, dimensions, &T::DTYPE)
    }

    /// Add a variable of the netCDF type `dtype` to this group.
    ///
    /// # Errors
    ///
    /// The errors of [`FileMut::add_variable_with_type`].
    pub fn add_variable_with_type(
        &mut self,
        name: &str,
        dimensions: &[&str],
        dtype: &DType,
    ) -> Result<VariableMut<'_>> {
        let variable = self
            .state
            .add_variable(self.group, name, dimensions, dtype)?;
        Ok(VariableMut {
            state: self.state,
            variable,
        })
    }

    /// Add a variable of netCDF strings to this group.
    ///
    /// # Errors
    ///
    /// The errors of [`FileMut::add_variable`].
    pub fn add_string_variable(
        &mut self,
        name: &str,
        dimensions: &[&str],
    ) -> Result<VariableMut<'_>> {
        self.add_variable_with_type(name, dimensions, &DType::String)
    }

    /// Add a group to this group.
    ///
    /// # Errors
    ///
    /// The errors of [`FileMut::add_group`].
    pub fn add_group(&mut self, name: &str) -> Result<GroupMut<'_>> {
        let group = self.state.add_group(self.group, name)?;
        Ok(GroupMut {
            state: self.state,
            group,
        })
    }

    /// The variable `name` of this group.
    pub fn variable_mut(&mut self, name: &str) -> Option<VariableMut<'_>> {
        let variable = self.state.find_variable(self.group, name)?;
        Some(VariableMut {
            state: self.state,
            variable,
        })
    }

    /// The group `name` of this group.
    pub fn group_mut(&mut self, name: &str) -> Option<GroupMut<'_>> {
        let group = self.state.find_group(self.group, name)?;
        Some(GroupMut {
            state: self.state,
            group,
        })
    }
}

impl VariableMut<'_> {
    fn variable(&self) -> &Variable {
        &self.state.variables[self.variable]
    }

    /// Set the chunk shape. The variable then has chunked storage.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] after the first write, for a scalar
    /// variable, or for a shape that does not fit the dimensions.
    pub fn set_chunking(&mut self, chunk_shape: &[usize]) -> Result<()> {
        let chunks = chunk_shape.iter().map(|&n| n as u64).collect();
        self.redefine(|d| d.chunks = Some(chunks))
    }

    /// Set the deflate level, from 0 to 9, and the shuffle filter.
    ///
    /// A variable with no chunk shape gets the default chunk shape.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] after the first write, for a scalar
    /// variable, for a level outside its range, or for a variable with
    /// Zstandard.
    pub fn set_compression(&mut self, deflate_level: i32, shuffle: bool) -> Result<()> {
        let level = u8::try_from(deflate_level).map_err(|_| {
            Error::bad_request(format!(
                "a deflate level of {deflate_level} is not from 0 to 9"
            ))
        })?;
        self.redefine(|d| {
            d.deflate = Some(level);
            d.shuffle = shuffle;
        })
    }

    /// Add or remove the shuffle filter of each chunk.
    ///
    /// The filter helps a compression filter that comes after it. Use this
    /// for shuffle with Zstandard, or for shuffle with no compression.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] after the first write or for a scalar
    /// variable.
    pub fn set_shuffle(&mut self, on: bool) -> Result<()> {
        self.redefine(|d| d.shuffle = on)
    }

    /// Set the Zstandard level, from 1 to 22. A reader needs the Zstandard plugin.
    ///
    /// A variable takes deflate or Zstandard, not the two.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] after the first write, for a scalar
    /// variable, for a level outside its range, or for a variable with
    /// deflate.
    pub fn set_zstd(&mut self, level: i32) -> Result<()> {
        let level = u8::try_from(level).map_err(|_| {
            Error::bad_request(format!("a Zstandard level of {level} is not from 1 to 22"))
        })?;
        self.redefine(|d| d.zstd = Some(level))
    }

    /// Add or remove the Fletcher-32 checksum of each chunk.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] after the first write, or for a scalar
    /// variable with the checksum.
    pub fn set_fletcher32(&mut self, on: bool) -> Result<()> {
        self.redefine(|d| d.fletcher32 = on)
    }

    /// Set the byte order of the stored values. The default is little-endian.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] after the first write.
    pub fn set_endianness(&mut self, order: ByteOrder) -> Result<()> {
        self.redefine(|d| d.order = order)
    }

    /// Set the fill value and the `_FillValue` attribute.
    ///
    /// The value converts to the type of the variable. A variable with no
    /// call gets the default fill value of netCDF.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] after the first write. Returns
    /// [`Error::Unsupported`] for a string variable.
    pub fn set_fill_value<T: Element>(&mut self, fill_value: T) -> Result<()> {
        let dtype = &self.variable().definition.dtype;
        if *dtype == DType::String {
            return Err(Error::unsupported("a fill value for a string variable"));
        }
        let fill = converted(&[fill_value], dtype)?;
        self.redefine(|d| d.fill = Some(fill))
    }

    /// Change the definition of a variable that has no write.
    fn redefine(&mut self, change: impl FnOnce(&mut Definition)) -> Result<()> {
        let variable = self.variable();
        if variable.written {
            return Err(Error::bad_request(format!(
                "variable {} has values: its definition cannot change",
                variable.name
            )));
        }
        let mut definition = variable.definition.clone();
        change(&mut definition);
        let spec = self.state.spec(&definition, &variable.dimensions)?;
        let dataset = variable.dataset;
        self.state.writer.redefine_dataset(dataset, spec)?;
        if let Some(fill) = &definition.fill {
            let attribute = Attribute {
                name: "_FillValue".to_owned(),
                datatype: datatype_of(&definition.dtype, definition.order)?,
                // netcdf-c writes the fill value of a `char` variable as one scalar text.
                dataspace: match definition.dtype {
                    DType::Char => scalar(),
                    _ => Dataspace {
                        kind: DataspaceKind::Simple,
                        dims: vec![1],
                        max_dims: None,
                    },
                },
                data: in_order(fill.clone(), definition.order),
            };
            self.state.writer.put_attribute(dataset, attribute)?;
        }
        self.state.variables[self.variable].definition = definition;
        Ok(())
    }

    /// Put an attribute on this variable.
    ///
    /// A second put with the same name replaces the value. Use
    /// [`Self::set_fill_value`] for `_FillValue`.
    ///
    /// # Errors
    ///
    /// The errors of [`FileMut::add_attribute`]. `_FillValue` is a name that
    /// netCDF reserves.
    pub fn put_attribute<T: Into<AttributeValue>>(&mut self, name: &str, value: T) -> Result<()> {
        if name == "_FillValue" {
            return Err(Error::bad_request(
                "set_fill_value makes the attribute _FillValue",
            ));
        }
        let object = self.variable().dataset.into();
        put_attribute(&mut self.state.writer, object, name, value.into())
    }

    /// Write `values` to the elements that `extents` selects.
    ///
    /// The extents have the forms of [`crate::Variable::get_values`]. On an
    /// unlimited axis, a selection can end past the current length. A full
    /// selection takes the current length of each dimension.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] for a string variable, a wrong number of
    /// values, or a selection outside the variable. Returns
    /// [`Error::Unsupported`] for a stride other than 1. Returns [`Error::Io`]
    /// if the write to the file fails.
    pub fn put_values<T: Element, E>(&mut self, values: &[T], extents: E) -> Result<()>
    where
        E: TryInto<Extents>,
        E::Error: Into<Error>,
    {
        let dtype = self.variable().definition.dtype.clone();
        if dtype == DType::String {
            return Err(Error::bad_request(format!(
                "variable {} holds strings: use put_strings",
                self.variable().name
            )));
        }
        let slab = self.prepare(extents, values.len())?;
        let dataset = self.variable().dataset;
        if dtype == T::DTYPE {
            self.state.writer.write(dataset, &slab, values)?;
        } else {
            let order = self.variable().definition.order;
            let bytes = in_order_of(converted(values, &dtype)?, order, &dtype);
            self.state.writer.write_raw(dataset, &slab, &bytes)?;
        }
        self.state.variables[self.variable].written = true;
        Ok(())
    }

    /// Write one value to the one element that `extents` selects.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::put_values`].
    pub fn put_value<T: Element, E>(&mut self, value: T, extents: E) -> Result<()>
    where
        E: TryInto<Extents>,
        E::Error: Into<Error>,
    {
        self.put_values(&[value], extents)
    }

    /// Write strings to the elements that `extents` selects in a string variable.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] for a variable of another type or a
    /// string with a NUL byte. The other errors are those of
    /// [`Self::put_values`].
    pub fn put_strings<T: AsRef<str>, E>(&mut self, values: &[T], extents: E) -> Result<()>
    where
        E: TryInto<Extents>,
        E::Error: Into<Error>,
    {
        let name = &self.variable().name;
        if self.variable().definition.dtype != DType::String {
            return Err(Error::bad_request(format!(
                "variable {name} holds no strings: use put_values"
            )));
        }
        // The HDF5 writer also rejects a NUL, but only after the variable grows.
        if let Some(at) = values.iter().position(|v| v.as_ref().contains('\0')) {
            return Err(Error::bad_request(format!(
                "string {at} for variable {name} holds a NUL byte"
            )));
        }
        let slab = self.prepare(extents, values.len())?;
        let dataset = self.variable().dataset;
        self.state.writer.write_strings(dataset, &slab, values)?;
        self.state.variables[self.variable].written = true;
        Ok(())
    }

    /// Write one string to the one element that `extents` selects.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::put_strings`].
    pub fn put_string<E>(&mut self, value: &str, extents: E) -> Result<()>
    where
        E: TryInto<Extents>,
        E::Error: Into<Error>,
    {
        self.put_strings(&[value], extents)
    }

    /// Resolve `extents` for `count` values and grow the variable to hold them.
    fn prepare<E>(&mut self, extents: E, count: usize) -> Result<Hyperslab>
    where
        E: TryInto<Extents>,
        E::Error: Into<Error>,
    {
        let extents: Extents = extents.try_into().map_err(Into::into)?;
        let State {
            writer,
            dimensions,
            variables,
            ..
        } = &mut *self.state;
        let variable = &mut variables[self.variable];
        let axes = variable.dimensions.iter().zip(&variable.shape);
        let grows: Vec<bool> = axes
            .clone()
            .map(|(&d, _)| dimensions[d].unlimited)
            .collect();
        // A variable has the length of its dimension on an unlimited axis.
        let shape: Vec<u64> = axes
            .map(|(&d, &len)| match dimensions[d].unlimited {
                true => dimensions[d].len,
                false => len,
            })
            .collect();
        let slab = extents.to_write_hyperslab(&variable.name, &shape, &grows)?;
        if u64::try_from(count).ok() != Some(slab.element_count()) {
            return Err(Error::bad_request(format!(
                "{count} values for a selection of {} elements of variable {}",
                slab.element_count(),
                variable.name
            )));
        }
        if slab.element_count() == 0 {
            return Ok(slab);
        }
        let ends = slab.start.iter().zip(&slab.count).map(|(s, c)| s + c);
        let needed: Vec<u64> = ends.zip(&variable.shape).map(|(e, &l)| e.max(l)).collect();
        if needed != variable.shape {
            // The variable cannot shrink again, so a refused write must not grow it.
            writer.check_write(variable.dataset, &slab)?;
            writer.resize(variable.dataset, &needed)?;
            for (&d, &len) in variable.dimensions.iter().zip(&needed) {
                dimensions[d].len = dimensions[d].len.max(len);
            }
            variable.shape = needed;
        }
        Ok(slab)
    }
}

impl State {
    /// The model of `file`, which `writer` has open.
    fn from_file(writer: Hdf5Writer<Sink>, file: &NetcdfFile, hdf5: &Hdf5File) -> Result<Self> {
        let mut state = Self {
            writer,
            groups: Vec::new(),
            dimensions: Vec::new(),
            variables: Vec::new(),
        };
        let root = state.writer.root();
        // Each dimension is in the model before a variable looks for it.
        let mut dimensions = Vec::new();
        state.import_groups(file.root(), root, None, hdf5, &mut dimensions)?;
        dimensions.sort_by_key(|(id, _)| *id);
        for (position, (id, dimension)) in dimensions.into_iter().enumerate() {
            // The position of a dimension in the model is its id in the file.
            if i64::try_from(position).ok() != Some(id) {
                return Err(Error::unsupported(
                    "an append to a file with dimension ids that are not 0, 1, 2 and so on",
                ));
            }
            state.groups[dimension.group].dimensions.push(position);
            state.dimensions.push(dimension);
        }
        state.import_variables(file.root(), ROOT, hdf5)?;
        Ok(state)
    }

    /// Add `group` and its child groups to the model. The dimensions go to `dimensions` with their ids.
    fn import_groups(
        &mut self,
        group: &NcGroup,
        id: GroupId,
        parent: Option<usize>,
        hdf5: &Hdf5File,
        dimensions: &mut Vec<(i64, Dimension)>,
    ) -> Result<()> {
        let index = self.groups.len();
        self.groups.push(Group {
            id,
            parent,
            dimensions: Vec::new(),
            variables: Vec::new(),
            groups: Vec::new(),
        });
        for dimension in &group.dimensions {
            let (Some(dimension_id), false) = (dimension.id, dimension.is_phony) else {
                return Err(Error::unsupported(format!(
                    "an append to a file with no dimension scale for {}",
                    dimension.name
                )));
            };
            let scale = match dimension.has_coordinate_variable {
                true => None,
                false => {
                    let path = format!("{}/{}", group.path.trim_end_matches('/'), dimension.name);
                    let dataset = self.writer.find_dataset(id, &dimension.name);
                    let len = hdf5.dataset(&path).and_then(|d| d.shape.first().copied());
                    Some(dataset.zip(len).ok_or_else(|| {
                        Error::unsupported(format!("an append to a file with no dataset {path}"))
                    })?)
                }
            };
            let model = Dimension {
                name: dimension.name.clone(),
                group: index,
                len: dimension.len,
                unlimited: dimension.is_unlimited,
                coordinate: None,
                scale,
            };
            dimensions.push((dimension_id, model));
        }
        for child in &group.groups {
            let child_id = self.writer.find_group(id, &child.name).ok_or_else(|| {
                Error::unsupported(format!("an append to a file with no group {}", child.path))
            })?;
            let child_index = self.groups.len();
            self.groups[index]
                .groups
                .push((child.name.clone(), child_index));
            self.import_groups(child, child_id, Some(index), hdf5, dimensions)?;
        }
        Ok(())
    }

    /// Add the variables of `group` and of its child groups to the model.
    fn import_variables(&mut self, group: &NcGroup, index: usize, hdf5: &Hdf5File) -> Result<()> {
        for info in &group.variables {
            let missing = || {
                Error::unsupported(format!(
                    "an append to a file with no dataset for {}",
                    info.path
                ))
            };
            let dataset = crate::netcdf::hdf5_dataset(hdf5, info).ok_or_else(missing)?;
            let id = self
                .writer
                .find_dataset(self.groups[index].id, &dataset.name);
            let mut dimensions = Vec::with_capacity(info.dimensions.len());
            for name in &info.dimensions {
                dimensions.push(self.find_dimension(index, name).ok_or_else(missing)?);
            }
            let variable = self.variables.len();
            if let (true, Some(&first)) = (info.is_coordinate, dimensions.first()) {
                if self.dimensions[first].group == index && self.dimensions[first].name == info.name
                {
                    self.dimensions[first].coordinate = Some(variable);
                }
            }
            let chunks = match &dataset.layout {
                oxcdf_hdf5::message::Layout::Chunked { chunk_dims, .. } => {
                    Some(chunk_dims.iter().map(|&dim| u64::from(dim)).collect())
                }
                _ => None,
            };
            let definition = Definition {
                dtype: DType::of(&dataset.datatype),
                order: dataset.datatype.byte_order().unwrap_or(ByteOrder::Little),
                chunks,
                shuffle: false,
                deflate: None,
                zstd: None,
                fletcher32: false,
                fill: None,
            };
            self.groups[index].variables.push(variable);
            self.variables.push(Variable {
                name: info.name.clone(),
                dataset: id.ok_or_else(missing)?,
                definition,
                dimensions,
                shape: dataset.shape.clone(),
                written: true,
                attached: true,
            });
        }
        for (position, child) in group.groups.iter().enumerate() {
            let child_index = self.groups[index].groups[position].1;
            self.import_variables(child, child_index, hdf5)?;
        }
        Ok(())
    }

    /// The dimension `name` that a variable of `group` sees.
    fn find_dimension(&self, group: usize, name: &str) -> Option<usize> {
        let mut at = Some(group);
        while let Some(group) = at {
            let own = &self.groups[group].dimensions;
            if let Some(&d) = own.iter().find(|&&d| self.dimensions[d].name == name) {
                return Some(d);
            }
            at = self.groups[group].parent;
        }
        None
    }

    /// The dimension `name` that `group` defines.
    fn own_dimension(&self, group: usize, name: &str) -> Option<usize> {
        let own = &self.groups[group].dimensions;
        own.iter()
            .copied()
            .find(|&d| self.dimensions[d].name == name)
    }

    fn find_variable(&self, group: usize, name: &str) -> Option<usize> {
        let own = &self.groups[group].variables;
        own.iter()
            .copied()
            .find(|&v| self.variables[v].name == name)
    }

    fn find_group(&self, group: usize, name: &str) -> Option<usize> {
        let own = &self.groups[group].groups;
        own.iter().find(|(n, _)| n == name).map(|&(_, g)| g)
    }

    fn add_dimension(&mut self, group: usize, name: &str, len: u64, unlimited: bool) -> Result<()> {
        check_name(name)?;
        if self.own_dimension(group, name).is_some() || self.find_group(group, name).is_some() {
            return Err(in_use(name));
        }
        if i32::try_from(self.dimensions.len()).is_err() {
            return Err(Error::unsupported("more dimensions than netCDF can number"));
        }
        if let Some(variable) = self.find_variable(group, name) {
            // The variable does not use the new dimension, so it is no coordinate.
            let link = format!("{NON_COORDINATE_PREFIX}{name}");
            let dataset = self.variables[variable].dataset;
            self.writer
                .rename_dataset(self.groups[group].id, dataset, &link)?;
        }
        self.groups[group].dimensions.push(self.dimensions.len());
        self.dimensions.push(Dimension {
            name: name.to_owned(),
            group,
            len,
            unlimited,
            coordinate: None,
            scale: None,
        });
        Ok(())
    }

    fn add_group(&mut self, parent: usize, name: &str) -> Result<usize> {
        check_name(name)?;
        let used = self.own_dimension(parent, name).is_some()
            || self.find_group(parent, name).is_some()
            || self.find_variable(parent, name).is_some();
        if used {
            return Err(in_use(name));
        }
        let id = self.writer.add_group(self.groups[parent].id, name)?;
        let group = self.groups.len();
        self.groups.push(Group {
            id,
            parent: Some(parent),
            dimensions: Vec::new(),
            variables: Vec::new(),
            groups: Vec::new(),
        });
        self.groups[parent].groups.push((name.to_owned(), group));
        Ok(group)
    }

    fn add_variable(
        &mut self,
        group: usize,
        name: &str,
        dimension_names: &[&str],
        dtype: &DType,
    ) -> Result<usize> {
        check_name(name)?;
        if self.find_variable(group, name).is_some() || self.find_group(group, name).is_some() {
            return Err(in_use(name));
        }
        let mut dimensions = Vec::with_capacity(dimension_names.len());
        for dimension in dimension_names {
            let found = self.find_dimension(group, dimension).ok_or_else(|| {
                Error::bad_request(format!(
                    "variable {name:?} uses the unknown dimension {dimension:?}"
                ))
            })?;
            dimensions.push(found);
        }
        // A variable with the name of a dimension is its coordinate only on axis 0.
        let namesake = self.own_dimension(group, name);
        let coordinate = namesake.filter(|d| dimensions.first() == Some(d));
        // The dataset of such a dimension holds the link that the coordinate needs.
        if coordinate.is_some_and(|d| self.dimensions[d].scale.is_some()) {
            return Err(Error::unsupported(format!(
                "a new coordinate variable for the dimension {name:?} of the file"
            )));
        }
        let definition = Definition {
            dtype: dtype.clone(),
            order: ByteOrder::Little,
            chunks: None,
            shuffle: false,
            deflate: None,
            zstd: None,
            fletcher32: false,
            fill: None,
        };
        let spec = self.spec(&definition, &dimensions)?;
        let shape = spec.shape.clone();
        let link = match (namesake, coordinate) {
            (Some(_), None) => format!("{NON_COORDINATE_PREFIX}{name}"),
            _ => name.to_owned(),
        };
        let dataset = self
            .writer
            .add_dataset(self.groups[group].id, &link, spec)?;
        if !dimensions.is_empty() {
            // The check in `add_dimension` keeps each index inside an `i32`.
            let ids: Vec<i32> = dimensions.iter().map(|&d| d as i32).collect();
            self.writer
                .put_numeric_attribute(dataset, "_Netcdf4Coordinates", &ids)?;
        }
        let variable = self.variables.len();
        if let Some(dimension) = coordinate {
            self.dimensions[dimension].coordinate = Some(variable);
        }
        self.groups[group].variables.push(variable);
        self.variables.push(Variable {
            name: name.to_owned(),
            dataset,
            definition,
            dimensions,
            shape,
            written: false,
            attached: false,
        });
        Ok(variable)
    }

    /// The HDF5 dataset for a variable with no values.
    fn spec(&self, definition: &Definition, dimensions: &[usize]) -> Result<DatasetSpec> {
        let axes: Vec<&Dimension> = dimensions.iter().map(|&d| &self.dimensions[d]).collect();
        // An unlimited axis starts with no elements and grows at a write.
        let shape: Vec<u64> = axes
            .iter()
            .map(|d| if d.unlimited { 0 } else { d.len })
            .collect();
        let datatype = datatype_of(&definition.dtype, definition.order)?;
        let element_bytes = u64::from(datatype.size);
        let mut spec = DatasetSpec::new(datatype, &shape);

        let mut filters = Vec::new();
        if definition.fletcher32 {
            filters.push(Filter::Fletcher32);
        }
        if definition.shuffle {
            filters.push(Filter::Shuffle);
        }
        match (definition.deflate, definition.zstd) {
            (Some(_), Some(_)) => {
                return Err(Error::bad_request(
                    "a variable takes deflate or Zstandard, not the two",
                ))
            }
            (Some(level), None) => filters.push(Filter::Deflate(level)),
            (None, Some(level)) => filters.push(Filter::Zstd(level)),
            (None, None) => {}
        }

        let unlimited = axes.iter().any(|d| d.unlimited);
        if unlimited || definition.chunks.is_some() || !filters.is_empty() {
            if axes.is_empty() {
                return Err(Error::bad_request(
                    "a scalar variable takes no chunks and no filters",
                ));
            }
            let chunks = match &definition.chunks {
                Some(chunks) => chunks.clone(),
                None => default_chunks(&axes, element_bytes),
            };
            let max: Vec<Option<u64>> = axes
                .iter()
                .map(|d| (!d.unlimited).then_some(d.len))
                .collect();
            spec = spec
                .chunked(&chunks)
                .with_max_shape(&max)
                .with_filters(&filters);
        }

        let fill = match (&definition.dtype, &definition.fill) {
            (DType::String, _) => None,
            (_, Some(fill)) => Some(fill.clone()),
            (dtype, None) => default_fill(dtype),
        };
        if let Some(fill) = fill {
            spec = spec.with_fill_value(in_order(fill, definition.order));
        }
        Ok(spec)
    }

    /// Write the dimension scales and close the HDF5 writer.
    fn finish(mut self) -> Result<Sink> {
        let mut scales = Vec::with_capacity(self.dimensions.len());
        for (id, dimension) in self.dimensions.iter().enumerate() {
            let (scale, name) = match (dimension.coordinate, dimension.scale) {
                (Some(variable), _) => {
                    let variable = &self.variables[variable];
                    let mut shape = variable.shape.clone();
                    // The scale gives the length of the dimension to a reader.
                    if dimension.unlimited && shape[0] < dimension.len {
                        shape[0] = dimension.len;
                        self.writer.resize(variable.dataset, &shape)?;
                    }
                    (variable.dataset, dimension.name.clone())
                }
                (None, Some((scale, len))) => {
                    // The scale gives the length of the dimension to a reader.
                    if dimension.unlimited && len < dimension.len {
                        self.writer.resize(scale, &[dimension.len])?;
                    }
                    let name = format!("{PHANTOM_DIMENSION_PREFIX}.{:>10}", dimension.len);
                    (scale, name)
                }
                (None, None) => {
                    let datatype = numeric_datatype(DType::Float(4), ByteOrder::Big)?;
                    let mut spec = DatasetSpec::new(datatype, &[dimension.len]);
                    if dimension.unlimited {
                        spec = spec.chunked(&[SCALE_CHUNK]).with_max_shape(&[None]);
                    }
                    let group = self.groups[dimension.group].id;
                    let scale = self.writer.add_dataset(group, &dimension.name, spec)?;
                    let name = format!("{PHANTOM_DIMENSION_PREFIX}.{:>10}", dimension.len);
                    (scale, name)
                }
            };
            let class = terminated_text("CLASS", "DIMENSION_SCALE");
            self.writer.put_attribute(scale, class)?;
            self.writer
                .put_attribute(scale, terminated_text("NAME", &name))?;
            let id = Attribute {
                name: "_Netcdf4Dimid".to_owned(),
                datatype: numeric_datatype(DType::Int(4), ByteOrder::Little)?,
                dataspace: scalar(),
                // The check in `add_dimension` keeps each index inside an `i32`.
                data: (id as i32).to_le_bytes().to_vec(),
            };
            self.writer.put_attribute(scale, id)?;
            scales.push(scale);
        }
        for (index, variable) in self.variables.iter().enumerate() {
            // The HDF5 library gives no scale to a dataset that is a scale.
            let is_scale = variable
                .dimensions
                .first()
                .is_some_and(|&d| self.dimensions[d].coordinate == Some(index));
            if is_scale || variable.attached {
                continue;
            }
            for (axis, &dimension) in variable.dimensions.iter().enumerate() {
                self.writer
                    .attach_scale(variable.dataset, axis, scales[dimension])?;
            }
        }
        self.writer.close()
    }
}

/// The error for a file with no state. Only a defect in the writer causes it.
fn state_defect() -> Error {
    Error::Malformed("a defect in the writer: the file has no state".to_owned())
}

/// The error for a wrong close of a file from [`append`], after its in-place close.
///
/// `next` tells the caller what to do with the file after [`FileMut::close`].
fn in_place_close(next: &str) -> Error {
    Error::bad_request(format!(
        "a file from append closes in place, and the writer wrote it: use close, then {next}"
    ))
}

/// The error for a file in memory that has a file sink. Only a defect in the writer causes it.
fn sink_defect() -> Error {
    Error::Malformed("a defect in the writer: a file in memory has a file sink".to_owned())
}

/// Reject a name that netcdf-c does not accept.
///
/// The first character is a letter, a digit, an underscore or not ASCII.
/// Each other character is printable or not ASCII, and is not a slash. The
/// last character is not a space.
fn check_name(name: &str) -> Result<()> {
    let first = |c: char| c.is_ascii_alphanumeric() || c == '_' || !c.is_ascii();
    let other = |c: char| (c == ' ' || c.is_ascii_graphic() || !c.is_ascii()) && c != '/';
    let wrong = !name.starts_with(first)
        || !name.chars().all(other)
        || name.ends_with(' ')
        || name.len() > MAX_NAME_BYTES
        || name.starts_with(NON_COORDINATE_PREFIX);
    if wrong {
        return Err(Error::bad_request(format!(
            "{name:?} is not a valid netCDF name"
        )));
    }
    Ok(())
}

/// The error for a name that the group has.
fn in_use(name: &str) -> Error {
    Error::bad_request(format!("the group already uses the name {name:?}"))
}

/// The scalar dataspace.
fn scalar() -> Dataspace {
    Dataspace {
        kind: DataspaceKind::Scalar,
        dims: Vec::new(),
        max_dims: None,
    }
}

/// A scalar text attribute with a NUL at its end, as the HDF5 scale interface writes it.
fn terminated_text(name: &str, text: &str) -> Attribute {
    let mut data = text.as_bytes().to_vec();
    data.push(0);
    Attribute {
        name: name.to_owned(),
        datatype: Datatype {
            version: 1,
            // A name has 256 bytes at most, so the size fits.
            size: data.len() as u32,
            class: DatatypeClass::String {
                pad: StringPad::NullTerminate,
                charset: CharSet::Ascii,
            },
        },
        dataspace: scalar(),
        data,
    }
}

/// The HDF5 datatype of the netCDF type `dtype`.
fn datatype_of(dtype: &DType, order: ByteOrder) -> Result<Datatype> {
    match dtype {
        DType::Int(_) | DType::Uint(_) | DType::Float(_) => numeric_datatype(dtype.clone(), order),
        DType::Char => Ok(Datatype {
            version: 1,
            size: 1,
            class: DatatypeClass::String {
                pad: StringPad::NullTerminate,
                charset: CharSet::Ascii,
            },
        }),
        DType::String => Ok(vlen_string_datatype()),
        other => Err(Error::unsupported(format!(
            "a variable of the type {other:?}"
        ))),
    }
}

/// The default fill value of netCDF for `dtype`, in native byte order.
///
/// `char` has a fill value of zero, which is the value of bytes with no write.
fn default_fill(dtype: &DType) -> Option<Vec<u8>> {
    Some(match dtype {
        DType::Int(1) => (-127i8).to_ne_bytes().to_vec(),
        DType::Int(2) => (-32767i16).to_ne_bytes().to_vec(),
        DType::Int(4) => (-2_147_483_647i32).to_ne_bytes().to_vec(),
        DType::Int(8) => (-9_223_372_036_854_775_806i64).to_ne_bytes().to_vec(),
        DType::Uint(1) => u8::MAX.to_ne_bytes().to_vec(),
        DType::Uint(2) => u16::MAX.to_ne_bytes().to_vec(),
        DType::Uint(4) => u32::MAX.to_ne_bytes().to_vec(),
        DType::Uint(8) => (u64::MAX - 1).to_ne_bytes().to_vec(),
        DType::Float(4) => 9.969_21e36_f32.to_ne_bytes().to_vec(),
        DType::Float(8) => 9.969_209_968_386_869e36_f64.to_ne_bytes().to_vec(),
        _ => return None,
    })
}

/// The default chunk shape, after the rule of netcdf-c.
fn default_chunks(axes: &[&Dimension], element_bytes: u64) -> Vec<u64> {
    let unlimited = axes.iter().filter(|d| d.unlimited).count();
    let fixed = axes.len() - unlimited;
    let budget = (DEFAULT_CHUNK_BYTES / element_bytes).max(1) as f64;
    let mut chunks: Vec<u64> = if axes.len() == 1 && unlimited == 1 {
        vec![(DEFAULT_RECORD_CHUNK_BYTES / element_bytes).max(1)]
    } else if fixed == 0 {
        let side = budget.powf(1.0 / axes.len() as f64);
        vec![(side as u64).max(1); axes.len()]
    } else {
        let values: f64 = axes
            .iter()
            .filter(|d| !d.unlimited)
            .map(|d| d.len.max(1) as f64)
            .product();
        // Each fixed axis gets the same part of its length.
        let part = (budget / values).powf(1.0 / fixed as f64);
        axes.iter()
            .map(|d| match d.unlimited {
                true => 1,
                false => ((part * d.len as f64 - 0.5) as u64).clamp(1, d.len.max(1)),
            })
            .collect()
    };
    let bytes = |chunks: &[u64]| {
        chunks
            .iter()
            .fold(element_bytes, |n, &c| n.saturating_mul(c))
    };
    while bytes(&chunks) > DEFAULT_CHUNK_BYTES && chunks.iter().any(|&c| c > 1) {
        for chunk in &mut chunks {
            *chunk = (*chunk / 2).max(1);
        }
    }
    chunks
}

/// Convert `values` to the stored type `dtype`, as bytes in native order.
fn converted<T: Element>(values: &[T], dtype: &DType) -> Result<Vec<u8>> {
    fn to<T: Element, U: Element>(values: &[T]) -> Vec<u8> {
        let values: Vec<U> = values.iter().map(|&v| v.convert()).collect();
        let mut bytes = Vec::new();
        U::extend_ne_bytes(&values, &mut bytes);
        bytes
    }
    Ok(match dtype {
        DType::Int(1) => to::<T, i8>(values),
        DType::Int(2) => to::<T, i16>(values),
        DType::Int(4) => to::<T, i32>(values),
        DType::Int(8) => to::<T, i64>(values),
        DType::Uint(1) | DType::Char => to::<T, u8>(values),
        DType::Uint(2) => to::<T, u16>(values),
        DType::Uint(4) => to::<T, u32>(values),
        DType::Uint(8) => to::<T, u64>(values),
        DType::Float(4) => to::<T, f32>(values),
        DType::Float(8) => to::<T, f64>(values),
        other => {
            return Err(Error::unsupported(format!(
                "numbers for a variable of the type {other:?}"
            )))
        }
    })
}

/// Turn one element in native byte order into the byte order `order`.
fn in_order(mut element: Vec<u8>, order: ByteOrder) -> Vec<u8> {
    let native = cfg!(target_endian = "little") == (order == ByteOrder::Little);
    if !native {
        element.reverse();
    }
    element
}

/// Turn elements of `dtype` in native byte order into the byte order `order`.
fn in_order_of(mut bytes: Vec<u8>, order: ByteOrder, dtype: &DType) -> Vec<u8> {
    let size = match dtype {
        DType::Int(n) | DType::Uint(n) | DType::Float(n) => usize::from(*n),
        _ => 1,
    };
    let native = cfg!(target_endian = "little") == (order == ByteOrder::Little);
    if !native && size > 1 {
        for element in bytes.chunks_exact_mut(size) {
            element.reverse();
        }
    }
    bytes
}

/// Put `value` as the attribute `name` of `object`.
fn put_attribute(
    writer: &mut Hdf5Writer<Sink>,
    object: ObjectId,
    name: &str,
    value: AttributeValue,
) -> Result<()> {
    if RESERVED_ATTRIBUTES.contains(&name) || name == "_FillValue" {
        return Err(Error::bad_request(format!(
            "netCDF reserves the attribute name {name:?}"
        )));
    }
    use AttributeValue as V;
    match value {
        V::Uchar(v) => writer.put_numeric_attribute(object, name, &[v]),
        V::Uchars(v) => writer.put_numeric_attribute(object, name, &v),
        V::Schar(v) => writer.put_numeric_attribute(object, name, &[v]),
        V::Schars(v) => writer.put_numeric_attribute(object, name, &v),
        V::Ushort(v) => writer.put_numeric_attribute(object, name, &[v]),
        V::Ushorts(v) => writer.put_numeric_attribute(object, name, &v),
        V::Short(v) => writer.put_numeric_attribute(object, name, &[v]),
        V::Shorts(v) => writer.put_numeric_attribute(object, name, &v),
        V::Uint(v) => writer.put_numeric_attribute(object, name, &[v]),
        V::Uints(v) => writer.put_numeric_attribute(object, name, &v),
        V::Int(v) => writer.put_numeric_attribute(object, name, &[v]),
        V::Ints(v) => writer.put_numeric_attribute(object, name, &v),
        V::Ulonglong(v) => writer.put_numeric_attribute(object, name, &[v]),
        V::Ulonglongs(v) => writer.put_numeric_attribute(object, name, &v),
        V::Longlong(v) => writer.put_numeric_attribute(object, name, &[v]),
        V::Longlongs(v) => writer.put_numeric_attribute(object, name, &v),
        V::Float(v) => writer.put_numeric_attribute(object, name, &[v]),
        V::Floats(v) => writer.put_numeric_attribute(object, name, &v),
        V::Double(v) => writer.put_numeric_attribute(object, name, &[v]),
        V::Doubles(v) => writer.put_numeric_attribute(object, name, &v),
        // A fixed string needs one byte at least.
        V::Str(text) if text.is_empty() => writer.put_attribute(object, terminated_text(name, "")),
        V::Str(text) => writer.put_text_attribute(object, name, &text),
        V::Strs(texts) => writer.put_string_attribute(object, name, &texts),
        V::Raw(_) => Err(Error::unsupported("an attribute of raw bytes")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis(len: u64, unlimited: bool) -> Dimension {
        Dimension {
            name: String::new(),
            group: ROOT,
            len,
            unlimited,
            coordinate: None,
            scale: None,
        }
    }

    #[test]
    fn default_chunks_stay_inside_each_axis_and_the_size_target() {
        let time = axis(0, true);
        let lat = axis(180, false);
        let lon = axis(360, false);
        assert_eq!(default_chunks(&[&time], 8), [512]);
        assert_eq!(default_chunks(&[&time, &lat, &lon], 4), [1, 180, 360]);
        assert_eq!(default_chunks(&[&lat, &lon], 4), [180, 360]);
        let wide = axis(100_000_000, false);
        let chunks = default_chunks(&[&wide, &lon], 8);
        assert!(chunks.iter().all(|&c| c >= 1));
        assert!(chunks[0] <= wide.len && chunks[1] <= lon.len);
        assert!(chunks.iter().product::<u64>() * 8 <= DEFAULT_CHUNK_BYTES);
        let squares = default_chunks(&[&time, &time], 4);
        assert_eq!(squares, [1024, 1024]);
    }

    #[test]
    fn an_append_in_memory_keeps_one_reference_to_the_buffer() {
        let mut file = create_in_memory().unwrap();
        file.add_dimension("x", 3).unwrap();
        let mut v = file.add_variable::<i32>("v", &["x"]).unwrap();
        v.put_values(&[1, 2, 3], ..).unwrap();
        let bytes = Arc::new(file.into_bytes().unwrap());
        let weak = Arc::downgrade(&bytes);

        let mut file = append_shared(bytes).unwrap();
        // The reader is gone and the sink holds the only reference.
        assert_eq!(weak.strong_count(), 1);
        let mut v = file.variable_mut("v").unwrap();
        v.put_value(9, [1]).unwrap();
        assert_eq!(weak.strong_count(), 0);
        let source = Arc::new(oxcdf_hdf5::MemorySource::new(file.into_bytes().unwrap()));
        let file = NetcdfFile::from_source(source).unwrap();
        let v = file.variable("v").unwrap();
        assert_eq!(v.get_values::<i32, _>(..).unwrap(), [1, 9, 3]);
    }

    #[test]
    fn a_memory_buffer_moves_a_unique_buffer_and_copies_a_shared_one() {
        let bytes = Arc::new(b"abcd".to_vec());
        let address = bytes.as_ptr();
        let mut buffer = MemoryBuffer::shared(bytes);
        buffer.write_all_at(1, b"x").unwrap();
        let moved = buffer.into_inner();
        assert_eq!(moved, b"axcd");
        assert_eq!(moved.as_ptr(), address);

        let bytes = Arc::new(b"abcd".to_vec());
        let mut buffer = MemoryBuffer::shared(Arc::clone(&bytes));
        let mut read = [0u8; 2];
        buffer.read_exact_at(2, &mut read).unwrap();
        assert_eq!(&read, b"cd");
        buffer.write_all_at(0, b"z").unwrap();
        assert_eq!(buffer.into_inner(), b"zbcd");
        assert_eq!(*bytes, b"abcd");
    }

    #[test]
    fn a_shared_source_rejects_a_read_past_the_end() {
        let source = SharedSource(Arc::new(b"abc".to_vec()));
        assert_eq!(source.read_vec(1, 2).unwrap(), b"bc");
        let error = source.read_vec(2, 2).unwrap_err();
        assert!(matches!(error, Error::OutOfBounds { .. }), "{error:?}");
        assert!(source.read_vec(u64::MAX, 1).is_err());
    }

    #[test]
    fn a_wrong_name_is_a_bad_request() {
        for name in ["", ".", "a/b", "a\0", "_nc4_non_coord_x"] {
            assert!(matches!(check_name(name), Err(Error::BadRequest(_))));
        }
        assert!(check_name(&"n".repeat(256)).is_ok());
        assert!(check_name(&"n".repeat(257)).is_err());
    }
}
