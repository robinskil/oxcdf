//! The asynchronous netCDF interface.
//!
//! This module mirrors [`crate::netcdf`]. An open awaits. A read of values
//! awaits. Everything else answers at once.
//!
//! ```no_run
//! # async fn run(source: std::sync::Arc<dyn oxcdf::AsyncByteSource>) -> oxcdf::Result<()> {
//! let file = oxcdf::open_async(source).await?;
//!
//! for d in file.dimensions() {
//!     println!("{} = {}", d.name, d.len);
//! }
//!
//! let temp = file.variable("TEMP").unwrap();
//! println!("{:?}", temp.attribute("units").map(|a| &a.value));
//!
//! let all = temp.read().await?.to_f64()?;
//! let part = temp.read_slice(&[0..8, 10..30]).await?.to_f64()?;
//! # Ok(()) }
//! ```
//!
//! # How the open works
//!
//! An open walks the file metadata. The walk is a chain of dependent reads.
//! This module runs the synchronous walk over pages held in memory. It fetches
//! the pages it lacks and runs the walk again. See [`crate::replay`].
//!
//! A chunk index and a string heap resolve the same way, on first use. A caller
//! never calls `prepare`.

use std::sync::Arc;

use crate::async_source::AsyncByteSource;
use crate::error::Result;
use crate::hdf5::context::Ctx;
use crate::index::{DatasetIndex, Hdf5File, OpenOptions};
use crate::netcdf::{Chunk, NcAttribute, NcDimension, NcGroup, NcVariable, NetcdfFile, Values};
use crate::read::Hyperslab;

/// A netCDF-4 file opened over an asynchronous byte source.
///
/// The metadata is complete when the open returns. A metadata method therefore
/// takes no `await`. A read of values takes one.
///
/// This type is `Send + Sync`. Share it behind an [`Arc`]. Concurrent reads
/// need no coordination.
#[derive(Debug, Clone)]
pub struct AsyncFile {
    netcdf: NetcdfFile,
    source: Arc<dyn AsyncByteSource>,
    options: OpenOptions,
}

impl AsyncFile {
    /// Open a file with default options.
    pub async fn open(source: Arc<dyn AsyncByteSource>) -> Result<Self> {
        Self::open_with(source, OpenOptions::default()).await
    }

    /// Open a file with explicit options.
    ///
    /// Use [`OpenOptions::remote`] for object storage. It sets a 256 KiB
    /// request size and a 128 MiB byte cache.
    pub async fn open_with(source: Arc<dyn AsyncByteSource>, options: OpenOptions) -> Result<Self> {
        let io_cache = options.build_io_cache();
        let page_size = options.request_size();

        // Walk the metadata over pages held in memory, fetching what the walk
        // asks for. The byte cache keeps the pages for the reads that follow.
        let netcdf = crate::replay::replay(
            source.as_ref(),
            io_cache.as_ref(),
            page_size,
            options.prefetch_bytes(),
            |bytes| {
                let hdf5 = Hdf5File::from_source_reusing(bytes, &options, None)?;
                NetcdfFile::from_hdf5(hdf5)
            },
        )
        .await?;

        // Give the file the caches the reads need. The open built its own index
        // over a private source, so replace that too.
        let netcdf = netcdf.map_hdf5(|hdf5| {
            hdf5.with_io_cache(io_cache)
                .with_cache(options.build_chunk_cache())
                .with_io(options.merge_policy())
        });

        Ok(Self {
            netcdf,
            source,
            options,
        })
    }

    /// The root group.
    pub fn root(&self) -> &NcGroup {
        self.netcdf.root()
    }

    /// Every variable in the file, depth first.
    pub fn variables(&self) -> Vec<AsyncVariable<'_>> {
        self.netcdf
            .root()
            .variables_recursive()
            .into_iter()
            .filter_map(|info| self.bind(info))
            .collect()
    }

    /// One variable by name. A leading slash is optional.
    pub fn variable(&self, path: &str) -> Option<AsyncVariable<'_>> {
        self.bind(self.netcdf.variable_info(path)?)
    }

    /// The dimensions of the root group.
    pub fn dimensions(&self) -> &[NcDimension] {
        &self.netcdf.root().dimensions
    }

    /// The global attributes.
    pub fn attributes(&self) -> &[NcAttribute] {
        &self.netcdf.root().attributes
    }

    /// One global attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&NcAttribute> {
        self.netcdf.root().attribute(name)
    }

    /// A group by absolute path, such as `/forecast`.
    pub fn group(&self, path: &str) -> Option<&NcGroup> {
        self.netcdf.group(path)
    }

    /// The parsed file, for the metadata this layer does not model.
    ///
    /// Its synchronous read methods fail with [`Error::Incomplete`]. Only its
    /// metadata is in memory.
    pub fn netcdf(&self) -> &NetcdfFile {
        &self.netcdf
    }

    /// The byte source this file reads from.
    pub fn source(&self) -> &Arc<dyn AsyncByteSource> {
        &self.source
    }

    fn bind<'a>(&'a self, info: &'a NcVariable) -> Option<AsyncVariable<'a>> {
        Some(AsyncVariable {
            file: self,
            info,
            dataset: self.netcdf.hdf5().dataset(&info.path)?,
        })
    }

    /// Run a synchronous walk over the file, fetching whatever it needs.
    async fn walk<T, F>(&self, build: F) -> Result<T>
    where
        F: Fn(Ctx<'_>) -> Result<T>,
    {
        let hdf5 = self.netcdf.hdf5();
        crate::replay::replay(
            self.source.as_ref(),
            hdf5.io_cache(),
            self.options.request_size(),
            // The metadata this reaches is scattered, not clustered at the
            // front, so read the page that was asked for and little else.
            self.options.request_size(),
            |bytes| {
                let ctx = Ctx::new(bytes.as_ref(), hdf5.superblock())
                    .with_cache(hdf5.cache())
                    .with_io(hdf5.io());
                build(ctx)
            },
        )
        .await
    }
}

/// A readable handle on one variable.
///
/// Dereferences to [`NcVariable`], so `variable.shape`, `variable.dimensions`
/// and the rest are available directly.
#[derive(Clone, Copy)]
pub struct AsyncVariable<'a> {
    file: &'a AsyncFile,
    info: &'a NcVariable,
    dataset: &'a DatasetIndex,
}

impl std::fmt::Debug for AsyncVariable<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncVariable")
            .field("path", &self.info.path)
            .field("shape", &self.info.shape)
            .finish()
    }
}

impl std::ops::Deref for AsyncVariable<'_> {
    type Target = NcVariable;
    fn deref(&self) -> &Self::Target {
        self.info
    }
}

impl<'a> AsyncVariable<'a> {
    /// The variable's metadata.
    pub fn info(&self) -> &'a NcVariable {
        self.info
    }

    /// The HDF5 dataset behind the variable.
    pub fn dataset(&self) -> &'a DatasetIndex {
        self.dataset
    }

    /// The variable's attributes.
    pub fn attributes(&self) -> &'a [NcAttribute] {
        &self.info.attributes
    }

    /// One attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&'a NcAttribute> {
        self.info.attributes.iter().find(|a| a.name == name)
    }

    /// The value type.
    pub fn dtype(&self) -> crate::netcdf::DType {
        crate::netcdf::DType::of(&self.dataset.datatype)
    }

    /// The element type as HDF5 records it.
    pub fn datatype(&self) -> &'a crate::hdf5::message::Datatype {
        &self.dataset.datatype
    }

    /// Number of elements in the whole variable.
    pub fn element_count(&self) -> u64 {
        self.info.shape.iter().product()
    }

    /// The shape of one storage chunk, when the variable is chunked.
    ///
    /// `None` means the variable is stored contiguously and has no chunk grid.
    pub fn chunk_shape(&self) -> Option<Vec<u64>> {
        match &self.dataset.layout {
            crate::hdf5::message::Layout::Chunked { chunk_dims, .. } => {
                Some(chunk_dims.iter().map(|&d| d as u64).collect())
            }
            _ => None,
        }
    }

    /// Whether this reader can decode the variable's values.
    ///
    /// Cheap. It reads nothing.
    pub fn is_readable(&self) -> bool {
        self.dataset.is_readable()
    }

    /// Read the whole variable.
    pub async fn read(&self) -> Result<Values> {
        self.read_selection(&Hyperslab::all(&self.info.shape)).await
    }

    /// Read a slice, given one range for each axis.
    pub async fn read_slice(&self, ranges: &[std::ops::Range<u64>]) -> Result<Values> {
        self.read_selection(&Hyperslab::from_ranges(&self.info.path, &self.info.shape, ranges)?)
            .await
    }

    /// Read an explicit selection.
    pub async fn read_selection(&self, slab: &Hyperslab) -> Result<Values> {
        let hdf5 = self.file.netcdf.hdf5();
        self.resolve_index().await?;

        let raw = crate::read::read_hyperslab_async_with(
            self.file.source.as_ref(),
            hdf5.superblock(),
            hdf5.cache(),
            hdf5.io_cache(),
            hdf5.io(),
            self.dataset,
            slab,
        )
        .await?;

        // A variable-length value stores a pointer into a heap. Follow it.
        self.file
            .walk(|ctx| crate::netcdf::values_from_raw(ctx, self.dataset, raw.clone()))
            .await
    }

    /// Every storage chunk of the variable.
    ///
    /// A chunk is clipped to the variable. The chunks cover it exactly once.
    /// Read them concurrently: each one is a separate byte range.
    pub async fn chunks(&self) -> Result<Vec<Chunk>> {
        self.resolve_index().await?;
        crate::netcdf::chunks_of(self.dataset)
    }

    /// Read one chunk.
    pub async fn read_chunk(&self, chunk: &Chunk) -> Result<Values> {
        self.read_selection(&chunk.selection()).await
    }

    /// Read the whole variable into an `ndarray` of `f64`.
    #[cfg(feature = "ndarray")]
    pub async fn read_array_f64(&self) -> Result<ndarray::ArrayD<f64>> {
        self.read().await?.to_array_f64()
    }

    /// Read the whole variable into an `ndarray` of `i64`.
    #[cfg(feature = "ndarray")]
    pub async fn read_array_i64(&self) -> Result<ndarray::ArrayD<i64>> {
        self.read().await?.to_array_i64()
    }

    /// Resolve the chunk index, if it is not resolved already.
    ///
    /// The walk is pure. Two tasks that race it both do the work. The duplicate
    /// wastes effort. It does not give a wrong answer, and it avoids a lock.
    async fn resolve_index(&self) -> Result<()> {
        if self.dataset.is_prepared() {
            return Ok(());
        }
        self.file.walk(|ctx| self.dataset.prepare(ctx)).await
    }
}

#[cfg(feature = "object-store")]
impl AsyncFile {
    /// Open a file held in object storage.
    ///
    /// This uses [`OpenOptions::remote`]. The reader reads byte ranges. It
    /// needs no local copy.
    ///
    /// ```no_run
    /// # async fn run(
    /// #     store: std::sync::Arc<dyn object_store::ObjectStore>,
    /// # ) -> oxcdf::Result<()> {
    /// use object_store::path::Path;
    ///
    /// let file = oxcdf::AsyncFile::open_store(store, Path::from("13857_prof.nc")).await?;
    /// let values = file.variable("TEMP").unwrap().read().await?.to_f64()?;
    /// # Ok(()) }
    /// ```
    pub async fn open_store(
        store: Arc<dyn object_store::ObjectStore>,
        path: object_store::path::Path,
    ) -> Result<Self> {
        Self::open_store_with(store, path, OpenOptions::remote()).await
    }

    /// Open a file held in object storage, with explicit options.
    pub async fn open_store_with(
        store: Arc<dyn object_store::ObjectStore>,
        path: object_store::path::Path,
        options: OpenOptions,
    ) -> Result<Self> {
        let source = crate::async_source::AsyncObjectStoreSource::new(store, path).await?;
        Self::open_with(Arc::new(source), options).await
    }
}
