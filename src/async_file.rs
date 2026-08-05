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
//! let all = temp.get_values::<f32, _>(..).await?;
//! let part = temp.get_values::<f32, _>([0..8, 10..30]).await?;
//! let one = temp.get_value::<f32, _>([0, 3]).await?;
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
use crate::error::{Error, Result};
use crate::hdf5::context::Ctx;
use crate::index::{DatasetIndex, OpenOptions};
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
            // `from_source` reads the magic bytes, so one walk opens either
            // container.
            NetcdfFile::from_source,
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

    /// One dimension of the root group by name.
    pub fn dimension(&self, name: &str) -> Option<&NcDimension> {
        self.netcdf.dimension(name)
    }

    /// The length of one dimension of the root group.
    pub fn dimension_len(&self, name: &str) -> Option<u64> {
        self.netcdf.dimension_len(name)
    }

    /// The groups directly inside the root group.
    pub fn groups(&self) -> &[NcGroup] {
        self.netcdf.groups()
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

    /// Which container holds the file.
    pub fn container(&self) -> crate::Container {
        self.netcdf.container()
    }

    /// The parsed file, for the metadata this layer does not model.
    ///
    /// Its synchronous read methods fail with [`crate::Error::Incomplete`].
    /// Only its metadata is in memory.
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
            store: self.netcdf.store_for(info)?,
        })
    }

    /// Run a synchronous walk over raw bytes, fetching whatever it needs.
    ///
    /// This serves the classic container, which addresses bytes directly and
    /// has no HDF5 context.
    async fn walk_bytes<T, F>(&self, build: F) -> Result<T>
    where
        F: Fn(Arc<dyn crate::source::ByteSource>) -> Result<T>,
    {
        crate::replay::replay(
            self.source.as_ref(),
            self.netcdf.hdf5().and_then(|h| h.io_cache()),
            self.options.request_size(),
            self.options.request_size(),
            build,
        )
        .await
    }

    /// Run a synchronous HDF5 walk over the file, fetching whatever it needs.
    async fn walk<T, F>(&self, build: F) -> Result<T>
    where
        F: Fn(Ctx<'_>) -> Result<T>,
    {
        let hdf5 = self
            .netcdf
            .hdf5()
            .ok_or_else(|| Error::malformed("an HDF5 walk on a classic file"))?;
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
    store: crate::netcdf::VarStore<'a>,
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

    /// The HDF5 dataset behind the variable. `None` for a classic file.
    pub fn dataset(&self) -> Option<&'a DatasetIndex> {
        self.store.hdf5()
    }

    /// The variable's attributes.
    pub fn attributes(&self) -> &'a [NcAttribute] {
        &self.info.attributes
    }

    /// One attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&'a NcAttribute> {
        self.info.attributes.iter().find(|a| a.name == name)
    }

    /// The variable's netCDF type. This matches `netcdf::Variable::vartype`.
    pub fn vartype(&self) -> crate::netcdf::DType {
        crate::netcdf::DType::of(self.store.datatype())
    }

    /// The element type as HDF5 records it.
    pub fn datatype(&self) -> &'a crate::hdf5::message::Datatype {
        self.store.datatype()
    }

    /// Total number of elements. This matches `netcdf::Variable::len`.
    pub fn len(&self) -> u64 {
        self.info.shape.iter().product()
    }

    /// Whether the variable holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The shape of one storage chunk, when the variable is chunked.
    ///
    /// `None` means the variable is stored contiguously and has no chunk grid.
    pub fn chunk_shape(&self) -> Option<Vec<u64>> {
        match &self.store.hdf5()?.layout {
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
        self.store.hdf5().is_none_or(|d| d.is_readable())
    }

    /// Read values as `T`, over any selection.
    ///
    /// The asynchronous twin of [`crate::netcdf::Variable::get_values`]. It
    /// takes the same selection forms and applies the same type rules.
    ///
    /// ```no_run
    /// # use oxcdf::Extents;
    /// # async fn run(var: oxcdf::AsyncVariable<'_>) -> oxcdf::Result<()> {
    /// let all = var.get_values::<f32, _>(Extents::All).await?;
    /// let block = var.get_values::<f32, _>([0..8, 10..30]).await?;
    /// # Ok(()) }
    /// ```
    pub async fn get_values<T: crate::netcdf::Element, E>(&self, extents: E) -> Result<Vec<T>>
    where
        E: TryInto<crate::extent::Extents>,
        E::Error: Into<crate::Error>,
    {
        let extents: crate::extent::Extents = extents.try_into().map_err(Into::into)?;
        let slab = extents.to_hyperslab(&self.info.path, &self.info.shape)?;
        self.read_selection(&slab).await?.get()
    }

    /// Read one value as `T`.
    ///
    /// The asynchronous twin of [`crate::netcdf::Variable::get_value`].
    pub async fn get_value<T: crate::netcdf::Element, E>(&self, extents: E) -> Result<T>
    where
        E: TryInto<crate::extent::Extents>,
        E::Error: Into<crate::Error>,
    {
        let values = self.get_values::<T, E>(extents).await?;
        crate::netcdf::one_value(&self.info.path, values)
    }

    /// Read strings, over any selection.
    ///
    /// The asynchronous twin of [`crate::netcdf::Variable::get_strings`]. A
    /// `string` variable resolves through the global heap, which this reads for
    /// itself.
    pub async fn get_strings<E>(&self, extents: E) -> Result<Vec<String>>
    where
        E: TryInto<crate::extent::Extents>,
        E::Error: Into<crate::Error>,
    {
        let extents: crate::extent::Extents = extents.try_into().map_err(Into::into)?;
        let slab = extents.to_hyperslab(&self.info.path, &self.info.shape)?;
        self.read_selection(&slab).await?.to_strings()
    }

    /// Read one string.
    ///
    /// The asynchronous twin of [`crate::netcdf::Variable::get_string`].
    pub async fn get_string<E>(&self, extents: E) -> Result<String>
    where
        E: TryInto<crate::extent::Extents>,
        E::Error: Into<crate::Error>,
    {
        let strings = self.get_strings(extents).await?;
        crate::netcdf::one_value(&self.info.path, strings)
    }

    /// Read values as an `ndarray` of `T`, over any selection.
    ///
    /// The asynchronous twin of [`crate::netcdf::Variable::get`].
    #[cfg(feature = "ndarray")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
    pub async fn get<T: crate::netcdf::Element, E>(&self, extents: E) -> Result<ndarray::ArrayD<T>>
    where
        E: TryInto<crate::extent::Extents>,
        E::Error: Into<crate::Error>,
    {
        let extents: crate::extent::Extents = extents.try_into().map_err(Into::into)?;
        let slab = extents.to_hyperslab(&self.info.path, &self.info.shape)?;
        self.read_selection(&slab).await?.to_array()
    }

    /// Read the raw bytes of a selection, in native order and row-major.
    ///
    /// The asynchronous twin of [`crate::netcdf::Variable::get_raw_values`].
    pub async fn get_raw_values<E>(&self, extents: E) -> Result<Vec<u8>>
    where
        E: TryInto<crate::extent::Extents>,
        E::Error: Into<crate::Error>,
    {
        let extents: crate::extent::Extents = extents.try_into().map_err(Into::into)?;
        let slab = extents.to_hyperslab(&self.info.path, &self.info.shape)?;
        Ok(self.read_selection(&slab).await?.into_raw().bytes)
    }

    /// Read the whole variable as [`Values`].
    ///
    /// See [`crate::netcdf::Variable::read`].
    pub async fn read(&self) -> Result<Values> {
        self.read_selection(&Hyperslab::all(&self.info.shape)).await
    }

    /// Read an explicit selection as [`Values`].
    pub async fn read_selection(&self, slab: &Hyperslab) -> Result<Values> {
        let Some(dataset) = self.store.hdf5() else {
            return self.read_classic_selection(slab).await;
        };
        let hdf5 = self
            .file
            .netcdf
            .hdf5()
            .ok_or_else(|| Error::malformed("an HDF5 variable without an HDF5 file"))?;
        self.resolve_index().await?;

        let raw = crate::read::read_hyperslab_async_with(
            self.file.source.as_ref(),
            hdf5.superblock(),
            hdf5.cache(),
            hdf5.io_cache(),
            hdf5.io(),
            dataset,
            slab,
        )
        .await?;

        // A variable-length value stores a pointer into a heap. Follow it.
        self.file
            .walk(|ctx| crate::netcdf::values_from_raw(ctx, dataset, raw.clone()))
            .await
    }

    /// Read a classic variable.
    ///
    /// A classic read is a set of byte ranges over a parsed header. The header
    /// is already in memory, so the replay driver only fetches the data pages.
    /// It holds them in the same byte cache the metadata walk filled.
    async fn read_classic_selection(&self, slab: &Hyperslab) -> Result<Values> {
        let classic = self
            .file
            .netcdf
            .classic()
            .ok_or_else(|| Error::malformed("a classic variable without a classic file"))?;
        let variable = classic
            .variables
            .iter()
            .find(|v| v.name == self.info.name)
            .ok_or_else(|| Error::not_found(format!("classic variable {}", self.info.name)))?;

        slab.validate(&self.info.shape)?;
        let bytes = self
            .file
            .walk_bytes(|source| classic.read_selection_with(source.as_ref(), variable, slab))
            .await?;

        Ok(Values::from_parts(
            crate::read::RawData {
                bytes,
                element_size: variable.nc_type.size(),
                shape: slab.count.clone(),
            },
            variable.datatype.clone(),
        ))
    }

    /// Every storage chunk of the variable.
    ///
    /// A chunk is clipped to the variable. The chunks cover it exactly once.
    /// Read them concurrently: each one is a separate byte range.
    pub async fn chunks(&self) -> Result<Vec<Chunk>> {
        self.resolve_index().await?;
        match self.store.hdf5() {
            Some(dataset) => crate::netcdf::chunks_of(dataset),
            // A classic variable is one contiguous block, so it reports one
            // chunk. A caller then uses the same loop for either container.
            None => Ok(vec![Chunk {
                offset: vec![0; self.info.shape.len()],
                shape: self.info.shape.clone(),
                stored_size: self.len() * self.store.datatype().size as u64,
            }]),
        }
    }

    /// Read one chunk.
    pub async fn read_chunk(&self, chunk: &Chunk) -> Result<Values> {
        self.read_selection(&chunk.selection()).await
    }

    /// Resolve the chunk index, if it is not resolved already.
    ///
    /// The walk is pure. Two tasks that race it both do the work. The duplicate
    /// wastes effort. It does not give a wrong answer, and it avoids a lock.
    async fn resolve_index(&self) -> Result<()> {
        let Some(dataset) = self.store.hdf5() else {
            // A classic file stores every variable contiguously. There is no
            // index to resolve.
            return Ok(());
        };
        if dataset.is_prepared() {
            return Ok(());
        }
        self.file.walk(|ctx| dataset.prepare(ctx)).await
    }
}

#[cfg(feature = "object-store")]
#[cfg_attr(docsrs, doc(cfg(feature = "object-store")))]
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
    /// let values = file.variable("TEMP").unwrap().read().await?.get::<f64>()?;
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
