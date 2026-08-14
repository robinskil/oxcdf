//! The asynchronous netCDF interface.
//!
//! This module mirrors [`crate::netcdf`]. An open awaits. A read of values
//! awaits. Every other call answers at once.
//!
//! ```no_run
//! # async fn run(source: std::sync::Arc<dyn oxcdf::AsyncByteSource>) -> oxcdf::Result<()> {
//! let file = oxcdf::open_async(source).await?;
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
//! The open runs the synchronous walk over pages in memory. It fetches the
//! pages it lacks, then runs the walk again. See [`oxcdf_hdf5::replay`].
//!
//! A chunk index and a string heap resolve the same way, on first use.

use std::sync::Arc;

use crate::netcdf::{NcAttribute, NcDimension, NcGroup, NcVariable, NetcdfFile, Values};
use oxcdf_hdf5::async_source::AsyncByteSource;
use oxcdf_hdf5::context::Ctx;
use oxcdf_hdf5::error::{Error, Result};
use oxcdf_hdf5::index::{DatasetIndex, OpenOptions};
use oxcdf_hdf5::read::Hyperslab;

/// A netCDF file opened over an asynchronous byte source.
///
/// This is the asynchronous counterpart of [`NetcdfFile`]. The open reads the
/// magic bytes, so it takes netCDF-4 and netCDF classic alike.
///
/// The metadata is complete when the open returns. A metadata method therefore
/// takes no `await`. A read of values takes one.
///
/// This type is `Send + Sync`. Share it behind an [`Arc`]. Concurrent reads
/// need no coordination.
#[derive(Debug, Clone)]
pub struct AsyncNetcdfFile {
    netcdf: NetcdfFile,
    source: Arc<dyn AsyncByteSource>,
    options: OpenOptions,
}

impl AsyncNetcdfFile {
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
        let netcdf = oxcdf_hdf5::replay::replay(
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
    pub fn container(&self) -> oxcdf_hdf5::Container {
        self.netcdf.container()
    }

    /// The parsed file, for the metadata this layer does not model.
    ///
    /// Its metadata is complete. Its byte source is the pages the open held,
    /// not the file. Do not read values through it.
    ///
    /// A synchronous read returns [`oxcdf_hdf5::Error::Incomplete`] for a page
    /// the open did not fetch. It never blocks. It never returns wrong bytes.
    ///
    /// The read can also succeed, because the open window often covers a small
    /// file. Rely on neither outcome. Read through [`AsyncVariable`].
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
        F: Fn(Arc<dyn oxcdf_hdf5::source::ByteSource>) -> Result<T>,
    {
        oxcdf_hdf5::replay::replay(
            self.source.as_ref(),
            self.netcdf.hdf5().and_then(|h| h.io_cache()),
            self.options.request_size(),
            self.options.request_size(),
            build,
        )
        .await
    }

    /// Run a synchronous HDF5 walk over the file, fetching whatever it needs.
    ///
    /// The lower crate owns this primitive, so both async handles drive the
    /// replay engine the same way.
    async fn walk<T, F>(&self, build: F) -> Result<T>
    where
        F: Fn(Ctx<'_>) -> Result<T>,
    {
        let hdf5 = self
            .netcdf
            .hdf5()
            .ok_or_else(|| Error::malformed("an HDF5 walk on a classic file"))?;
        oxcdf_hdf5::async_hdf5::walk(
            self.source.as_ref(),
            hdf5,
            self.options.request_size(),
            build,
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
    file: &'a AsyncNetcdfFile,
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
    pub fn datatype(&self) -> &'a oxcdf_hdf5::message::Datatype {
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
    /// This matches `netcdf::Variable::chunking`. `None` means the variable is
    /// stored contiguously and has no chunk grid.
    pub fn chunking(&self) -> Result<Option<Vec<usize>>> {
        let Some(dataset) = self.store.hdf5() else {
            // A classic file has no chunking.
            return Ok(None);
        };
        Ok(match &dataset.layout {
            oxcdf_hdf5::message::Layout::Chunked { chunk_dims, .. } => {
                Some(chunk_dims.iter().map(|&d| d as usize).collect())
            }
            _ => None,
        })
    }

    /// The variable's fill value, as `T`.
    ///
    /// This matches `netcdf::Variable::fill_value`. It needs no `await`: a fill
    /// value is metadata, and the open already read it.
    pub fn fill_value<T: crate::netcdf::Element>(&self) -> Result<Option<T>> {
        crate::netcdf::fill_value_of(self.attribute("_FillValue"))
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
        E::Error: Into<oxcdf_hdf5::Error>,
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
        E::Error: Into<oxcdf_hdf5::Error>,
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
        E::Error: Into<oxcdf_hdf5::Error>,
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
        E::Error: Into<oxcdf_hdf5::Error>,
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
        E::Error: Into<oxcdf_hdf5::Error>,
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
        E::Error: Into<oxcdf_hdf5::Error>,
    {
        let extents: crate::extent::Extents = extents.try_into().map_err(Into::into)?;
        let slab = extents.to_hyperslab(&self.info.path, &self.info.shape)?;
        Ok(self.read_selection(&slab).await?.into_raw().bytes)
    }

    /// Read an explicit selection.
    ///
    /// This is the engine behind every `get_*` method. It is private because
    /// the public interface mirrors the `netcdf` crate, which has no such
    /// call.
    pub(crate) async fn read_selection(&self, slab: &Hyperslab) -> Result<Values> {
        let Some(dataset) = self.store.hdf5() else {
            return self.read_classic_selection(slab).await;
        };
        let hdf5 = self
            .file
            .netcdf
            .hdf5()
            .ok_or_else(|| Error::malformed("an HDF5 variable without an HDF5 file"))?;
        self.resolve_index().await?;

        let raw = oxcdf_hdf5::read::read_hyperslab_async_with(
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
        //
        // Only that case needs the file again. The walk is a replay, so it
        // takes the block by value and may run its closure more than once;
        // asking for it unconditionally copied every byte just read, which was
        // 3.3% of the profile in issue #2.
        if !crate::netcdf::holds_heap_pointers(dataset) {
            return Ok(Values::from_parts(raw, dataset.datatype.clone()));
        }

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
            oxcdf_hdf5::read::RawData {
                bytes,
                element_size: variable.nc_type.size(),
                shape: slab.count.clone(),
            },
            variable.datatype.clone(),
        ))
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
impl AsyncNetcdfFile {
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
    /// let file = oxcdf::AsyncNetcdfFile::open_store(store, Path::from("13857_prof.nc")).await?;
    /// let values = file.variable("TEMP").unwrap().get_values::<f64, _>(..).await?;
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
        let source = oxcdf_hdf5::async_source::AsyncObjectStoreSource::new(store, path).await?;
        Self::open_with(Arc::new(source), options).await
    }
}
