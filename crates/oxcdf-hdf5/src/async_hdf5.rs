//! The asynchronous HDF5 interface.
//!
//! This module mirrors [`crate::index`]. An open awaits. A read of values
//! awaits. Every other call answers at once.
//!
//! ```no_run
//! # async fn run(source: std::sync::Arc<dyn oxcdf_hdf5::AsyncByteSource>)
//! # -> oxcdf_hdf5::Result<()> {
//! let file = oxcdf_hdf5::AsyncHdf5File::open(source).await?;
//!
//! let temp = file.dataset("/TEMP").unwrap();
//! println!("{:?} {:?}", temp.shape, temp.datatype.class);
//!
//! let all = temp.read().await?.get::<f64>(&temp)?;
//! # Ok(()) }
//! ```
//!
//! The open runs the synchronous walk over pages in memory. It fetches the
//! pages it lacks, then runs the walk again. See [`crate::replay`].
//!
//! A chunk index resolves the same way, on first use. A caller never calls
//! [`AsyncDataset::prepare`].
//!
//! For a netCDF file, use `AsyncNetcdfFile` from the `oxcdf` crate. It adds
//! dimensions, named variables and typed attributes.

use std::sync::Arc;

use crate::async_source::AsyncByteSource;
use crate::context::Ctx;
use crate::error::Result;
use crate::index::{DatasetIndex, GroupIndex, Hdf5File, OpenOptions};
use crate::read::{Chunk, Hyperslab, RawData};

/// Run a synchronous HDF5 walk over an asynchronous source.
///
/// `build` receives a context over pages held in memory. A read outside those
/// pages makes the driver fetch them and run `build` again, so `build` must be
/// pure. See [`crate::replay`].
///
/// This is the primitive behind [`AsyncHdf5File::walk`]. It is public because
/// the netCDF layer above drives its own walks with it, and because a caller
/// that reaches metadata this crate does not model needs the same door.
pub async fn walk<T, F>(
    source: &dyn AsyncByteSource,
    hdf5: &Hdf5File,
    page_size: usize,
    build: F,
) -> Result<T>
where
    F: Fn(Ctx<'_>) -> Result<T>,
{
    crate::replay::replay(
        source,
        hdf5.io_cache(),
        page_size,
        // The metadata this reaches is scattered, not clustered at the front,
        // so read the page that was asked for and little else.
        page_size,
        |bytes| {
            let ctx = Ctx::new(bytes.as_ref(), hdf5.superblock())
                .with_cache(hdf5.cache())
                .with_io(hdf5.io());
            build(ctx)
        },
    )
    .await
}

/// An HDF5 file opened over an asynchronous byte source.
///
/// This is the asynchronous counterpart of [`Hdf5File`].
///
/// The metadata is complete when the open returns. A metadata method therefore
/// takes no `await`. A read of values takes one.
///
/// This type is `Send + Sync`. Share it behind an [`Arc`]. Concurrent reads
/// need no coordination.
#[derive(Debug, Clone)]
pub struct AsyncHdf5File {
    hdf5: Hdf5File,
    source: Arc<dyn AsyncByteSource>,
    options: OpenOptions,
}

impl AsyncHdf5File {
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
        let hdf5 = crate::replay::replay(
            source.as_ref(),
            io_cache.as_ref(),
            page_size,
            options.prefetch_bytes(),
            Hdf5File::from_source,
        )
        .await?;

        // Give the file the caches the reads need. The open built its own index
        // over a private source, so replace that too.
        let hdf5 = hdf5
            .with_io_cache(io_cache)
            .with_cache(options.build_chunk_cache())
            .with_io(options.merge_policy());

        Ok(Self {
            hdf5,
            source,
            options,
        })
    }

    /// The parsed file.
    ///
    /// Its metadata is complete. Its byte source is the pages the open held,
    /// not the file, so do not read values through it. A synchronous read
    /// returns [`crate::Error::Incomplete`] when it reaches a page the open did
    /// not fetch. It never blocks, and it never returns wrong bytes.
    ///
    /// A read can also succeed, because the open window often covers a whole
    /// small file. Do not rely on either outcome. Read through
    /// [`AsyncDataset`].
    pub fn hdf5(&self) -> &Hdf5File {
        &self.hdf5
    }

    /// The byte source this file reads from.
    pub fn source(&self) -> &Arc<dyn AsyncByteSource> {
        &self.source
    }

    /// The options this file was opened with.
    pub fn options(&self) -> &OpenOptions {
        &self.options
    }

    /// The root group.
    pub fn root(&self) -> &GroupIndex {
        self.hdf5.root()
    }

    /// One dataset by path, such as `/forecast/TEMP`.
    pub fn dataset(&self, path: &str) -> Option<AsyncDataset<'_>> {
        self.hdf5
            .dataset(path)
            .map(|index| AsyncDataset { file: self, index })
    }

    /// Every dataset in the file, depth first.
    pub fn datasets(&self) -> Vec<AsyncDataset<'_>> {
        self.hdf5
            .datasets()
            .into_iter()
            .map(|index| AsyncDataset { file: self, index })
            .collect()
    }

    /// Run a synchronous HDF5 walk over the file, fetching whatever it needs.
    ///
    /// Use this for metadata this crate does not model. `build` must be pure:
    /// the driver discards the result of every round but the last.
    pub async fn walk<T, F>(&self, build: F) -> Result<T>
    where
        F: Fn(Ctx<'_>) -> Result<T>,
    {
        walk(
            self.source.as_ref(),
            &self.hdf5,
            self.options.request_size(),
            build,
        )
        .await
    }
}

#[cfg(feature = "object-store")]
#[cfg_attr(docsrs, doc(cfg(feature = "object-store")))]
impl AsyncHdf5File {
    /// Open a file held in object storage.
    ///
    /// This uses [`OpenOptions::remote`]. The reader reads byte ranges. It
    /// needs no local copy.
    ///
    /// ```no_run
    /// # async fn run(
    /// #     store: std::sync::Arc<dyn object_store::ObjectStore>,
    /// # ) -> oxcdf_hdf5::Result<()> {
    /// use object_store::path::Path;
    ///
    /// let file = oxcdf_hdf5::AsyncHdf5File::open_store(store, Path::from("a.h5")).await?;
    /// let temp = file.dataset("/TEMP").unwrap();
    /// let values = temp.read().await?.get::<f64>(&temp)?;
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

/// A readable handle on one dataset.
///
/// Dereferences to [`DatasetIndex`], so `dataset.shape`, `dataset.datatype` and
/// the rest are available directly.
#[derive(Clone, Copy)]
pub struct AsyncDataset<'a> {
    file: &'a AsyncHdf5File,
    index: &'a DatasetIndex,
}

impl std::fmt::Debug for AsyncDataset<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncDataset")
            .field("path", &self.index.path)
            .field("shape", &self.index.shape)
            .finish()
    }
}

impl std::ops::Deref for AsyncDataset<'_> {
    type Target = DatasetIndex;

    fn deref(&self) -> &Self::Target {
        self.index
    }
}

impl<'a> AsyncDataset<'a> {
    /// The parsed dataset.
    pub fn index(&self) -> &'a DatasetIndex {
        self.index
    }

    /// The file this dataset belongs to.
    pub fn file(&self) -> &'a AsyncHdf5File {
        self.file
    }

    /// Read the whole dataset.
    pub async fn read(&self) -> Result<RawData> {
        self.read_selection(&Hyperslab::all(&self.index.shape))
            .await
    }

    /// Read an explicit selection.
    pub async fn read_selection(&self, slab: &Hyperslab) -> Result<RawData> {
        self.prepare().await?;
        let hdf5 = &self.file.hdf5;
        crate::read::read_hyperslab_async_with(
            self.file.source.as_ref(),
            hdf5.superblock(),
            hdf5.cache(),
            hdf5.io_cache(),
            hdf5.io(),
            self.index,
            slab,
        )
        .await
    }

    /// Every stored chunk of the dataset, clipped to its bounds.
    ///
    /// A chunk is a separate byte range with its own filters. Read chunks
    /// concurrently: nothing is shared between them.
    ///
    /// A dataset that is not chunked reports one chunk covering everything, so
    /// a caller uses one loop either way.
    pub async fn chunks(&self) -> Result<Vec<Chunk>> {
        self.prepare().await?;
        crate::read::chunks_of(self.index)
    }

    /// Read one chunk.
    pub async fn read_chunk(&self, chunk: &Chunk) -> Result<RawData> {
        self.read_selection(&chunk.selection()).await
    }

    /// Resolve the chunk index, if it is not resolved already.
    ///
    /// A read does this itself. Call it to move the cost off the read path.
    ///
    /// The walk is pure. Two tasks that race it both do the work. The duplicate
    /// wastes effort. It does not give a wrong answer, and it avoids a lock.
    pub async fn prepare(&self) -> Result<()> {
        if self.index.is_prepared() {
            return Ok(());
        }
        self.file.walk(|ctx| self.index.prepare(ctx)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::FileSource;

    fn corpus() -> Vec<String> {
        crate::test_corpus::paths()
    }

    async fn open(path: &str) -> AsyncHdf5File {
        let source = Arc::new(crate::async_source::SyncAsAsync(
            FileSource::open(path).unwrap(),
        ));
        AsyncHdf5File::open(source).await.unwrap()
    }

    #[tokio::test]
    async fn the_two_opens_find_the_same_datasets() {
        for path in corpus() {
            let sync = Hdf5File::open(&path).unwrap();
            let file = open(&path).await;

            let mut want: Vec<&str> = sync.datasets().iter().map(|d| d.path.as_str()).collect();
            let bound = file.datasets();
            let mut got: Vec<&str> = bound.iter().map(|d| d.path.as_str()).collect();
            want.sort_unstable();
            got.sort_unstable();
            assert_eq!(got, want, "{path}");
        }
    }

    #[tokio::test]
    async fn the_two_engines_read_the_same_bytes() {
        for path in corpus() {
            let sync = Hdf5File::open(&path).unwrap();
            let file = open(&path).await;

            for want in sync.datasets() {
                if !want.is_readable() {
                    continue;
                }
                want.prepare(sync.ctx()).unwrap();
                let slab = Hyperslab::all(&want.shape);
                let Ok(expected) = crate::read::read_hyperslab(sync.ctx(), want, &slab) else {
                    continue;
                };

                let got = file.dataset(&want.path).unwrap().read().await.unwrap();
                assert_eq!(got.bytes, expected.bytes, "{} in {path}", want.path);
                assert_eq!(got.shape, expected.shape, "{} in {path}", want.path);
            }
        }
    }

    /// Chunk enumeration and per-chunk reads must agree between the engines.
    ///
    /// The netCDF layer has no chunk API, so this is the only place the two
    /// engines are compared chunk by chunk.
    #[tokio::test]
    async fn chunks_match_the_synchronous_engine() {
        for path in corpus() {
            let sync = Hdf5File::open(&path).unwrap();
            let file = open(&path).await;

            for want in sync.datasets() {
                if !want.is_readable() || want.shape.is_empty() {
                    continue;
                }
                want.prepare(sync.ctx()).unwrap();
                let Ok(want_chunks) = crate::read::chunks_of(want) else {
                    continue;
                };

                let got = file.dataset(&want.path).unwrap();
                let got_chunks = got.chunks().await.unwrap();
                assert_eq!(got_chunks, want_chunks, "chunks of {} in {path}", want.path);

                for chunk in got_chunks.iter().take(3) {
                    let expected =
                        crate::read::read_hyperslab(sync.ctx(), want, &chunk.selection()).unwrap();
                    assert_eq!(
                        got.read_chunk(chunk).await.unwrap().bytes,
                        expected.bytes,
                        "chunk {:?} of {} in {path}",
                        chunk.offset,
                        want.path
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn a_dataset_dereferences_to_its_index() {
        let path = corpus().into_iter().next().unwrap();
        let file = open(&path).await;
        let d = file.datasets().into_iter().next().unwrap();
        // Through `Deref`, with no method of its own.
        assert_eq!(d.rank(), d.shape.len());
        assert!(!d.path.is_empty());
    }

    #[tokio::test]
    async fn chunks_cover_the_dataset_exactly_once() {
        for path in corpus() {
            let file = open(&path).await;
            for d in file.datasets() {
                if !d.is_readable() || d.shape.is_empty() {
                    continue;
                }
                let Ok(chunks) = d.chunks().await else {
                    continue;
                };
                let covered: u64 = chunks.iter().map(|c| c.element_count()).sum();
                assert_eq!(covered, d.element_count(), "{} in {path}", d.path);
            }
        }
    }
}
