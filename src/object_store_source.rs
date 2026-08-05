//! A [`ByteSource`] backed by the `object_store` crate.
//!
//! This is what makes the reader work natively against S3, GCS, Azure and plain
//! HTTP, with no local copy and no `#mode=bytes` trick. It exists because the
//! whole reader addresses bytes by absolute offset and never seeks: a netCDF-4
//! file on object storage is just a byte range server, and that is precisely
//! what [`ObjectStore::get_range`] provides.
//!
//! # Why this is a good fit
//!
//! Chunked datasets are ideal for range requests. Each chunk is an independent,
//! self-contained byte range, so reading a hyperslab means fetching only the
//! chunks it touches. [`ByteSource::read_ranges`] is overridden here to issue
//! those fetches concurrently through [`ObjectStore::get_ranges`], which also
//! coalesces neighbouring ranges. On object storage that is the difference
//! between one round trip per chunk and a handful overall.
//!
//! # The blocking bridge
//!
//! `ObjectStore` is async and [`ByteSource`] is sync, deliberately: keeping the
//! parser sync means it has no runtime dependency and can be called from a
//! thread pool, from `rayon`, or from nothing at all.
//!
//! The bridge blocks on the async call. That is only sound off a runtime worker
//! thread, so **every read through this source must happen inside
//! [`tokio::task::spawn_blocking`]** (or on any other non-worker thread).
//! That is the right shape anyway: decompressing a chunk is CPU-bound work that
//! has no business on an async worker.
//!
//! Calling this from a runtime worker thread panics rather than deadlocking
//! quietly.

use std::sync::Arc;

use object_store::path::Path;
// `head` and `get_range` live on the extension trait as of object_store 0.13;
// `get_ranges` is still on the base trait.
use object_store::{ObjectStore, ObjectStoreExt};

use crate::error::{Error, Result};
use crate::source::ByteSource;

/// A byte source that reads ranges from an object store.
#[derive(Debug, Clone)]
pub struct ObjectStoreSource {
    store: Arc<dyn ObjectStore>,
    path: Path,
    size: u64,
    handle: tokio::runtime::Handle,
}

impl ObjectStoreSource {
    /// Build a source for `path` in `store`, taking the object's size from a
    /// `head` request.
    ///
    /// Call this from async code; the reads that follow must be blocking.
    pub async fn new(store: Arc<dyn ObjectStore>, path: Path) -> Result<Self> {
        let meta = store
            .head(&path)
            .await
            .map_err(|e| Error::malformed(format!("failed to stat {path}: {e}")))?;
        Ok(Self {
            store,
            path,
            size: meta.size,
            handle: tokio::runtime::Handle::current(),
        })
    }

    /// Build a source when the object's size is already known, avoiding the
    /// extra `head` request.
    pub fn with_size(
        store: Arc<dyn ObjectStore>,
        path: Path,
        size: u64,
        handle: tokio::runtime::Handle,
    ) -> Self {
        Self {
            store,
            path,
            size,
            handle,
        }
    }

    /// Run an async call to completion from a blocking context.
    fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        self.handle.block_on(future)
    }

    fn check_range(&self, offset: u64, len: u64) -> Result<()> {
        if offset.saturating_add(len) > self.size {
            return Err(Error::OutOfBounds {
                what: "object",
                offset,
                len,
                available: self.size.saturating_sub(offset),
            });
        }
        Ok(())
    }
}

impl ByteSource for ObjectStoreSource {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let len = buf.len() as u64;
        self.check_range(offset, len)?;
        if len == 0 {
            return Ok(());
        }

        let bytes = self
            .block_on(self.store.get_range(&self.path, offset..offset + len))
            .map_err(|e| {
                Error::malformed(format!(
                    "failed to read {len} bytes at {offset} from {}: {e}",
                    self.path
                ))
            })?;

        if bytes.len() as u64 != len {
            return Err(Error::OutOfBounds {
                what: "object",
                offset,
                len,
                available: bytes.len() as u64,
            });
        }
        buf.copy_from_slice(&bytes);
        Ok(())
    }

    /// Fetch many ranges in one go.
    ///
    /// `get_ranges` coalesces ranges that are close together and issues the
    /// rest concurrently, which is the single biggest win when reading a
    /// chunked dataset over the network.
    fn read_ranges(&self, ranges: &[(u64, usize)]) -> Result<Vec<Vec<u8>>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        for &(offset, len) in ranges {
            self.check_range(offset, len as u64)?;
        }

        let wanted: Vec<std::ops::Range<u64>> = ranges
            .iter()
            .map(|&(offset, len)| offset..offset + len as u64)
            .collect();

        let fetched = self
            .block_on(self.store.get_ranges(&self.path, &wanted))
            .map_err(|e| {
                Error::malformed(format!(
                    "failed to read {} ranges from {}: {e}",
                    ranges.len(),
                    self.path
                ))
            })?;

        Ok(fetched.into_iter().map(|b| b.to_vec()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Hdf5File;
    use crate::read::{read_hyperslab, Hyperslab};
    use object_store::memory::InMemory;
    use object_store::PutPayload;

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_files/legacy_v1_objheader.h5"
    );

    /// Serve the fixture out of an in-memory object store and read it through
    /// the same code path S3 would use. Values must match the local read.
    #[tokio::test(flavor = "multi_thread")]
    async fn reads_a_dataset_through_an_object_store() {
        let bytes = std::fs::read(FIXTURE).unwrap();
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("fixture.h5");
        store
            .put(&path, PutPayload::from(bytes))
            .await
            .unwrap();

        let source = ObjectStoreSource::new(store, path).await.unwrap();

        // Reads block, so they belong on a blocking thread.
        let values = tokio::task::spawn_blocking(move || {
            let file = Hdf5File::from_source(Arc::new(source)).unwrap();
            let d = file.dataset("/chunked_i32").unwrap();
            read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
                .unwrap()
                .get::<i64>(d)
                .unwrap()
        })
        .await
        .unwrap();

        assert_eq!(values.len(), 40 * 6);
        for (i, v) in values.iter().enumerate() {
            assert_eq!(*v, i as i64 * 3 - 100, "element {i}");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_a_range_past_the_end_of_the_object() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("small.bin");
        store
            .put(&path, PutPayload::from(vec![1u8, 2, 3, 4]))
            .await
            .unwrap();

        let source = ObjectStoreSource::new(store, path).await.unwrap();
        assert_eq!(source.size(), 4);

        tokio::task::spawn_blocking(move || {
            assert!(source.read_vec(2, 8).is_err());
            assert_eq!(source.read_vec(1, 2).unwrap(), vec![2, 3]);
        })
        .await
        .unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn batched_ranges_come_back_in_request_order() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("ranges.bin");
        let data: Vec<u8> = (0..64).collect();
        store
            .put(&path, PutPayload::from(data))
            .await
            .unwrap();

        let source = ObjectStoreSource::new(store, path).await.unwrap();
        tokio::task::spawn_blocking(move || {
            let got = source.read_ranges(&[(10, 3), (0, 2), (60, 4)]).unwrap();
            assert_eq!(got[0], vec![10, 11, 12]);
            assert_eq!(got[1], vec![0, 1]);
            assert_eq!(got[2], vec![60, 61, 62, 63]);
        })
        .await
        .unwrap();
    }
}
