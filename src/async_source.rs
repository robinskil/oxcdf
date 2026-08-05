//! Byte sources for asynchronous access.
//!
//! This is the fetch half of the async engine. It sits beside [`crate::source::ByteSource`].
//! It does not replace it. Both engines share every pure part of the crate.
//!
//! # Why a separate trait
//!
//! [`crate::source::ByteSource`] is synchronous by design. The parser then needs no runtime.
//! It runs under `rayon` or under a plain thread pool.
//!
//! An async [`crate::source::ByteSource`] would move inflate and unshuffle onto runtime
//! workers. Those steps use the processor. They would block the reactor.
//!
//! # Why not the blocking adapter
//!
//! [`crate::object_store_source::ObjectStoreSource`] blocks on its futures. A
//! block is only safe off a runtime worker. Every read then needs
//! `spawn_blocking`.
//!
//! An implementation of this trait does not block. It works on a
//! current-thread runtime. It fetches the ranges of one read together.

use std::sync::Arc;

use bytes::Bytes;

use crate::error::Result;
// Only the object-store source raises errors of its own. Everything else here
// forwards what the inner source returns.
#[cfg(any(feature = "object-store", test))]
use crate::error::Error;

/// A source of bytes addressed by absolute file offset, fetched asynchronously.
///
/// The mirror of [`crate::source::ByteSource`]. Implementations must be safe to
/// call concurrently; like the sync trait, there is no cursor to race on.
#[async_trait::async_trait]
pub trait AsyncByteSource: Send + Sync + std::fmt::Debug {
    /// Total size of the source in bytes.
    fn size(&self) -> u64;

    /// Read `len` bytes at `offset`.
    async fn read_at(&self, offset: u64, len: usize) -> Result<Bytes>;

    /// Read several ranges.
    ///
    /// The default runs them in sequence. Implementations backed by a network
    /// should override this to coalesce and pipeline, which is where nearly all
    /// of the win is: a chunked read asks for many ranges at once, and issuing
    /// them one at a time turns one round trip into dozens.
    async fn read_ranges(&self, ranges: &[(u64, usize)]) -> Result<Vec<Bytes>> {
        let mut out = Vec::with_capacity(ranges.len());
        for &(offset, len) in ranges {
            out.push(self.read_at(offset, len).await?);
        }
        Ok(out)
    }
}

#[async_trait::async_trait]
impl AsyncByteSource for Arc<dyn AsyncByteSource> {
    fn size(&self) -> u64 {
        (**self).size()
    }
    async fn read_at(&self, offset: u64, len: usize) -> Result<Bytes> {
        (**self).read_at(offset, len).await
    }
    async fn read_ranges(&self, ranges: &[(u64, usize)]) -> Result<Vec<Bytes>> {
        (**self).read_ranges(ranges).await
    }
}

/// Presents any synchronous [`crate::source::ByteSource`] as an asynchronous one.
///
/// Reads complete immediately rather than yielding, so this is for sources where
/// that is honest: an in-memory buffer, or a local file whose pages are cached.
/// It is **not** a way to put a slow blocking source behind an async interface —
/// that would stall the runtime, which is exactly what the async engine exists
/// to avoid. For a real file under load, read it on a blocking thread or use a
/// natively async source.
#[derive(Debug)]
pub struct SyncAsAsync<S: crate::source::ByteSource>(pub S);

#[async_trait::async_trait]
impl<S: crate::source::ByteSource + 'static> AsyncByteSource for SyncAsAsync<S> {
    fn size(&self) -> u64 {
        self.0.size()
    }

    async fn read_at(&self, offset: u64, len: usize) -> Result<Bytes> {
        Ok(Bytes::from(self.0.read_vec(offset, len)?))
    }

    async fn read_ranges(&self, ranges: &[(u64, usize)]) -> Result<Vec<Bytes>> {
        Ok(self
            .0
            .read_ranges(ranges)?
            .into_iter()
            .map(Bytes::from)
            .collect())
    }
}

/// An asynchronous byte source backed by the `object_store` crate.
///
/// Unlike the blocking adapter this needs no runtime handle, never blocks, and
/// works on a current-thread runtime.
#[cfg(feature = "object-store")]
#[cfg_attr(docsrs, doc(cfg(feature = "object-store")))]
#[derive(Debug, Clone)]
pub struct AsyncObjectStoreSource {
    store: Arc<dyn object_store::ObjectStore>,
    path: object_store::path::Path,
    size: u64,
}

#[cfg(feature = "object-store")]
#[cfg_attr(docsrs, doc(cfg(feature = "object-store")))]
impl AsyncObjectStoreSource {
    /// Build a source, taking the object's size from a `head` request.
    pub async fn new(
        store: Arc<dyn object_store::ObjectStore>,
        path: object_store::path::Path,
    ) -> Result<Self> {
        use object_store::ObjectStoreExt;
        let meta = store
            .head(&path)
            .await
            .map_err(|e| Error::malformed(format!("failed to stat {path}: {e}")))?;
        Ok(Self {
            store,
            path,
            size: meta.size,
        })
    }

    /// Build a source when the size is already known, skipping the `head`.
    pub fn with_size(
        store: Arc<dyn object_store::ObjectStore>,
        path: object_store::path::Path,
        size: u64,
    ) -> Self {
        Self { store, path, size }
    }

    fn check(&self, offset: u64, len: u64) -> Result<()> {
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

#[cfg(feature = "object-store")]
#[cfg_attr(docsrs, doc(cfg(feature = "object-store")))]
#[async_trait::async_trait]
impl AsyncByteSource for AsyncObjectStoreSource {
    fn size(&self) -> u64 {
        self.size
    }

    async fn read_at(&self, offset: u64, len: usize) -> Result<Bytes> {
        use object_store::ObjectStoreExt;
        self.check(offset, len as u64)?;
        if len == 0 {
            return Ok(Bytes::new());
        }
        let bytes = self
            .store
            .get_range(&self.path, offset..offset + len as u64)
            .await
            .map_err(|e| {
                Error::malformed(format!(
                    "failed to read {len} bytes at {offset} from {}: {e}",
                    self.path
                ))
            })?;
        if bytes.len() != len {
            return Err(Error::OutOfBounds {
                what: "object",
                offset,
                len: len as u64,
                available: bytes.len() as u64,
            });
        }
        Ok(bytes)
    }

    /// Fetch many ranges in one call.
    ///
    /// `get_ranges` coalesces neighbouring ranges and issues the rest
    /// concurrently. This is the method the chunked read path leans on.
    async fn read_ranges(&self, ranges: &[(u64, usize)]) -> Result<Vec<Bytes>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        for &(offset, len) in ranges {
            self.check(offset, len as u64)?;
        }
        let wanted: Vec<std::ops::Range<u64>> = ranges
            .iter()
            .map(|&(offset, len)| offset..offset + len as u64)
            .collect();

        let fetched = self
            .store
            .get_ranges(&self.path, &wanted)
            .await
            .map_err(|e| {
                Error::malformed(format!(
                    "failed to read {} ranges from {}: {e}",
                    ranges.len(),
                    self.path
                ))
            })?;
        // `object_store` already returns `Bytes`; passing them through keeps
        // the fetch zero-copy.
        Ok(fetched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::MemorySource;

    #[tokio::test]
    async fn a_sync_source_can_be_presented_as_async() {
        let src = SyncAsAsync(MemorySource::new((0..64u8).collect()));
        assert_eq!(src.size(), 64);
        assert_eq!(&src.read_at(8, 4).await.unwrap()[..], &[8, 9, 10, 11]);

        let got = src.read_ranges(&[(0, 2), (60, 4)]).await.unwrap();
        assert_eq!(&got[0][..], &[0, 1]);
        assert_eq!(&got[1][..], &[60, 61, 62, 63]);
    }

    #[tokio::test]
    async fn out_of_bounds_is_reported_the_same_way_as_the_sync_trait() {
        let src = SyncAsAsync(MemorySource::new(vec![1, 2, 3, 4]));
        let err = src.read_at(2, 8).await.unwrap_err();
        assert!(matches!(err, Error::OutOfBounds { .. }), "got {err:?}");
    }

    #[tokio::test]
    async fn the_trait_is_object_safe_and_shareable() {
        let src: Arc<dyn AsyncByteSource> =
            Arc::new(SyncAsAsync(MemorySource::new(vec![7; 16])));
        // Cloning the Arc across tasks is the usage that matters.
        let a = Arc::clone(&src);
        let handle = tokio::spawn(async move { a.read_at(0, 4).await });
        assert_eq!(&handle.await.unwrap().unwrap()[..], &[7, 7, 7, 7]);
    }

    #[cfg(feature = "object-store")]
    #[cfg_attr(docsrs, doc(cfg(feature = "object-store")))]
    #[tokio::test]
    async fn reads_ranges_from_an_object_store_without_blocking() {
        use object_store::memory::InMemory;
        use object_store::{ObjectStore, ObjectStoreExt, PutPayload};

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = object_store::path::Path::from("bytes.bin");
        let data: Vec<u8> = (0..128u8).collect();
        store.put(&path, PutPayload::from(data)).await.unwrap();

        let src = AsyncObjectStoreSource::new(store, path).await.unwrap();
        assert_eq!(src.size(), 128);

        // No spawn_blocking, no runtime handle: this is a plain await.
        assert_eq!(&src.read_at(10, 3).await.unwrap()[..], &[10, 11, 12]);

        let got = src.read_ranges(&[(100, 4), (0, 2), (64, 1)]).await.unwrap();
        assert_eq!(&got[0][..], &[100, 101, 102, 103]);
        assert_eq!(&got[1][..], &[0, 1]);
        assert_eq!(&got[2][..], &[64]);
    }

    /// The whole point: this works on a current-thread runtime, which the
    /// blocking adapter cannot do.
    #[cfg(feature = "object-store")]
    #[cfg_attr(docsrs, doc(cfg(feature = "object-store")))]
    #[test]
    fn works_on_a_current_thread_runtime() {
        use object_store::memory::InMemory;
        use object_store::{ObjectStore, ObjectStoreExt, PutPayload};

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
            let path = object_store::path::Path::from("x.bin");
            store
                .put(&path, PutPayload::from(vec![1u8, 2, 3, 4]))
                .await
                .unwrap();

            let src = AsyncObjectStoreSource::new(store, path).await.unwrap();
            assert_eq!(&src.read_at(1, 2).await.unwrap()[..], &[2, 3]);
        });
    }
}
