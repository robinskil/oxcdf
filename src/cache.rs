//! Caches for decoded chunks and for raw file bytes.
//!
//! # Two caches
//!
//! [`ChunkCache`] holds decoded chunks. A decode costs one read, one inflate
//! and one unshuffle. Two reads of one chunk pay that cost once.
//!
//! [`IoCache`] sits below it. It holds raw file bytes in pages of a fixed size.
//! A file makes many small reads that cluster together. Each read is a separate
//! request. On object storage a request costs milliseconds.
//!
//! # No locks
//!
//! Both caches use [`moka`]. A cache hit takes no lock. Readers do not block
//! each other. A lock here would return the contention that this crate removes.
//!
//! Two threads may decode one cold entry twice. That is deliberate. The work is
//! pure, so a repeat wastes processor time only. A lock would prevent the
//! repeat and cost more.

use bytes::Bytes;
use std::sync::atomic::{AtomicU64, Ordering};

/// Default number of decoded chunks to keep.
///
/// Chunks vary in size, so this bounds the count rather than the bytes. A
/// netcdf-c chunk cache defaults to 1 MB per variable; this is a similar order
/// for typical chunk sizes.
pub const DEFAULT_CAPACITY: u64 = 512;

/// Counter for cache identity, so two caches never collide.
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// A cache of decoded chunk bytes, keyed by the chunk's address in the file.
///
/// One cache belongs to one open file, so the address alone identifies a chunk.
/// Cloning shares the underlying cache; `moka` is reference-counted internally.
#[derive(Clone, Debug)]
pub struct ChunkCache {
    inner: moka::sync::Cache<u64, Bytes>,
    id: u64,
    readahead: usize,
}

/// How many chunks past the ones a read needs to fetch and decode eagerly.
///
/// Scans walk chunks in order, so the chunks just past the current selection
/// are the ones most likely wanted next. Fetching them in the same batched I/O
/// call costs one round trip instead of one per chunk later, which is the
/// difference that matters on object storage.
pub const DEFAULT_READAHEAD: usize = 4;

impl ChunkCache {
    /// Build a cache holding up to `capacity` decoded chunks.
    pub fn new(capacity: u64) -> Self {
        Self {
            inner: moka::sync::Cache::new(capacity),
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            readahead: DEFAULT_READAHEAD,
        }
    }

    /// Set how many chunks past a read's own to prefetch. Zero disables it.
    pub fn with_readahead(mut self, chunks: usize) -> Self {
        self.readahead = chunks;
        self
    }

    /// How many chunks are prefetched past each read.
    pub fn readahead(&self) -> usize {
        self.readahead
    }

    /// Whether a chunk is already decoded and held.
    pub fn contains(&self, address: u64) -> bool {
        self.inner.contains_key(&address)
    }

    /// Store an already-decoded chunk.
    pub fn insert(&self, address: u64, bytes: Bytes) {
        self.inner.insert(address, bytes);
    }

    /// Fetch a decoded chunk if it is already held, without decoding.
    pub fn get(&self, address: u64) -> Option<Bytes> {
        self.inner.get(&address)
    }

    /// A cache with the default capacity.
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }

    /// This cache's identity, distinct from every other cache.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Fetch a decoded chunk, decoding it with `decode` on a miss.
    ///
    /// `decode` may run more than once for the same address if two threads race
    /// a cold entry. It must therefore be pure, which chunk decoding is.
    pub fn get_or_decode<F>(&self, address: u64, decode: F) -> crate::Result<Bytes>
    where
        F: FnOnce() -> crate::Result<Vec<u8>>,
    {
        if let Some(hit) = self.inner.get(&address) {
            return Ok(hit);
        }
        // `Bytes` clones by refcount, so handing the caller and the cache the
        // same buffer costs nothing.
        let decoded = Bytes::from(decode()?);
        self.inner.insert(address, decoded.clone());
        Ok(decoded)
    }

    /// Number of entries currently held. Approximate, as eviction is deferred.
    pub fn len(&self) -> u64 {
        self.inner.entry_count()
    }

    /// Whether the cache currently holds nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop every entry.
    pub fn clear(&self) {
        self.inner.invalidate_all();
    }
}

impl Default for ChunkCache {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn a_miss_decodes_and_a_hit_does_not() {
        let cache = ChunkCache::new(8);
        let calls = AtomicUsize::new(0);

        let first = cache
            .get_or_decode(100, || {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(vec![1, 2, 3])
            })
            .unwrap();
        assert_eq!(&first[..], &[1, 2, 3]);
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let second = cache.get_or_decode(100, || panic!("must not decode again")).unwrap();
        assert_eq!(&second[..], &[1, 2, 3]);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn different_addresses_are_separate_entries() {
        let cache = ChunkCache::new(8);
        cache.get_or_decode(1, || Ok(vec![b'a'])).unwrap();
        cache.get_or_decode(2, || Ok(vec![b'b'])).unwrap();
        assert_eq!(&cache.get_or_decode(1, || panic!()).unwrap()[..], b"a");
        assert_eq!(&cache.get_or_decode(2, || panic!()).unwrap()[..], b"b");
    }

    #[test]
    fn a_decode_error_is_not_cached() {
        let cache = ChunkCache::new(8);
        assert!(cache
            .get_or_decode(7, || Err(crate::Error::malformed("boom")))
            .is_err());
        // The next attempt must be allowed to try again.
        assert_eq!(&cache.get_or_decode(7, || Ok(vec![9])).unwrap()[..], &[9]);
    }

    #[test]
    fn caches_have_distinct_identities() {
        let a = ChunkCache::new(4);
        let b = ChunkCache::new(4);
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn clearing_drops_entries() {
        let cache = ChunkCache::new(8);
        cache.get_or_decode(1, || Ok(vec![1])).unwrap();
        cache.clear();
        cache.inner.run_pending_tasks();
        assert!(cache.is_empty());
    }

    /// Many threads hitting the same entry must all get the same bytes without
    /// blocking one another.
    #[test]
    fn concurrent_readers_share_one_entry() {
        let cache = std::sync::Arc::new(ChunkCache::new(64));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cache = std::sync::Arc::clone(&cache);
                std::thread::spawn(move || {
                    for address in 0..32u64 {
                        let got = cache
                            .get_or_decode(address, || Ok(vec![address as u8; 4]))
                            .unwrap();
                        assert_eq!(&got[..], &vec![address as u8; 4][..]);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}

// ─── raw byte cache ────────────────────────────────────────────────────────

/// Default page size. Large enough that scattered metadata reads share pages,
/// small enough that a stray read does not pull a lot of waste.
pub const DEFAULT_PAGE_SIZE: usize = 64 * 1024;

/// Default number of pages held, giving 32 MiB at the default page size.
pub const DEFAULT_PAGE_CAPACITY: u64 = 512;

/// A cache of raw file bytes, held in fixed-size pages.
///
/// This sits below [`ChunkCache`]: that one caches *decoded* chunks, this one
/// caches the bytes any read touched, decoded or not. It exists because a file's
/// small reads cluster. Opening a file walks object headers, heaps and B-trees
/// that sit near each other; a scan then reads chunks from the same region. Each
/// of those is a separate request, and on object storage a request costs
/// milliseconds regardless of how few bytes it returns.
///
/// # Why pages rather than exact ranges
///
/// Caching arbitrary ranges needs an interval search to answer "is this covered".
/// Fixed pages make it a hash lookup, and they coalesce naturally: two reads
/// 100 bytes apart land in the same page, so the second is free. The cost is
/// reading up to a page of bytes nobody asked for, which is the same trade as
/// any block cache.
///
/// Like [`ChunkCache`] this is lock-free on the hit path, and two threads racing
/// a cold page may both fetch it.
#[derive(Clone, Debug)]
pub struct IoCache {
    pages: moka::sync::Cache<u64, Bytes>,
    page_size: usize,
    id: u64,
}

impl IoCache {
    /// Build a cache of `capacity` pages of `page_size` bytes each.
    pub fn new(capacity: u64, page_size: usize) -> Self {
        assert!(page_size > 0, "page size must be positive");
        Self {
            pages: moka::sync::Cache::new(capacity),
            page_size,
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// A cache with the default page size and capacity.
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_PAGE_CAPACITY, DEFAULT_PAGE_SIZE)
    }

    /// Build a cache holding roughly `bytes` of data, in pages of `page_size`.
    ///
    /// `page_size` is the I/O request size: every miss fetches a whole page, so
    /// a larger value means fewer, bigger requests. 256 KiB suits object
    /// storage, where a request costs milliseconds; the 64 KiB default suits
    /// local files, where over-reading is pure waste.
    pub fn with_capacity_bytes(bytes: usize, page_size: usize) -> Self {
        let pages = (bytes / page_size.max(1)).max(1) as u64;
        Self::new(pages, page_size)
    }

    /// Approximate bytes held, from the page count and size.
    pub fn capacity_bytes(&self) -> u64 {
        self.pages.policy().max_capacity().unwrap_or(0) * self.page_size as u64
    }

    /// Page size in bytes.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// This cache's identity.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Number of pages currently held. Approximate; eviction is deferred.
    pub fn len(&self) -> u64 {
        self.pages.entry_count()
    }

    /// Whether the cache holds nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop every page.
    pub fn clear(&self) {
        self.pages.invalidate_all();
    }

    /// One page by index, if the cache holds it.
    ///
    /// The index counts pages of [`IoCache::page_size`] from the start of the
    /// file. The asynchronous engine uses this to share pages between walks.
    pub fn page(&self, index: u64) -> Option<Bytes> {
        self.pages.get(&index)
    }

    /// Store one page by index.
    ///
    /// The bytes must be the file's bytes at `index * page_size`. A page other
    /// than the last one must be exactly [`IoCache::page_size`] long.
    pub fn put_page(&self, index: u64, bytes: Bytes) {
        self.pages.insert(index, bytes);
    }

    /// Apply the cache's pending inserts and evictions now.
    ///
    /// A write becomes visible to a reader a moment after the insert. This
    /// method removes that delay. Tests need it. Normal code does not: a page
    /// that a reader misses costs one fetch, not a wrong answer.
    pub fn run_pending(&self) {
        self.pages.run_pending_tasks();
    }

    /// Read `len` bytes at `offset`, serving whatever is already cached.
    ///
    /// Missing pages are fetched in runs, so cached pages in the middle of a
    /// range do not split it into extra requests.
    pub fn read(
        &self,
        source: &dyn crate::source::ByteSource,
        offset: u64,
        len: usize,
    ) -> crate::Result<Bytes> {
        let Some((first, last)) = self.bounds(source.size(), offset, len)? else {
            return Ok(Bytes::new());
        };
        let mut pages = self.cached(first, last);

        for (start, end) in self.missing_runs(&pages, first) {
            let (begin, want) = self.run_extent(start, end, source.size());
            let bytes = Bytes::from(source.read_vec(begin, want)?);
            self.store_run(start, end, first, bytes, &mut pages);
        }

        self.assemble(offset, len, first, &pages)
    }

    /// The asynchronous twin of [`IoCache::read`].
    ///
    /// The cache itself is synchronous; only the fetch awaits. Everything else
    /// — page bookkeeping, run detection, assembly — is shared.
    #[cfg(feature = "async")]
    #[cfg_attr(docsrs, doc(cfg(feature = "async")))]
    pub async fn read_async(
        &self,
        source: &dyn crate::async_source::AsyncByteSource,
        offset: u64,
        len: usize,
    ) -> crate::Result<Bytes> {
        let Some((first, last)) = self.bounds(source.size(), offset, len)? else {
            return Ok(Bytes::new());
        };
        let mut pages = self.cached(first, last);

        for (start, end) in self.missing_runs(&pages, first) {
            let (begin, want) = self.run_extent(start, end, source.size());
            let bytes = source.read_at(begin, want).await?;
            self.store_run(start, end, first, bytes, &mut pages);
        }

        self.assemble(offset, len, first, &pages)
    }

    /// The page range a read touches, or `None` for an empty read.
    fn bounds(&self, size: u64, offset: u64, len: usize) -> crate::Result<Option<(u64, u64)>> {
        if len == 0 {
            return Ok(None);
        }
        // Reject up front, so the page walk never has to cope with a partial
        // fetch. This matches how `ByteSource` treats a short read.
        if offset.saturating_add(len as u64) > size {
            return Err(crate::Error::OutOfBounds {
                what: "cached read",
                offset,
                len: len as u64,
                available: size.saturating_sub(offset),
            });
        }
        let page_size = self.page_size as u64;
        Ok(Some((offset / page_size, (offset + len as u64 - 1) / page_size)))
    }

    /// Whatever of `first..=last` is already held.
    fn cached(&self, first: u64, last: u64) -> Vec<Option<Bytes>> {
        (first..=last).map(|index| self.pages.get(&index)).collect()
    }

    /// Runs of consecutive missing pages, as inclusive page-index pairs.
    fn missing_runs(&self, pages: &[Option<Bytes>], first: u64) -> Vec<(u64, u64)> {
        let mut runs = Vec::new();
        let mut start: Option<u64> = None;
        for (slot, page) in pages.iter().enumerate() {
            let index = first + slot as u64;
            match (page.is_none(), start) {
                (true, None) => start = Some(index),
                (false, Some(s)) => {
                    runs.push((s, index - 1));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            runs.push((s, first + pages.len() as u64 - 1));
        }
        runs
    }

    /// Byte extent of a page run, clamped to the end of the file.
    fn run_extent(&self, start: u64, end: u64, size: u64) -> (u64, usize) {
        let page_size = self.page_size as u64;
        let begin = start * page_size;
        let want = ((end + 1) * page_size).min(size).saturating_sub(begin);
        (begin, want as usize)
    }

    /// Split a fetched run into pages and record them.
    fn store_run(
        &self,
        start: u64,
        end: u64,
        first: u64,
        bytes: Bytes,
        pages: &mut [Option<Bytes>],
    ) {
        for index in start..=end {
            let at = ((index - start) as usize) * self.page_size;
            if at >= bytes.len() {
                break;
            }
            let take = self.page_size.min(bytes.len() - at);
            let page = bytes.slice(at..at + take);
            self.pages.insert(index, page.clone());
            pages[(index - first) as usize] = Some(page);
        }
    }

    /// Build the requested range out of the pages holding it.
    fn assemble(
        &self,
        offset: u64,
        len: usize,
        first: u64,
        pages: &[Option<Bytes>],
    ) -> crate::Result<Bytes> {
        let page_size = self.page_size as u64;

        // One page: a slice of the cached buffer, no copy at all.
        if pages.len() == 1 {
            let page = pages[0].as_ref().ok_or_else(|| {
                crate::Error::malformed("page cache lost a page it had just fetched")
            })?;
            let at = (offset - first * page_size) as usize;
            if at + len > page.len() {
                return Err(crate::Error::OutOfBounds {
                    what: "cached read",
                    offset,
                    len: len as u64,
                    available: page.len().saturating_sub(at) as u64,
                });
            }
            return Ok(page.slice(at..at + len));
        }

        let mut out = Vec::with_capacity(len);
        for (slot, page) in pages.iter().enumerate() {
            let page = page.as_ref().ok_or_else(|| {
                crate::Error::malformed("page cache lost a page it had just fetched")
            })?;
            let page_start = (first + slot as u64) * page_size;
            let from = offset.max(page_start);
            let to = (offset + len as u64).min(page_start + page.len() as u64);
            if from < to {
                out.extend_from_slice(&page[(from - page_start) as usize..(to - page_start) as usize]);
            }
        }
        if out.len() != len {
            return Err(crate::Error::OutOfBounds {
                what: "cached read",
                offset,
                len: len as u64,
                available: out.len() as u64,
            });
        }
        Ok(Bytes::from(out))
    }
}

impl Default for IoCache {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod io_cache_tests {
    use super::*;
    use crate::source::{ByteSource, MemorySource};
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    /// Counts how many times the underlying source was touched.
    #[derive(Debug)]
    struct Counting {
        inner: MemorySource,
        reads: AtomicUsize,
    }
    impl ByteSource for Counting {
        fn size(&self) -> u64 {
            self.inner.size()
        }
        fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> crate::Result<()> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.inner.read_exact_at(offset, buf)
        }
    }
    fn source(n: usize) -> Counting {
        Counting {
            inner: MemorySource::new((0..n).map(|i| (i % 251) as u8).collect()),
            reads: AtomicUsize::new(0),
        }
    }

    #[test]
    fn a_second_read_in_the_same_page_costs_no_io() {
        let src = source(4096);
        let cache = IoCache::new(16, 1024);

        assert_eq!(&cache.read(&src, 10, 4).unwrap()[..], &[10, 11, 12, 13]);
        assert_eq!(src.reads.load(Ordering::Relaxed), 1);

        // Different offset, same page.
        let want: Vec<u8> = (500..502).map(|i: usize| (i % 251) as u8).collect();
        assert_eq!(&cache.read(&src, 500, 2).unwrap()[..], &want[..]);
        assert_eq!(src.reads.load(Ordering::Relaxed), 1, "served from cache");
    }

    #[test]
    fn scattered_small_reads_collapse_onto_few_pages() {
        let src = source(64 * 1024);
        let cache = IoCache::new(64, 16 * 1024);

        // 50 tiny reads spread over the first two pages.
        for i in 0..50u64 {
            let offset = i * 600;
            cache.read(&src, offset, 8).unwrap();
        }
        assert!(
            src.reads.load(Ordering::Relaxed) <= 2,
            "50 scattered reads should touch at most 2 pages, took {}",
            src.reads.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn a_read_spanning_pages_is_assembled_correctly() {
        let src = source(4096);
        let cache = IoCache::new(16, 256);
        let got = cache.read(&src, 250, 20).unwrap();
        let want: Vec<u8> = (250..270).map(|i| (i % 251) as u8).collect();
        assert_eq!(&got[..], &want[..]);
    }

    #[test]
    fn values_match_an_uncached_read_everywhere() {
        let src = source(5000);
        let cache = IoCache::new(64, 512);
        for &(offset, len) in &[(0u64, 1usize), (511, 2), (1000, 700), (4990, 10), (0, 5000)] {
            let cached = cache.read(&src, offset, len).unwrap();
            let direct = src.inner.read_vec(offset, len).unwrap();
            assert_eq!(&cached[..], &direct[..], "at {offset}+{len}");
        }
    }

    #[test]
    fn a_short_final_page_is_handled() {
        // 300 bytes with a 256-byte page: the second page holds only 44.
        let src = source(300);
        let cache = IoCache::new(8, 256);
        let got = cache.read(&src, 290, 10).unwrap();
        let want: Vec<u8> = (290..300).map(|i| (i % 251) as u8).collect();
        assert_eq!(&got[..], &want[..]);
    }

    #[test]
    fn reading_past_the_end_is_reported() {
        let src = source(100);
        let cache = IoCache::new(8, 64);
        assert!(cache.read(&src, 90, 50).is_err());
    }

    #[test]
    fn concurrent_readers_share_pages() {
        let src = Arc::new(source(1 << 20));
        let cache = Arc::new(IoCache::new(256, 64 * 1024));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let src = Arc::clone(&src);
                let cache = Arc::clone(&cache);
                std::thread::spawn(move || {
                    for i in 0..200u64 {
                        let got = cache.read(src.as_ref(), i * 1000, 16).unwrap();
                        assert_eq!(got.len(), 16);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
