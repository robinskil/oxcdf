//! Asynchronous metadata walks, by replay.
//!
//! # The problem
//!
//! An open walks the file metadata. The walk is a chain of dependent reads. A
//! superblock names a root object header. That header names a heap. That heap
//! names more headers. Each step needs the bytes of the step before it.
//!
//! The walk is one large recursive parser. A second asynchronous copy of that
//! parser costs more than a thousand lines. The two copies then drift apart.
//!
//! # The method
//!
//! This module runs the **same** parser. It gives the parser a byte source that
//! holds pages in memory. The parser never waits.
//!
//! 1. Fetch a window of pages.
//! 2. Run the synchronous walk over those pages.
//! 3. A read outside the pages records the pages it wants. It returns
//!    [`Error::Incomplete`].
//! 4. Fetch every recorded page in one batch. Go to step 2.
//! 5. A walk that finishes returns the result.
//!
//! One round of this loop is one batch of requests. The count of rounds equals
//! the depth of the dependent reads. An asynchronous parser needs the same
//! count of round trips. This method also fetches every page of one round
//! together, which a naive recursive parser does not.
//!
//! # Cost
//!
//! A round repeats the parse work of the round before it. The parse is pure
//! processor work over memory. It costs microseconds. A round trip to object
//! storage costs milliseconds. The repeat is not measurable.
//!
//! The first window normally covers the whole walk. netCDF writes its metadata
//! near the front of the file. Most opens take one round trip.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use bytes::Bytes;

use crate::async_source::AsyncByteSource;
use crate::error::{Error, Result};
use crate::source::ByteSource;

/// Stop a damaged or hostile file from looping forever.
const MAX_ROUNDS: usize = 64;

/// How many bytes to fetch before the first round.
///
/// netCDF writes its metadata near the front of the file. One window of this
/// size normally covers a whole open.
pub const DEFAULT_PREFETCH_BYTES: usize = 4 << 20;

/// A byte source over the pages held in memory.
///
/// A read inside the held pages succeeds. A read outside them records the pages
/// it wants. It then fails with [`Error::Incomplete`].
///
/// The page map does not change while a walk runs, so a read takes no lock. The
/// driver builds a new source for each round.
pub(crate) struct ReplaySource {
    size: u64,
    page_size: usize,
    pages: HashMap<u64, Bytes>,
    /// Pages a read asked for and did not find.
    ///
    /// This is the only mutable state. A walk touches it once for each missing
    /// page, and only during an open. The lock is never contended.
    wanted: Mutex<BTreeSet<u64>>,
}

impl std::fmt::Debug for ReplaySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplaySource")
            .field("size", &self.size)
            .field("page_size", &self.page_size)
            .field("pages_held", &self.pages.len())
            .finish()
    }
}

impl ReplaySource {
    pub(crate) fn new(size: u64, page_size: usize, pages: HashMap<u64, Bytes>) -> Self {
        Self {
            size,
            page_size: page_size.max(1),
            pages,
            wanted: Mutex::new(BTreeSet::new()),
        }
    }

    /// The pages this round asked for and did not find.
    fn wanted(&self) -> BTreeSet<u64> {
        self.wanted.lock().expect("replay lock").clone()
    }

    /// The page range a read touches.
    fn span(&self, offset: u64, len: usize) -> (u64, u64) {
        let page_size = self.page_size as u64;
        (offset / page_size, (offset + len as u64 - 1) / page_size)
    }
}

impl ByteSource for ReplaySource {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        if offset.saturating_add(buf.len() as u64) > self.size {
            return Err(Error::OutOfBounds {
                what: "file",
                offset,
                len: buf.len() as u64,
                available: self.size.saturating_sub(offset),
            });
        }

        let (first, last) = self.span(offset, buf.len());

        // Record every page the read needs, not only the first one. One round
        // then fetches the whole range.
        let mut missing = false;
        for index in first..=last {
            if !self.pages.contains_key(&index) {
                self.wanted.lock().expect("replay lock").insert(index);
                missing = true;
            }
        }
        if missing {
            return Err(Error::Incomplete);
        }

        let page_size = self.page_size as u64;
        let mut done = 0usize;
        for index in first..=last {
            let page = &self.pages[&index];
            let page_start = index * page_size;
            let from = (offset.max(page_start) - page_start) as usize;
            if from >= page.len() {
                break;
            }
            let take = (buf.len() - done).min(page.len() - from);
            buf[done..done + take].copy_from_slice(&page[from..from + take]);
            done += take;
        }

        if done != buf.len() {
            return Err(Error::malformed(format!(
                "a held page is short: filled {done} of {} bytes at offset {offset}",
                buf.len()
            )));
        }
        Ok(())
    }

    /// Read several ranges.
    ///
    /// This records the missing pages of **every** range before it fails. The
    /// default would stop at the first range, which costs one round for each
    /// range instead of one round for all of them.
    fn read_ranges(&self, ranges: &[(u64, usize)]) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::with_capacity(ranges.len());
        let mut missing = false;

        for &(offset, len) in ranges {
            match self.read_vec(offset, len) {
                Ok(bytes) if !missing => out.push(bytes),
                Ok(_) => {}
                Err(Error::Incomplete) => missing = true,
                Err(other) => return Err(other),
            }
        }

        if missing {
            return Err(Error::Incomplete);
        }
        Ok(out)
    }
}

/// Run a synchronous walk over an asynchronous source.
///
/// `build` receives a byte source and returns the walk's result. The driver
/// calls it once per round. It must be pure: the driver discards the result of
/// every round but the last.
///
/// `io_cache` seeds the page store and receives every page fetched. Two walks
/// of one file then share their pages, and the read path reuses them.
pub(crate) async fn replay<T, F>(
    source: &dyn AsyncByteSource,
    io_cache: Option<&crate::cache::IoCache>,
    page_size: usize,
    prefetch_bytes: usize,
    build: F,
) -> Result<T>
where
    F: Fn(Arc<dyn ByteSource>) -> Result<T>,
{
    let size = source.size();
    let page_size = page_size.max(1);
    if size == 0 {
        return Err(Error::malformed("the source is empty"));
    }

    let last_page = (size - 1) / page_size as u64;
    let mut pages: HashMap<u64, Bytes> = HashMap::new();

    // Seed with a window at the front of the file.
    let window = (prefetch_bytes.div_ceil(page_size) as u64).max(1);
    let mut wanted: BTreeSet<u64> = (0..window.min(last_page + 1)).collect();
    let mut previous: BTreeSet<u64> = BTreeSet::new();

    for _ in 0..MAX_ROUNDS {
        fetch(source, io_cache, page_size, size, &wanted, &mut pages).await?;

        let replay = Arc::new(ReplaySource::new(size, page_size, pages.clone()));
        let outcome = build(replay.clone() as Arc<dyn ByteSource>);
        let missing = replay.wanted();

        match outcome {
            // A walk that asked for nothing it lacked has read the real bytes.
            Ok(value) if missing.is_empty() => return Ok(value),
            // A walk can also swallow a missing read and return a short answer.
            // Fetch what it asked for and run it again. Accept the answer when
            // another round changes nothing.
            Ok(value) if missing == previous => return Ok(value),
            Ok(_) => {}
            Err(Error::Incomplete) => {
                if missing.is_empty() || missing == previous {
                    return Err(Error::malformed(
                        "the asynchronous walk asked for no new bytes and did not finish",
                    ));
                }
            }
            Err(other) => return Err(other),
        }

        previous = missing.clone();
        wanted = missing;
    }

    Err(Error::malformed(format!(
        "the asynchronous walk did not finish in {MAX_ROUNDS} rounds"
    )))
}

/// Fetch every wanted page into `pages`.
///
/// Consecutive pages become one request. Every request goes out together.
async fn fetch(
    source: &dyn AsyncByteSource,
    io_cache: Option<&crate::cache::IoCache>,
    page_size: usize,
    size: u64,
    wanted: &BTreeSet<u64>,
    pages: &mut HashMap<u64, Bytes>,
) -> Result<()> {
    // A page another walk already fetched costs nothing.
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for &index in wanted {
        if pages.contains_key(&index) {
            continue;
        }
        if let Some(cached) = io_cache.filter(|c| c.page_size() == page_size).and_then(|c| c.page(index)) {
            pages.insert(index, cached);
            continue;
        }
        match runs.last_mut() {
            Some((_, end)) if *end + 1 == index => *end = index,
            _ => runs.push((index, index)),
        }
    }
    if runs.is_empty() {
        return Ok(());
    }

    let ranges: Vec<(u64, usize)> = runs
        .iter()
        .map(|&(first, last)| {
            let begin = first * page_size as u64;
            let end = ((last + 1) * page_size as u64).min(size);
            (begin, end.saturating_sub(begin) as usize)
        })
        .collect();

    let fetched = source.read_ranges(&ranges).await?;
    if fetched.len() != runs.len() {
        return Err(Error::malformed(format!(
            "the source returned {} buffers for {} requests",
            fetched.len(),
            runs.len()
        )));
    }

    for (&(first, _), bytes) in runs.iter().zip(fetched) {
        for (step, start) in (0..bytes.len()).step_by(page_size).enumerate() {
            let end = (start + page_size).min(bytes.len());
            let index = first + step as u64;
            let page = bytes.slice(start..end);
            if let Some(cache) = io_cache.filter(|c| c.page_size() == page_size) {
                cache.put_page(index, page.clone());
            }
            pages.insert(index, page);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::async_source::SyncAsAsync;
    use crate::source::MemorySource;

    fn data(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn a_held_read_succeeds_and_a_missing_one_records_its_pages() {
        let mut pages = HashMap::new();
        pages.insert(0u64, Bytes::from(data(16)));
        let src = ReplaySource::new(64, 16, pages);

        let mut buf = [0u8; 8];
        src.read_exact_at(4, &mut buf).unwrap();
        assert_eq!(&buf[..], &data(16)[4..12]);

        let mut buf = [0u8; 8];
        assert!(matches!(
            src.read_exact_at(20, &mut buf),
            Err(Error::Incomplete)
        ));
        assert_eq!(src.wanted(), BTreeSet::from([1]));
    }

    #[test]
    fn a_read_across_pages_records_every_page_it_needs() {
        let src = ReplaySource::new(1024, 16, HashMap::new());
        let mut buf = [0u8; 40];
        assert!(matches!(
            src.read_exact_at(20, &mut buf),
            Err(Error::Incomplete)
        ));
        // Offsets 20..60 span pages 1, 2 and 3.
        assert_eq!(src.wanted(), BTreeSet::from([1, 2, 3]));
    }

    #[test]
    fn a_read_spanning_held_pages_is_stitched_together() {
        let whole = data(48);
        let mut pages = HashMap::new();
        for i in 0..3u64 {
            let start = i as usize * 16;
            pages.insert(i, Bytes::copy_from_slice(&whole[start..start + 16]));
        }
        let src = ReplaySource::new(48, 16, pages);

        let mut buf = [0u8; 40];
        src.read_exact_at(5, &mut buf).unwrap();
        assert_eq!(&buf[..], &whole[5..45]);
    }

    #[test]
    fn a_batch_read_records_the_missing_pages_of_every_range() {
        let mut pages = HashMap::new();
        pages.insert(0u64, Bytes::from(data(16)));
        let src = ReplaySource::new(1024, 16, pages);

        // Range one is held. Ranges two and three are not.
        let err = src
            .read_ranges(&[(0, 8), (32, 8), (80, 8)])
            .unwrap_err();
        assert!(matches!(err, Error::Incomplete));
        assert_eq!(
            src.wanted(),
            BTreeSet::from([2, 5]),
            "one round must fetch both missing ranges"
        );
    }

    #[test]
    fn a_read_past_the_end_is_out_of_bounds_not_incomplete() {
        let src = ReplaySource::new(32, 16, HashMap::new());
        let mut buf = [0u8; 8];
        assert!(matches!(
            src.read_exact_at(30, &mut buf),
            Err(Error::OutOfBounds { .. })
        ));
    }

    #[tokio::test]
    async fn the_driver_converges_on_a_chain_of_dependent_reads() {
        // Byte 0 of each page names the next page to read. The walk therefore
        // cannot ask for page N+1 until it holds page N.
        let mut whole = data(16 * 8);
        for i in 0..8usize {
            whole[i * 16] = (i as u8 + 1) % 8;
        }
        let source = SyncAsAsync(MemorySource::new(whole.clone()));

        let rounds = std::sync::atomic::AtomicUsize::new(0);
        let seen = replay(&source, None, 16, 16, |src| {
            rounds.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut visited = Vec::new();
            let mut page = 0u64;
            for _ in 0..8 {
                let byte = src.read_vec(page * 16, 1)?;
                visited.push(page);
                page = byte[0] as u64;
            }
            Ok(visited)
        })
        .await
        .unwrap();

        assert_eq!(seen, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        // One round per link, plus the round that succeeds.
        assert!(rounds.load(std::sync::atomic::Ordering::Relaxed) <= 9);
    }

    #[tokio::test]
    async fn one_generous_window_finishes_in_a_single_round() {
        let source = SyncAsAsync(MemorySource::new(data(1024)));
        let rounds = std::sync::atomic::AtomicUsize::new(0);

        let total = replay(&source, None, 64, 1024, |src| {
            rounds.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(src.read_vec(0, 1024)?.len())
        })
        .await
        .unwrap();

        assert_eq!(total, 1024);
        assert_eq!(rounds.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn a_real_error_stops_the_driver_at_once() {
        let source = SyncAsAsync(MemorySource::new(data(256)));
        let err = replay(&source, None, 64, 256, |_| {
            Err::<(), _>(Error::malformed("bad signature"))
        })
        .await
        .unwrap_err();
        assert!(matches!(err, Error::Malformed(m) if m == "bad signature"));
    }

    #[tokio::test]
    async fn a_walk_that_never_settles_is_reported() {
        let source = SyncAsAsync(MemorySource::new(data(64 * 1024)));
        let round = std::sync::atomic::AtomicU64::new(0);

        // Ask for a page further away on every round, so the loop never ends.
        let err = replay(&source, None, 64, 64, |src| {
            let n = round.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            src.read_vec((n + 1) * 10 * 64, 64).map(|_| ())
        })
        .await
        .unwrap_err();

        assert!(matches!(err, Error::Malformed(_)), "got {err:?}");
        assert!(
            round.load(std::sync::atomic::Ordering::Relaxed) as usize == MAX_ROUNDS,
            "the driver must stop at the round limit"
        );
    }

    #[tokio::test]
    async fn fetched_pages_land_in_the_shared_byte_cache() {
        let source = SyncAsAsync(MemorySource::new(data(4096)));
        let cache = crate::cache::IoCache::new(64, 64);

        replay(&source, Some(&cache), 64, 256, |src| {
            src.read_vec(0, 256).map(|_| ())
        })
        .await
        .unwrap();

        cache.run_pending();
        assert!(cache.page(0).is_some(), "the first page is cached");
        assert_eq!(cache.page(0).unwrap().len(), 64);
    }
}
