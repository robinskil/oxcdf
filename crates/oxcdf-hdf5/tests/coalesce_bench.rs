//! Measures what coalescing actually changes: how many requests a read issues,
//! and how many bytes it pulls.
//!
//! Request count is the exact, machine-independent number, and it is what you
//! pay for on object storage in both money and tail latency. The wall-clock
//! figure here uses a simulated per-request delay, because there is no bucket to
//! measure against; it is an upper bound assuming requests are issued serially.
#![cfg(feature = "async")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use oxcdf_hdf5::async_source::AsyncByteSource;
use oxcdf_hdf5::index::Hdf5File;
use oxcdf_hdf5::io::IoConfig;
use oxcdf_hdf5::read::{read_hyperslab_async_with, Hyperslab};
use oxcdf_hdf5::source::{ByteSource, MemorySource};

/// Wraps an in-memory file, counting requests and optionally delaying each one.
#[derive(Debug)]
struct Metered {
    inner: MemorySource,
    requests: AtomicUsize,
    bytes: AtomicUsize,
    delay: Duration,
}

impl Metered {
    fn new(path: &str, delay: Duration) -> Self {
        Self {
            inner: MemorySource::open(path).unwrap(),
            requests: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
            delay,
        }
    }
    fn reset(&self) {
        self.requests.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
    }
}

#[async_trait::async_trait]
impl AsyncByteSource for Metered {
    fn size(&self) -> u64 {
        self.inner.size()
    }

    async fn read_at(&self, offset: u64, len: usize) -> oxcdf_hdf5::Result<Bytes> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(len, Ordering::Relaxed);
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }
        Ok(Bytes::from(self.inner.read_vec(offset, len)?))
    }

    async fn read_ranges(&self, ranges: &[(u64, usize)]) -> oxcdf_hdf5::Result<Vec<Bytes>> {
        // One store request per range, which is what object_store issues.
        let mut out = Vec::with_capacity(ranges.len());
        for &(offset, len) in ranges {
            out.push(self.read_at(offset, len).await?);
        }
        Ok(out)
    }
}

const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files/test_file.nc");

#[tokio::test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
async fn coalescing_reduces_requests() {
    if !std::path::Path::new(PATH).exists() {
        return;
    }

    // A whole-variable read of every chunked variable, which is the shape a
    // scan produces.
    let file = Hdf5File::open(PATH).unwrap();
    file.prepare_all().unwrap();
    let targets: Vec<_> = file
        .datasets()
        .iter()
        .filter(|d| d.is_readable() && !d.shape.is_empty())
        .map(|d| (*d).clone())
        .collect();
    println!("{} variables read per pass\n", targets.len());

    for (label, io) in [
        ("no coalescing ", IoConfig::NONE),
        ("local preset  ", IoConfig::LOCAL),
        ("remote preset ", IoConfig::REMOTE),
    ] {
        // Requests and bytes: exact, machine-independent.
        let source = Arc::new(Metered::new(PATH, Duration::ZERO));
        let file = Hdf5File::open(PATH).unwrap().with_cache(None);
        file.prepare_all().unwrap();
        source.reset();

        for d in file.datasets() {
            if !d.is_readable() || d.shape.is_empty() {
                continue;
            }
            let _ = read_hyperslab_async_with(
                source.as_ref(),
                file.superblock(),
                None,
                None,
                io,
                d,
                &Hyperslab::all(&d.shape),
            )
            .await;
        }

        let reqs = source.requests.load(Ordering::Relaxed);
        let bytes = source.bytes.load(Ordering::Relaxed);
        println!("{label} {reqs:>5} requests   {bytes:>8} bytes fetched");
    }

    // Simulated latency, serial worst case.
    println!("\nsimulated 1 ms per request (serial upper bound):");
    for (label, io) in [
        ("no coalescing ", IoConfig::NONE),
        ("remote preset ", IoConfig::REMOTE),
    ] {
        let source = Arc::new(Metered::new(PATH, Duration::from_millis(1)));
        let file = Hdf5File::open(PATH).unwrap().with_cache(None);
        file.prepare_all().unwrap();
        source.reset();

        let start = Instant::now();
        for d in file.datasets() {
            if !d.is_readable() || d.shape.is_empty() {
                continue;
            }
            let _ = read_hyperslab_async_with(
                source.as_ref(),
                file.superblock(),
                None,
                None,
                io,
                d,
                &Hyperslab::all(&d.shape),
            )
            .await;
        }
        println!("{label} {:>8.1?}", start.elapsed());
    }
}

/// The same measurement on a variable that actually has several chunks.
///
/// Coalescing merges ranges *within one read*. A variable stored as a single
/// chunk offers nothing to merge, so the win depends entirely on chunks per
/// variable, not on variables per file.
#[tokio::test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
async fn coalescing_on_a_multi_chunk_variable() {
    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_files/latest_v4_layout.h5"
    );

    for name in ["fixed_array", "implicit"] {
        println!("\n/{name}:");
        for (label, io) in [
            ("no coalescing ", IoConfig::NONE),
            ("local preset  ", IoConfig::LOCAL),
            ("remote preset ", IoConfig::REMOTE),
        ] {
            let source = Arc::new(Metered::new(FIXTURE, Duration::ZERO));
            let file = Hdf5File::open(FIXTURE).unwrap().with_cache(None);
            let d = file.dataset(&format!("/{name}")).unwrap();
            d.prepare(file.ctx()).unwrap();
            let chunks = d.resolved_chunks().map(|c| c.len()).unwrap_or(0);
            source.reset();

            read_hyperslab_async_with(
                source.as_ref(),
                file.superblock(),
                None,
                None,
                io,
                d,
                &Hyperslab::all(&d.shape),
            )
            .await
            .unwrap();

            println!(
                "  {label} {:>3} requests for {chunks} chunks, {:>6} bytes",
                source.requests.load(Ordering::Relaxed),
                source.bytes.load(Ordering::Relaxed)
            );
        }
    }
}
