//! The asynchronous open must agree with the synchronous one.
//!
//! Every test here opens one file both ways. It then compares the metadata and
//! the values. The synchronous engine is the reference: the differential tests
//! already compare that engine against netcdf-c, value by value.
//!
//! The tests also count round trips. A round trip is one call to the byte
//! source. The count states how many the asynchronous open costs.

#![cfg(feature = "async")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bytes::Bytes;
use oxcdf::async_source::AsyncByteSource;
use oxcdf::index::OpenOptions;
use oxcdf::source::{ByteSource, FileSource};
use oxcdf::{AsyncNetcdfFile, NetcdfFile};

/// A byte source that counts its calls and the bytes it serves.
#[derive(Debug)]
struct Counting {
    inner: FileSource,
    calls: AtomicUsize,
    ranges: AtomicUsize,
    bytes: AtomicUsize,
}

impl Counting {
    fn open(path: &str) -> Arc<Self> {
        Arc::new(Self {
            inner: FileSource::open(path).unwrap(),
            calls: AtomicUsize::new(0),
            ranges: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
        })
    }

    /// Calls to the source. Each one is one round trip on a network.
    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }

    fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }

    fn reset(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.ranges.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
    }
}

#[async_trait::async_trait]
impl AsyncByteSource for Counting {
    fn size(&self) -> u64 {
        self.inner.size()
    }

    async fn read_at(&self, offset: u64, len: usize) -> oxcdf::Result<Bytes> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.ranges.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(len, Ordering::Relaxed);
        Ok(Bytes::from(self.inner.read_vec(offset, len)?))
    }

    async fn read_ranges(&self, ranges: &[(u64, usize)]) -> oxcdf::Result<Vec<Bytes>> {
        // One call, however many ranges: that is what a batched fetch costs.
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.ranges.fetch_add(ranges.len(), Ordering::Relaxed);
        self.bytes
            .fetch_add(ranges.iter().map(|r| r.1).sum::<usize>(), Ordering::Relaxed);
        let mut out = Vec::with_capacity(ranges.len());
        for &(offset, len) in ranges {
            out.push(Bytes::from(self.inner.read_vec(offset, len)?));
        }
        Ok(out)
    }
}

fn corpus() -> Vec<String> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files");
    [
        "test_file.nc",
        "gridded-example.nc",
        "wod_ctd_1964.nc",
        "vlen_strings.nc",
        "vlen_seq.nc",
        "legacy_v1_objheader.h5",
        "latest_v4_layout.h5",
    ]
    .iter()
    .map(|p| format!("{root}/{p}"))
    .filter(|p| std::path::Path::new(p).exists())
    .collect()
}

// ─── metadata ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn the_two_opens_report_the_same_metadata() {
    for path in corpus() {
        let sync = NetcdfFile::open(&path).unwrap();
        let source = Counting::open(&path);
        let file = AsyncNetcdfFile::open(source.clone()).await.unwrap();

        let want: Vec<_> = sync.dimensions().iter().map(|d| (&d.name, d.len)).collect();
        let got: Vec<_> = file.dimensions().iter().map(|d| (&d.name, d.len)).collect();
        assert_eq!(got, want, "dimensions of {path}");

        let want: Vec<_> = sync.attributes().iter().map(|a| &a.name).collect();
        let got: Vec<_> = file.attributes().iter().map(|a| &a.name).collect();
        assert_eq!(got, want, "global attributes of {path}");

        let want: Vec<_> = sync
            .variables()
            .iter()
            .map(|v| (v.path.clone(), v.shape.clone(), v.dimensions.clone()))
            .collect();
        let got: Vec<_> = file
            .variables()
            .iter()
            .map(|v| (v.path.clone(), v.shape.clone(), v.dimensions.clone()))
            .collect();
        assert_eq!(got, want, "variables of {path}");

        println!(
            "{}: {} calls, {} KiB for the open",
            path.rsplit('/').next().unwrap(),
            source.calls(),
            source.bytes() / 1024
        );
    }
}

#[tokio::test]
async fn the_two_opens_report_the_same_variable_attributes() {
    for path in corpus() {
        let sync = NetcdfFile::open(&path).unwrap();
        let file = AsyncNetcdfFile::open(Counting::open(&path)).await.unwrap();

        for want in sync.variables() {
            let got = file
                .variable(&want.path)
                .unwrap_or_else(|| panic!("{} is missing from {path}", want.path));

            let want_attrs: Vec<_> = want
                .attributes
                .iter()
                .map(|a| (&a.name, format!("{:?}", a.value)))
                .collect();
            let got_attrs: Vec<_> = got
                .attributes()
                .iter()
                .map(|a| (&a.name, format!("{:?}", a.value)))
                .collect();
            assert_eq!(
                got_attrs, want_attrs,
                "attributes of {} in {path}",
                want.path
            );
            assert_eq!(got.vartype(), want.vartype(), "type of {}", want.path);
        }
    }
}

// ─── values ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn the_two_engines_read_the_same_bytes() {
    for path in corpus() {
        let sync = NetcdfFile::open(&path).unwrap();
        let file = AsyncNetcdfFile::open(Counting::open(&path)).await.unwrap();

        for want in sync.variables() {
            if !want.is_readable() || want.is_empty() {
                continue;
            }
            let got = file.variable(&want.path).unwrap();

            let want_values = want.get_raw_values(..).unwrap();
            let got_values = got.get_raw_values(..).await.unwrap();

            assert_eq!(got_values, want_values, "values of {} in {path}", want.path);
            // A variable-length value lives in a heap. Compare what it
            // resolves to, not the pointer that the raw bytes hold.
            if want.vartype().is_text() {
                assert_eq!(
                    got.get_strings(..).await.ok(),
                    want.get_strings(..).ok(),
                    "strings of {} in {path}",
                    want.path
                );
            }
        }
    }
}

#[tokio::test]
async fn a_slice_matches_the_synchronous_engine() {
    let path = format!(
        "{}/../../test_files/wod_ctd_1964.nc",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&path).exists() {
        return;
    }

    let sync = NetcdfFile::open(&path).unwrap();
    let file = AsyncNetcdfFile::open(Counting::open(&path)).await.unwrap();

    for want in sync.variables() {
        if !want.is_readable() || want.shape.len() != 1 || want.shape[0] < 8 {
            continue;
        }
        let ranges = vec![std::ops::Range {
            start: 2usize,
            end: 7,
        }];
        let got = file.variable(&want.path).unwrap();
        assert_eq!(
            got.get_raw_values(ranges.clone()).await.unwrap(),
            want.get_raw_values(ranges).unwrap(),
            "slice of {}",
            want.path
        );
    }
}

// ─── cost ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn an_open_costs_few_round_trips() {
    for path in corpus() {
        let source = Counting::open(&path);
        AsyncNetcdfFile::open_with(source.clone(), OpenOptions::remote())
            .await
            .unwrap();
        let calls = source.calls();
        assert!(
            calls <= 8,
            "{path} took {calls} round trips to open; the metadata window should cover it"
        );
    }
}

#[tokio::test]
async fn a_small_window_still_opens_the_file() {
    // Force many rounds by refusing to prefetch. The result must not change.
    for path in corpus() {
        let source = Counting::open(&path);
        let file = AsyncNetcdfFile::open_with(
            source.clone(),
            OpenOptions::new()
                .io_request_size(4096)
                .open_prefetch_bytes(4096),
        )
        .await
        .unwrap();

        let sync = NetcdfFile::open(&path).unwrap();
        assert_eq!(
            file.variables().len(),
            sync.variables().len(),
            "a small window must not change what {path} contains"
        );
        println!(
            "{}: {} calls with a 4 KiB window",
            path.rsplit('/').next().unwrap(),
            source.calls()
        );
    }
}

#[tokio::test]
async fn a_second_read_of_one_variable_makes_no_metadata_request() {
    let path = format!(
        "{}/../../test_files/wod_ctd_1964.nc",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&path).exists() {
        return;
    }
    let source = Counting::open(&path);
    let file = AsyncNetcdfFile::open_with(source.clone(), OpenOptions::remote())
        .await
        .unwrap();

    let name = file
        .variables()
        .iter()
        .find(|v| v.is_readable() && !v.is_empty())
        .map(|v| v.path.clone())
        .expect("a readable variable");

    let var = file.variable(&name).unwrap();
    var.get_raw_values(..).await.unwrap();
    source.reset();
    var.get_raw_values(..).await.unwrap();

    assert_eq!(
        source.calls(),
        0,
        "the byte cache and the chunk cache should serve a repeat read of {name}"
    );
}

// ─── the API shape ─────────────────────────────────────────────────────────

#[tokio::test]
async fn the_async_netcdf_file_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AsyncNetcdfFile>();
    assert_send_sync::<oxcdf::AsyncVariable<'_>>();
}

#[tokio::test]
async fn a_variable_dereferences_to_its_metadata() {
    let path = format!(
        "{}/../../test_files/test_file.nc",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&path).exists() {
        return;
    }
    let file = AsyncNetcdfFile::open(Counting::open(&path)).await.unwrap();
    let var = file.variables().into_iter().next().unwrap();

    // These come from `NcVariable` through `Deref`, with no await.
    let _: &[u64] = &var.shape;
    let _: &str = &var.path;
    let _: &[String] = &var.dimensions;
}
