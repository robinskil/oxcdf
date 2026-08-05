//! Temporary: is the async path really async?

#![cfg(feature = "async")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use bytes::Bytes;

/// A byte source that exists ONLY asynchronously.
///
/// It does not implement `ByteSource`, so no synchronous path can reach the
/// bytes. It also yields to the runtime on every fetch, so a caller that never
/// awaits cannot get data out of it.
#[derive(Debug)]
struct PureAsync {
    data: Vec<u8>,
    reads: AtomicUsize,
}

#[async_trait::async_trait]
impl oxcdf::AsyncByteSource for PureAsync {
    fn size(&self) -> u64 {
        self.data.len() as u64
    }

    async fn read_at(&self, offset: u64, len: usize) -> oxcdf::Result<Bytes> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        // Force a real suspension point. A blocking implementation could not
        // do this.
        tokio::task::yield_now().await;
        let start = offset as usize;
        let end = (start + len).min(self.data.len());
        Ok(Bytes::copy_from_slice(&self.data[start..end]))
    }
}

/// The biggest corpus file, so a small window cannot cover it.
const FILE: &str = "gridded-example.nc";

fn corpus(name: &str) -> Option<String> {
    let p = format!("{}/../../test_files/{name}", env!("CARGO_MANIFEST_DIR"));
    std::path::Path::new(&p).exists().then_some(p)
}

fn source(path: &str) -> Arc<PureAsync> {
    Arc::new(PureAsync {
        data: std::fs::read(path).unwrap(),
        reads: AtomicUsize::new(0),
    })
}

/// A 16 KiB window over a 608 KiB file, as a remote read really behaves.
fn tight() -> oxcdf::OpenOptions {
    oxcdf::OpenOptions::new()
        .open_prefetch_bytes(16 * 1024)
        .io_request_size(16 * 1024)
        .without_io_cache()
}

#[tokio::test]
async fn the_read_path_fetches_over_the_async_source() {
    let Some(path) = corpus(FILE) else { return };
    let src = source(&path);

    let file = oxcdf::open_async_with(src.clone(), tight()).await.unwrap();
    let after_open = src.reads.load(Ordering::Relaxed);

    let v = file.variable("analysed_sst").unwrap();
    let values = v.get_values::<f64, _>(..).await.unwrap();
    let after_read = src.reads.load(Ordering::Relaxed);

    println!(
        "open fetched {after_open} ranges; the read fetched {}",
        after_read - after_open
    );
    assert!(after_open > 0, "the open fetched nothing");
    assert!(after_read > after_open, "the read fetched nothing");

    // Byte-identical to the synchronous engine over a real file.
    let sync = oxcdf::open(&path).unwrap();
    let want = sync
        .variable("analysed_sst")
        .unwrap()
        .get_values::<f64, _>(..)
        .unwrap();
    assert_eq!(values.len(), want.len());
    assert!(values
        .iter()
        .zip(&want)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    println!("{} values match the synchronous engine exactly", want.len());
}

#[tokio::test]
async fn a_read_cannot_finish_without_awaiting() {
    let Some(path) = corpus(FILE) else { return };
    let src = source(&path);
    let file = oxcdf::open_async_with(src.clone(), tight()).await.unwrap();
    let v = file.variable("analysed_sst").unwrap();

    // Poll the read future exactly once. Every fetch yields, so a single poll
    // cannot reach the bytes.
    let fut = v.get_values::<f64, _>(..);
    tokio::pin!(fut);
    assert!(
        poll_once(&mut fut).is_none(),
        "the read completed on one poll, so it never awaited I/O"
    );

    let values = fut.await.unwrap();
    assert!(!values.is_empty());
    println!("the read needed more than one poll, as an async read must");
}

#[tokio::test]
async fn the_inner_sync_file_refuses_rather_than_blocking() {
    let Some(path) = corpus(FILE) else { return };
    let src = source(&path);
    let file = oxcdf::open_async_with(src.clone(), tight()).await.unwrap();

    // Metadata is complete and needs no await.
    let inner = file.netcdf();
    assert!(!inner.variables().is_empty());

    // A synchronous read through the inner file cannot invent bytes. It must
    // refuse, not block and not return wrong data.
    let before = src.reads.load(Ordering::Relaxed);
    let err = inner
        .variable("analysed_sst")
        .unwrap()
        .get_raw_values(..)
        .unwrap_err();
    let after = src.reads.load(Ordering::Relaxed);

    println!("inner sync read -> {err:?}");
    assert!(matches!(err, oxcdf::Error::Incomplete), "got {err:?}");
    assert_eq!(before, after, "the sync path performed I/O behind our back");
}

/// How many fetches a whole-file prefetch saves. Documents the real behaviour.
#[tokio::test]
async fn a_wide_window_serves_the_read_from_memory() {
    let Some(path) = corpus(FILE) else { return };
    let src = source(&path);

    // The default window is 4 MiB, larger than every corpus file.
    let file = oxcdf::open_async(src.clone()).await.unwrap();
    let after_open = src.reads.load(Ordering::Relaxed);

    let v = file.variable("analysed_sst").unwrap();
    v.get_raw_values(..).await.unwrap();
    let after_read = src.reads.load(Ordering::Relaxed);

    println!(
        "wide window: open fetched {after_open}, read fetched {}",
        after_read - after_open
    );
    assert_eq!(
        after_open, after_read,
        "a fully prefetched file should need no further fetch"
    );
}

/// Poll a future one time and report whether it finished.
fn poll_once<F: std::future::Future>(fut: &mut std::pin::Pin<&mut F>) -> Option<F::Output> {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn noop(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);

    match fut.as_mut().poll(&mut cx) {
        Poll::Ready(v) => Some(v),
        Poll::Pending => None,
    }
}
