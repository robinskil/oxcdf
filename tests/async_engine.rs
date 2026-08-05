//! The async engine, checked against the sync one.
//!
//! Both engines share the index, the chunk maths, the filters and the assembly,
//! so any difference in output means the fetch fork diverged. Every test here
//! compares the two.
#![cfg(feature = "async")]

use std::sync::Arc;

use oxcdf::async_source::{AsyncByteSource, SyncAsAsync};
use oxcdf::index::Hdf5File;
use oxcdf::read::{read_hyperslab, read_hyperslab_async, Hyperslab};
use oxcdf::source::MemorySource;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_files/legacy_v1_objheader.h5"
);

fn argo_path() -> Option<&'static str> {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_files/test_file.nc"
    );
    std::path::Path::new(p).exists().then_some(p)
}

/// Open sync (the index is engine-agnostic), then read async over the same
/// bytes, and require both engines to agree exactly.
async fn assert_engines_agree(path: &str, dataset: &str, slab: Option<Hyperslab>) {
    let file = Hdf5File::open(path).unwrap();
    let d = file.dataset(dataset).unwrap();
    let slab = slab.unwrap_or_else(|| Hyperslab::all(&d.shape));

    let sync = read_hyperslab(file.ctx(), d, &slab).unwrap();

    let async_source: Arc<dyn AsyncByteSource> =
        Arc::new(SyncAsAsync(MemorySource::open(path).unwrap()));
    let asynced = read_hyperslab_async(
        async_source.as_ref(),
        file.superblock(),
        file.cache(),
        d,
        &slab,
    )
    .await
    .unwrap();

    assert_eq!(asynced.shape, sync.shape, "{dataset}: shape differs");
    assert_eq!(
        asynced.bytes, sync.bytes,
        "{dataset}: the async engine produced different bytes"
    );
}

#[tokio::test]
async fn a_contiguous_variable_matches_the_sync_engine() {
    assert_engines_agree(FIXTURE, "/contig_f64", None).await;
}

#[tokio::test]
async fn a_chunked_and_compressed_variable_matches() {
    assert_engines_agree(FIXTURE, "/chunked_i32", None).await;
}

#[tokio::test]
async fn a_big_endian_variable_matches() {
    assert_engines_agree(FIXTURE, "/contig_f32be", None).await;
}

#[tokio::test]
async fn a_hyperslab_across_chunk_boundaries_matches() {
    let slab = Hyperslab {
        start: vec![5, 2],
        count: vec![10, 3],
    };
    assert_engines_agree(FIXTURE, "/chunked_i32", Some(slab)).await;
}

#[tokio::test]
async fn a_nested_variable_matches() {
    assert_engines_agree(FIXTURE, "/subgroup/nested_i16", None).await;
}

#[tokio::test]
async fn every_readable_variable_of_a_real_file_matches() {
    let Some(path) = argo_path() else { return };
    let file = Hdf5File::open(path).unwrap();

    let mut checked = 0;
    for d in file.datasets() {
        if !d.is_readable() || d.shape.is_empty() {
            continue;
        }
        assert_engines_agree(path, &d.path, None).await;
        checked += 1;
    }
    assert!(checked > 10, "expected many variables, checked {checked}");
}

/// Values must be right on the first read too, not only once the cache is warm.
#[tokio::test]
async fn a_cold_async_read_is_correct() {
    let file = Hdf5File::open(FIXTURE).unwrap().with_cache(None);
    let d = file.dataset("/chunked_i32").unwrap();
    // The async engine does not walk B-trees, so the index is resolved first.
    d.prepare(file.ctx()).unwrap();

    let source: Arc<dyn AsyncByteSource> =
        Arc::new(SyncAsAsync(MemorySource::open(FIXTURE).unwrap()));
    let got = read_hyperslab_async(
        source.as_ref(),
        file.superblock(),
        None,
        d,
        &Hyperslab::all(&d.shape),
    )
    .await
    .unwrap();

    let values = got.to_i64(d).unwrap();
    assert_eq!(values.len(), 240);
    for (i, v) in values.iter().enumerate() {
        assert_eq!(*v, i as i64 * 3 - 100, "element {i}");
    }
}

/// The whole point of the async engine: it works on a current-thread runtime,
/// which the blocking object-store adapter cannot.
#[test]
fn reads_on_a_current_thread_runtime() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        assert_engines_agree(FIXTURE, "/chunked_i32", None).await;
    });
}

/// Many tasks reading concurrently must all agree, with no lock between them.
#[tokio::test(flavor = "multi_thread")]
async fn many_tasks_read_concurrently() {
    let file = Arc::new(Hdf5File::open(FIXTURE).unwrap());
    let source: Arc<dyn AsyncByteSource> =
        Arc::new(SyncAsAsync(MemorySource::open(FIXTURE).unwrap()));

    let expected = {
        let d = file.dataset("/chunked_i32").unwrap();
        read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
            .unwrap()
            .bytes
    };

    let mut tasks = Vec::new();
    for _ in 0..8 {
        let file = Arc::clone(&file);
        let source = Arc::clone(&source);
        let expected = expected.clone();
        tasks.push(tokio::spawn(async move {
            let d = file.dataset("/chunked_i32").unwrap();
            let got = read_hyperslab_async(
                source.as_ref(),
                file.superblock(),
                file.cache(),
                d,
                &Hyperslab::all(&d.shape),
            )
            .await
            .unwrap();
            assert_eq!(got.bytes, expected);
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
}

/// The async engine deliberately does not walk chunk indexes: that is a chain of
/// dependent reads, which belongs in `prepare`. Reading without preparing must
/// say so rather than quietly return fill values.
#[tokio::test]
async fn an_unprepared_chunked_variable_is_refused() {
    let file = Hdf5File::open(FIXTURE).unwrap();
    let d = file.dataset("/chunked_i32").unwrap();
    assert!(!d.is_prepared(), "opening must not resolve the index");

    let source: Arc<dyn AsyncByteSource> =
        Arc::new(SyncAsAsync(MemorySource::open(FIXTURE).unwrap()));
    let err = read_hyperslab_async(
        source.as_ref(),
        file.superblock(),
        file.cache(),
        d,
        &Hyperslab::all(&d.shape),
    )
    .await
    .expect_err("an unprepared chunked read must fail");
    assert!(
        matches!(err, oxcdf::Error::BadRequest(_)),
        "got {err:?}"
    );

    // After preparing, the same read succeeds.
    d.prepare(file.ctx()).unwrap();
    assert!(d.is_prepared());
    read_hyperslab_async(
        source.as_ref(),
        file.superblock(),
        file.cache(),
        d,
        &Hyperslab::all(&d.shape),
    )
    .await
    .unwrap();
}

/// A contiguous variable needs no index, so it reads asynchronously with no
/// preparation at all.
#[tokio::test]
async fn a_contiguous_variable_needs_no_preparation() {
    let file = Hdf5File::open(FIXTURE).unwrap();
    let d = file.dataset("/contig_f64").unwrap();
    let source: Arc<dyn AsyncByteSource> =
        Arc::new(SyncAsAsync(MemorySource::open(FIXTURE).unwrap()));
    let got = read_hyperslab_async(
        source.as_ref(),
        file.superblock(),
        file.cache(),
        d,
        &Hyperslab::all(&d.shape),
    )
    .await
    .unwrap();
    assert_eq!(got.to_f64(d).unwrap().len(), 240);
}
