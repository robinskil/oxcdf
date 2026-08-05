//! The README must not drift from the API.
//!
//! Every example in the README appears here. A test that runs uses the test
//! corpus. A test that needs a network only has to compile, so it lives in a
//! function that nothing calls.

use std::sync::Arc;

use oxcdf::{AsyncFile, FileSource, SyncAsAsync};

fn corpus(name: &str) -> Option<String> {
    let path = format!("{}/test_files/{name}", env!("CARGO_MANIFEST_DIR"));
    std::path::Path::new(&path).exists().then_some(path)
}

// ─── "Read a local file" ───────────────────────────────────────────────────

#[test]
fn the_synchronous_example_works() -> oxcdf::Result<()> {
    let Some(path) = corpus("gridded-example.nc") else {
        return Ok(());
    };
    let file = oxcdf::open(&path)?;

    for d in file.dimensions() {
        let mark = if d.is_unlimited { " (unlimited)" } else { "" };
        println!("{} = {}{}", d.name, d.len, mark);
    }

    for a in file.attributes() {
        println!("{} = {:?}", a.name, a.value);
    }

    for v in file.variables() {
        println!("{} {:?} {:?} {:?}", v.name, v.dtype(), v.shape, v.dimensions);
    }

    let temp = file.variable("analysed_sst").expect("a variable");
    println!("{:?}", temp.attribute("units").map(|a| a.value.as_text()));
    println!("{:?}", temp.attribute("_FillValue").map(|a| a.value.as_f64()));

    let _all = temp.read()?;
    let _part = temp.read_slice(&[0..1, 0..4, 0..4])?;

    // A path into a subgroup resolves the same way. This file has no subgroup.
    assert!(file.variable("/forecast/TEMP").is_none());
    Ok(())
}

// ─── "Read into an ndarray" ────────────────────────────────────────────────

#[cfg(feature = "ndarray")]
#[test]
fn the_ndarray_example_works() -> oxcdf::Result<()> {
    let Some(path) = corpus("legacy_v1_objheader.h5") else {
        return Ok(());
    };
    let file = oxcdf::open(&path)?;
    let temp = file.variable("/contig_f64").expect("a 2-D variable");

    let a = temp.read_array_f64()?;
    assert_eq!(a.shape(), &[40, 6]);
    println!("{}", a[[0, 0]]);

    let row = a.index_axis(ndarray::Axis(0), 0);
    assert_eq!(row.len(), 6);

    let b = temp.read_slice(&[5..15, 2..5])?.to_array_f64()?;
    assert_eq!(b.shape(), &[10, 3]);

    let counts = file.variable("/chunked_i32").unwrap().read_array_i64()?;
    assert_eq!(counts.shape(), &[40, 6]);

    let names = file
        .variable("/fixed_strings")
        .unwrap()
        .read()?
        .to_array_strings()?;
    assert_eq!(names.shape(), &[5]);

    // `read_array_i64` refuses a float rather than round it. The README says so.
    assert!(temp.read_array_i64().is_err());
    Ok(())
}

// ─── "Read one chunk at a time" ────────────────────────────────────────────

#[test]
fn the_chunk_example_works() -> oxcdf::Result<()> {
    let Some(path) = corpus("legacy_v1_objheader.h5") else {
        return Ok(());
    };
    let file = oxcdf::open(&path)?;
    let temp = file.variable("/chunked_i32").expect("a chunked variable");

    // The README maps this with rayon. `rayon` is not a dependency here, so
    // check the same calls in sequence.
    let blocks: Vec<_> = temp.chunks().iter().map(|c| temp.read_chunk(c)).collect();
    assert!(blocks.iter().all(|b| b.is_ok()));
    Ok(())
}

// ─── "Read a local file asynchronously" ────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test]
async fn the_local_async_example_works() -> oxcdf::Result<()> {
    let Some(path) = corpus("legacy_v1_objheader.h5") else {
        return Ok(());
    };
    let source = Arc::new(SyncAsAsync(FileSource::open(&path)?));
    let file = oxcdf::open_async(source).await?;

    let temp = file.variable("/contig_f64").unwrap();
    let _values = temp.read().await?.to_f64()?;
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn the_async_interface_matches_the_table() -> oxcdf::Result<()> {
    let Some(path) = corpus("legacy_v1_objheader.h5") else {
        return Ok(());
    };
    let file = oxcdf::open_async(Arc::new(SyncAsAsync(FileSource::open(&path)?))).await?;

    // Every metadata call answers without an await.
    let _ = file.dimensions();
    let _ = file.attributes();
    let _ = file.variables();
    let _ = file.group("/subgroup");

    let temp = file.variable("/chunked_i32").unwrap();
    let _ = temp.attribute("units");
    let _ = temp.dtype();
    let _ = temp.shape.clone();
    let _ = temp.dimensions.clone();

    // Only the values await.
    let _ = temp.read().await?;
    let _ = temp.read_slice(&[0..2, 0..2]).await?;
    let chunks = temp.chunks().await?;
    let _ = temp.read_chunk(&chunks[0]).await?;
    Ok(())
}

#[cfg(all(feature = "async", feature = "ndarray"))]
#[tokio::test]
async fn the_async_ndarray_example_works() -> oxcdf::Result<()> {
    let Some(path) = corpus("legacy_v1_objheader.h5") else {
        return Ok(());
    };
    let file = oxcdf::open_async(Arc::new(SyncAsAsync(FileSource::open(&path)?))).await?;
    let temp = file.variable("/contig_f64").unwrap();

    let a = temp.read_array_f64().await?;
    assert_eq!(a.shape(), &[40, 6]);
    let b = temp.read_slice(&[5..15, 2..5]).await?.to_array_f64()?;
    assert_eq!(b.shape(), &[10, 3]);
    Ok(())
}

// ─── "Read from object storage" ────────────────────────────────────────────
//
// This needs a bucket, so it only has to compile.

#[cfg(all(feature = "async", feature = "object-store"))]
#[allow(dead_code)]
async fn the_object_store_example_compiles(
    store: Arc<dyn object_store::ObjectStore>,
) -> oxcdf::Result<()> {
    use object_store::path::Path;

    let file = AsyncFile::open_store(store, Path::from("dac/aoml/13857_prof.nc")).await?;

    for d in file.dimensions() {
        let mark = if d.is_unlimited { " (unlimited)" } else { "" };
        println!("{} = {}{}", d.name, d.len, mark);
    }

    for a in file.attributes() {
        println!("{} = {:?}", a.name, a.value);
    }

    for v in file.variables() {
        println!("{} {:?} {:?} {:?}", v.name, v.dtype(), v.shape, v.dimensions);
    }

    let temp = file.variable("TEMP").unwrap();
    println!("{:?}", temp.attribute("units").unwrap().value.as_text());

    let _all = temp.read().await?.to_f64()?;
    let _part = temp.read_slice(&[0..8, 10..30]).await?.to_f64()?;
    Ok(())
}

#[cfg(all(feature = "async", feature = "object-store"))]
#[allow(dead_code)]
async fn the_open_store_options_compile(
    store: Arc<dyn object_store::ObjectStore>,
) -> oxcdf::Result<()> {
    use object_store::path::Path;
    use oxcdf::OpenOptions;

    let _ = AsyncFile::open_store_with(
        store,
        Path::from("argo.nc"),
        OpenOptions::new()
            .io_request_size(256 * 1024)
            .io_cache_bytes(128 << 20),
    )
    .await?;
    Ok(())
}
