//! The README must not drift from the API.
//!
//! Every example in the README appears here. A test that runs uses the test
//! corpus. A test that needs a network only has to compile, so it lives in a
//! function that nothing calls.

#[cfg(feature = "async")]
use std::sync::Arc;

#[cfg(feature = "async")]
use oxcdf::{FileSource, SyncAsAsync};
#[cfg(all(feature = "async", feature = "object-store"))]
use oxcdf::AsyncFile;

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
        println!("{} {:?} {:?} {:?}", v.name, v.vartype(), v.shape, v.dimensions);
    }

    let temp = file.variable("analysed_sst").expect("a variable");
    println!("{:?}", temp.attribute("units").map(|a| a.value.as_text()));
    println!("{:?}", temp.attribute("_FillValue").map(|a| a.value.as_f64()));

    let _all = temp.get_values::<f32, _>(oxcdf::Extents::All)?;
    let _part = temp.get_values::<f32, _>([0..1, 0..4, 0..4])?;
    let _one = temp.get_value::<f32, _>([0, 3, 3])?;

    // A path into a subgroup resolves the same way. This file has no subgroup.
    assert!(file.variable("/forecast/TEMP").is_none());
    Ok(())
}

// ─── "Select the elements" ─────────────────────────────────────────────────

#[test]
fn every_selection_form_in_the_readme_works() -> oxcdf::Result<()> {
    use oxcdf::{Extent, Extents};

    let Some(path) = corpus("legacy_v1_objheader.h5") else {
        return Ok(());
    };
    let file = oxcdf::open(&path)?;
    let var = file.variable("/contig_f64").unwrap(); // shape [40, 6]

    var.get_values::<f64, _>(Extents::All)?;
    var.get_values::<f64, _>(..)?;
    var.get_values::<f64, _>([0..8, 2..5])?;
    var.get_values::<f64, _>([0, 3])?;
    var.get_values::<f64, _>([2.., 5..])?;
    var.get_values::<f64, _>([..8, ..5])?;
    var.get_values::<f64, _>([Extent::Index(3), (0..6).into()])?;
    var.get_values::<f64, _>(([0usize, 2].as_slice(), [8usize, 3].as_slice()))?;
    Ok(())
}

// ─── "Types" ───────────────────────────────────────────────────────────────

#[test]
fn the_types_example_works() -> oxcdf::Result<()> {
    let Some(path) = corpus("legacy_v1_objheader.h5") else {
        return Ok(());
    };
    let file = oxcdf::open(&path)?;
    let temp = file.variable("/contig_f32be").unwrap();

    assert_eq!(temp.vartype(), oxcdf::DType::Float(4));
    let exact = temp.get_values::<f32, _>(..)?;
    let wide = temp.get_values::<f64, _>(..)?;
    assert_eq!(exact.len(), wide.len());

    // A string read as a number names the stored type.
    let err = file
        .variable("/fixed_strings")
        .unwrap()
        .get_values::<f64, _>(..)
        .unwrap_err();
    assert!(matches!(err, oxcdf::Error::TypeMismatch { .. }), "got {err:?}");
    Ok(())
}

// ─── "Strings" ─────────────────────────────────────────────────────────────

#[test]
fn the_string_variable_example_works() -> oxcdf::Result<()> {
    let Some(path) = corpus("vlen_strings.nc") else {
        return Ok(());
    };
    let file = oxcdf::open(&path)?;
    let names = file.variable("station_name").unwrap();
    assert_eq!(names.vartype(), oxcdf::DType::String);

    let all = names.get_strings(..)?;
    let one = names.get_string([0])?;
    assert_eq!(one, all[0]);
    Ok(())
}

#[test]
fn the_char_variable_example_works() -> oxcdf::Result<()> {
    let Some(path) = corpus("wod_ctd_1964.nc") else {
        return Ok(());
    };
    let file = oxcdf::open(&path)?;
    let v = file.variable("country").unwrap();
    assert_eq!(v.vartype(), oxcdf::DType::Char);
    assert_eq!(v.shape, vec![47, 40]);

    let width = *v.shape.last().unwrap() as usize;
    let joined: Vec<String> = v
        .get_strings(..)?
        .chunks(width)
        .map(|row| row.concat().trim_end_matches('\0').to_string())
        .collect();
    assert_eq!(joined[0], "GREAT BRITAIN");

    // The raw bytes are the other route the page names.
    assert_eq!(v.read()?.as_bytes().len(), 47 * 40);
    Ok(())
}

// ─── "Other reads" ─────────────────────────────────────────────────────────

#[test]
fn the_values_example_works() -> oxcdf::Result<()> {
    let Some(path) = corpus("legacy_v1_objheader.h5") else {
        return Ok(());
    };
    let file = oxcdf::open(&path)?;
    let temp = file.variable("/contig_f32be").unwrap();

    let values = temp.read()?;
    println!("{:?} {:?}", values.dtype(), values.shape());
    let _numbers: Vec<f32> = values.get()?;
    let _bytes = values.as_bytes();

    let _text = file.variable("/fixed_strings").unwrap().read()?.to_strings()?;
    Ok(())
}

#[test]
fn the_dtype_table_matches_the_corpus() -> oxcdf::Result<()> {
    use oxcdf::DType;

    let Some(path) = corpus("wod_ctd_1964.nc") else {
        return Ok(());
    };
    let file = oxcdf::open(&path)?;

    // Every variable's type must be one the table names, and `size` must agree.
    for v in file.variables() {
        match v.vartype() {
            DType::Int(n) | DType::Uint(n) | DType::Float(n) => {
                assert_eq!(v.vartype().size(), Some(n as usize), "{}", v.path);
                assert!(v.vartype().is_integer() || v.vartype().is_float());
            }
            DType::Char => {
                assert_eq!(v.vartype().size(), Some(1), "{}", v.path);
                assert!(v.vartype().is_text());
            }
            DType::String => assert_eq!(v.vartype().size(), None, "{}", v.path),
            other => panic!("{} has an unexpected type {other:?}", v.path),
        }
    }
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

    let a = temp.get::<f64, _>(..)?;
    assert_eq!(a.shape(), &[40, 6]);
    println!("{}", a[[0, 0]]);

    let row = a.index_axis(ndarray::Axis(0), 0);
    assert_eq!(row.len(), 6);

    let b = temp.get::<f64, _>([5..15, 2..5])?;
    assert_eq!(b.shape(), &[10, 3]);

    let counts = file
        .variable("/chunked_i32")
        .unwrap()
        .get::<i32, _>(..)?;
    assert_eq!(counts.shape(), &[40, 6]);

    let names = file
        .variable("/fixed_strings")
        .unwrap()
        .read()?
        .to_array_strings()?;
    assert_eq!(names.shape(), &[5]);
    Ok(())
}

// ─── "The interface" table ─────────────────────────────────────────────────

#[test]
fn the_get_values_example_works() -> oxcdf::Result<()> {
    let Some(path) = corpus("legacy_v1_objheader.h5") else {
        return Ok(());
    };
    let file = oxcdf::open(&path)?;
    let temp = file.variable("/contig_f64").unwrap();

    let all = temp.get_values::<f64, _>(oxcdf::Extents::All)?;
    assert_eq!(all.len(), 240);
    assert_eq!(temp.get_values::<f64, _>(..)?.len(), 240);

    let block = temp.get_values::<f64, _>([0..8, 2..5])?;
    assert_eq!(block.len(), 24);

    let one = temp.get_value::<f64, _>([0, 0])?;
    assert_eq!(one, all[0]);
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
    let _values = temp.get_values::<f64, _>(..).await?;
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
    let _ = temp.vartype();
    let _ = temp.shape.clone();
    let _ = temp.dimensions.clone();

    // Only the values await.
    let _ = temp.read().await?;
    let _ = temp.get_values::<f64, _>([0..2, 0..2]).await?;
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

    let a = temp.get::<f64, _>(..).await?;
    assert_eq!(a.shape(), &[40, 6]);
    let b = temp.get::<f64, _>([5..15, 2..5]).await?;
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
        println!("{} {:?} {:?} {:?}", v.name, v.vartype(), v.shape, v.dimensions);
    }

    let temp = file.variable("TEMP").unwrap();
    println!("{:?}", temp.attribute("units").unwrap().value.as_text());

    let _all = temp.get_values::<f32, _>(oxcdf::Extents::All).await?;
    let _part = temp.get_values::<f32, _>([0..8, 10..30]).await?;
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

#[test]
fn the_ragged_array_example_works() -> oxcdf::Result<()> {
    let Some(path) = corpus("vlen_seq.nc") else {
        return Ok(());
    };
    let file = oxcdf::open(&path)?;
    let seqs = file.variable("rows").unwrap().read()?.to_sequences::<f32>()?;
    assert!(!seqs.is_empty());
    Ok(())
}

#[test]
fn the_char_raw_bytes_example_works() -> oxcdf::Result<()> {
    let Some(path) = corpus("wod_ctd_1964.nc") else {
        return Ok(());
    };
    let file = oxcdf::open(&path)?;
    let bytes = file.variable("country").unwrap().get_raw_values(..)?;
    assert_eq!(bytes.len(), 47 * 40);
    Ok(())
}
