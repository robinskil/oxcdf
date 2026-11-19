#![cfg(all(feature = "write", feature = "object-store"))]
//! The netCDF writer uploads a file to an object store at the close.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p oxcdf --features write,object-store --test write_store
//! ```

use std::sync::Arc;

use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use oxcdf::{FileMut, MemorySource, NetcdfFile};

/// Add `count` doubles to `file`. They are larger than one upload part for a high count.
fn fill(file: &mut FileMut, count: usize) -> Vec<f64> {
    let values: Vec<f64> = (0..count).map(|i| i as f64 * 0.25).collect();
    file.add_dimension("x", count).unwrap();
    let mut v = file.add_variable::<f64>("v", &["x"]).unwrap();
    v.put_values(&values, ..).unwrap();
    values
}

async fn read_back(store: &dyn ObjectStore, location: &Path) -> Vec<f64> {
    let bytes = store.get(location).await.unwrap().bytes().await.unwrap();
    let file = NetcdfFile::from_source(Arc::new(MemorySource::new(bytes.to_vec()))).unwrap();
    let v = file.variable("v").unwrap();
    v.get_values::<f64, _>(..).unwrap()
}

#[tokio::test]
async fn a_small_file_in_memory_goes_up_in_one_put() {
    let store = InMemory::new();
    let location = Path::from("data/small.nc");
    let mut file = oxcdf::create_in_memory().unwrap();
    let values = fill(&mut file, 100);
    file.close_to_store(&store, &location).await.unwrap();
    assert_eq!(read_back(&store, &location).await, values);
}

#[tokio::test]
async fn a_large_file_in_memory_goes_up_in_parts() {
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let location = Path::from("data/large.nc");
    let mut file = oxcdf::create_in_memory().unwrap();
    // 20 MiB of values need three parts of 8 MiB.
    let values = fill(&mut file, 20 << 17);
    file.close_to_store(store.as_ref(), &location)
        .await
        .unwrap();
    assert_eq!(read_back(store.as_ref(), &location).await, values);
}

#[tokio::test]
async fn a_file_on_disk_goes_up_and_leaves_no_local_file() {
    let dir = std::env::temp_dir().join(format!("oxcdf-store-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let store = InMemory::new();
    let location = Path::from("data/spooled.nc");
    let mut file = oxcdf::create(dir.join("spool.nc")).unwrap();
    let values = fill(&mut file, 20 << 17);
    file.close_to_store(&store, &location).await.unwrap();
    assert_eq!(read_back(&store, &location).await, values);
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_upload_can_move_to_another_thread() {
    fn send<T: Send>(_: &T) {}
    let store = InMemory::new();
    let location = Path::from("data/send.nc");
    let file = oxcdf::create_in_memory().unwrap();
    send(&file.close_to_store(&store, &location));
}
