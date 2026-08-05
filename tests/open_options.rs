//! Opening with an explicit I/O request size and cache budget.

use oxcdf::index::{Hdf5File, OpenOptions};
use oxcdf::io::IoConfig;
use oxcdf::netcdf::NetcdfFile;
use oxcdf::read::{read_hyperslab, Hyperslab};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_files/legacy_v1_objheader.h5"
);

#[test]
fn the_request_size_and_cache_budget_are_honoured() {
    let file = Hdf5File::open_with(
        FIXTURE,
        OpenOptions::new()
            .io_request_size(256 * 1024)
            .io_cache_bytes(8 << 20),
    )
    .unwrap();

    let cache = file.io_cache().expect("a cache was requested");
    assert_eq!(cache.page_size(), 256 * 1024);
    // 8 MiB in 256 KiB pages.
    assert_eq!(cache.capacity_bytes(), 8 << 20);
}

#[test]
fn values_are_identical_across_request_sizes() {
    let want = {
        let f = Hdf5File::open(FIXTURE).unwrap();
        let d = f.dataset("/chunked_i32").unwrap();
        read_hyperslab(f.ctx(), d, &Hyperslab::all(&d.shape))
            .unwrap()
            .to_i64(d)
            .unwrap()
    };

    // A page size below, around and far above the file size must all agree.
    for size in [512usize, 4096, 64 * 1024, 256 * 1024, 4 << 20] {
        let f = Hdf5File::open_with(FIXTURE, OpenOptions::new().io_request_size(size)).unwrap();
        let d = f.dataset("/chunked_i32").unwrap();
        let got = read_hyperslab(f.ctx(), d, &Hyperslab::all(&d.shape))
            .unwrap()
            .to_i64(d)
            .unwrap();
        assert_eq!(got, want, "request size {size} changed the values");
    }
}

#[test]
fn caches_can_be_turned_off_entirely() {
    let file = Hdf5File::open_with(
        FIXTURE,
        OpenOptions::new().without_io_cache().without_chunk_cache(),
    )
    .unwrap();
    assert!(file.io_cache().is_none());
    assert!(file.cache().is_none());

    // Reads still work, they just do more I/O.
    let d = file.dataset("/contig_f64").unwrap();
    assert_eq!(
        read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
            .unwrap()
            .to_f64(d)
            .unwrap()
            .len(),
        240
    );
}

#[test]
fn the_remote_preset_sets_all_three_knobs() {
    let file = Hdf5File::open_with(FIXTURE, OpenOptions::remote()).unwrap();
    assert_eq!(file.io_cache().unwrap().page_size(), 256 * 1024);
    assert_eq!(file.io_cache().unwrap().capacity_bytes(), 128 << 20);
    assert_eq!(file.io(), IoConfig::REMOTE);
}

#[test]
fn readahead_is_configurable_at_open() {
    let file = Hdf5File::open_with(FIXTURE, OpenOptions::new().readahead(0)).unwrap();
    assert_eq!(file.cache().unwrap().readahead(), 0);

    let file = Hdf5File::open_with(FIXTURE, OpenOptions::new().readahead(16)).unwrap();
    assert_eq!(file.cache().unwrap().readahead(), 16);
}

#[test]
fn the_netcdf_layer_forwards_the_options() {
    let file = NetcdfFile::open_with(
        FIXTURE,
        OpenOptions::new().io_request_size(128 * 1024).io_cache_bytes(4 << 20),
    )
    .unwrap();
    assert_eq!(file.hdf5().io_cache().unwrap().page_size(), 128 * 1024);
    assert_eq!(file.variables().len(), 5);
}
