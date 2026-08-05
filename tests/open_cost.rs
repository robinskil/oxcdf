#![cfg(feature = "diff-tests")]
use oxcdf::index::Hdf5File;
use std::time::Instant;

const PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_files/test_file.nc"
);

#[test]
#[ignore = "measurement"]
fn open_cost() {
    const N: usize = 200;

    let f = Hdf5File::open(PATH).unwrap();
    let prepared_at_open = f.datasets().iter().filter(|d| d.is_prepared()).count();
    // Resolving every index is now opt-in; count what it would cost.
    f.prepare_all().unwrap();
    let chunked = f
        .datasets()
        .iter()
        .filter(|d| d.resolved_chunks().is_some())
        .count();
    println!(
        "{} datasets, {chunked} chunked, {prepared_at_open} indexes resolved at open",
        f.datasets().len()
    );

    let start = Instant::now();
    for _ in 0..N {
        std::hint::black_box(Hdf5File::open(PATH).unwrap());
    }
    println!("native open (lazy)  {:?} per open", start.elapsed() / N as u32);

    let start = Instant::now();
    for _ in 0..N {
        let f = Hdf5File::open(PATH).unwrap();
        f.prepare_all().unwrap();
        std::hint::black_box(f);
    }
    println!("native open + all   {:?} per open", start.elapsed() / N as u32);

    let start = Instant::now();
    for _ in 0..N {
        std::hint::black_box(netcdf::open(PATH).unwrap());
    }
    println!("netcdf-c open     {:?} per open", start.elapsed() / N as u32);

    // One whole-variable read, for scale.
    let f = Hdf5File::open(PATH).unwrap();
    let d = f.dataset("/TEMP").unwrap();
    d.prepare(f.ctx()).unwrap();
    let slab = oxcdf::read::Hyperslab::all(&d.shape);
    let start = Instant::now();
    for _ in 0..N {
        std::hint::black_box(
            oxcdf::read::read_hyperslab(f.ctx(), d, &slab).unwrap(),
        );
    }
    println!("one TEMP read     {:?}", start.elapsed() / N as u32);
}
