//! A rough throughput comparison against netcdf-c. Not a microbenchmark: it
//! measures the thing that matters, which is many threads reading whole
//! variables out of the same file at once.
#![cfg(feature = "diff-tests")]

use std::sync::Arc;
use std::time::Instant;

use oxcdf::netcdf::NetcdfFile;

const PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_files/test_file.nc"
);

/// Read every numeric variable once, returning how many values were decoded.
fn native_pass(file: &NetcdfFile) -> usize {
    let mut n = 0;
    for v in file.variables() {
        if !v.is_readable() || v.shape.is_empty() {
            continue;
        }
        if let Ok(values) = v.read() {
            if let Ok(f) = values.get::<f64>() {
                n += f.len();
            }
        }
    }
    n
}

fn netcdf_c_pass(file: &netcdf::File) -> usize {
    use netcdf::types::{FloatType, NcVariableType};
    let mut n = 0;
    for v in file.variables() {
        match v.vartype() {
            NcVariableType::Float(FloatType::F32) => {
                if let Ok(d) = v.get_values::<f32, _>(netcdf::Extents::All) {
                    n += d.len();
                }
            }
            NcVariableType::Float(FloatType::F64) => {
                if let Ok(d) = v.get_values::<f64, _>(netcdf::Extents::All) {
                    n += d.len();
                }
            }
            _ => {}
        }
    }
    n
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn throughput_against_netcdf_c() {
    const PASSES: usize = 200;

    // ── single threaded ──────────────────────────────────────────────────
    let native = NetcdfFile::open(PATH).unwrap();
    let start = Instant::now();
    let mut values = 0;
    for _ in 0..PASSES {
        values += native_pass(&native);
    }
    let native_1t = start.elapsed();

    let refc = netcdf::open(PATH).unwrap();
    let start = Instant::now();
    let mut ref_values = 0;
    for _ in 0..PASSES {
        ref_values += netcdf_c_pass(&refc);
    }
    let c_1t = start.elapsed();

    println!("\n=== single thread, {PASSES} passes ===");
    println!("  netcdf-c  {c_1t:>10.2?}  ({ref_values} values)");
    println!("  native    {native_1t:>10.2?}  ({values} values)");
    println!(
        "  ratio     {:.2}x",
        c_1t.as_secs_f64() / native_1t.as_secs_f64()
    );

    // ── cold: no chunk cache ─────────────────────────────────────────────
    // The cache flatters a benchmark that reads the same file repeatedly. A
    // scan that touches each chunk once gets nothing from it, so measure that
    // separately to see the algorithmic win on its own.
    let uncached = NetcdfFile::from_hdf5(
        oxcdf::index::Hdf5File::open(PATH)
            .unwrap()
            .with_cache(None),
    )
    .unwrap();
    let start = Instant::now();
    for _ in 0..PASSES {
        native_pass(&uncached);
    }
    let native_nocache = start.elapsed();
    println!(
        "  native (no chunk cache) {native_nocache:>10.2?}   ratio vs netcdf-c {:.2}x",
        c_1t.as_secs_f64() / native_nocache.as_secs_f64()
    );

    // ── many threads on the same file ────────────────────────────────────
    for threads in [2usize, 4, 8] {
        let per_thread = PASSES / threads;

        let shared = Arc::new(NetcdfFile::open(PATH).unwrap());
        let start = Instant::now();
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let f = Arc::clone(&shared);
                std::thread::spawn(move || {
                    let mut n = 0;
                    for _ in 0..per_thread {
                        n += native_pass(&f);
                    }
                    n
                })
            })
            .collect();
        let _: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        let native_n = start.elapsed();

        // netcdf-c handles are not shareable, so each thread opens its own.
        // Every call still serialises on the crate's process-global mutex.
        let start = Instant::now();
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                std::thread::spawn(move || {
                    let f = netcdf::open(PATH).unwrap();
                    let mut n = 0;
                    for _ in 0..per_thread {
                        n += netcdf_c_pass(&f);
                    }
                    n
                })
            })
            .collect();
        let _: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        let c_n = start.elapsed();

        println!("\n=== {threads} threads, {PASSES} passes total ===");
        println!("  netcdf-c  {c_n:>10.2?}");
        println!("  native    {native_n:>10.2?}");
        println!(
            "  ratio     {:.2}x   (native scaling vs its own 1-thread: {:.2}x)",
            c_n.as_secs_f64() / native_n.as_secs_f64(),
            native_1t.as_secs_f64() / native_n.as_secs_f64()
        );
    }
}
