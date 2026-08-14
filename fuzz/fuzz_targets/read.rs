//! Fuzz the value read, not only the parse.
//!
//! This is the deeper target. It reaches the chunk index, the filter pipeline
//! and the global heap. It also reaches the C zstd decoder, which is the one
//! part of the crate that is not safe Rust.
//!
//! Any `Err` is a pass. A panic, a hang or a huge allocation is a defect.

#![no_main]

use std::sync::Arc;

use libfuzzer_sys::fuzz_target;
use oxcdf::netcdf::NetcdfFile;
use oxcdf::MemorySource;

/// Stop a corrupt shape from asking for more memory than the input could hold.
///
/// A file that claims a huge variable is a real finding, but it belongs to the
/// `open` target. Here it only starves the fuzzer.
const MAX_ELEMENTS: u64 = 1 << 20;

/// The same bound in bytes.
///
/// An element width comes from the file just as the shape does, so counting
/// elements alone does not bound the read. A variable of a million elements
/// that are 30 KB each asks for 30 GB, and a 10 KB input can declare exactly
/// that. Reading it is the caller's decision to make, from the length and the
/// type, which is what this does.
const MAX_BYTES: u64 = 1 << 26;

/// Read at most this many variables, so one input cannot dominate the run.
const MAX_VARIABLES: usize = 16;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let Ok(file) = NetcdfFile::from_source(Arc::new(MemorySource::new(data.to_vec()))) else {
        return;
    };

    for v in file.variables().into_iter().take(MAX_VARIABLES) {
        // The widest output any read below asks for, so one bound covers them
        // all: the stored width, or eight bytes once decoded as `f64`.
        let width = (v.datatype().size as u64).max(8);
        if v.len() > MAX_ELEMENTS || v.len().saturating_mul(width) > MAX_BYTES {
            continue;
        }
        // Each decode path, because each one reaches different code.
        let _ = v.get_raw_values(..);
        let _ = v.get_values::<f64, _>(..);
        let _ = v.get_values::<i32, _>(..);
        let _ = v.get_strings(..);

        // A selection exercises the run walker and the chunk arithmetic, which
        // a whole read can skip.
        if !v.shape.is_empty() {
            let corner: Vec<std::ops::Range<usize>> =
                v.shape.iter().map(|&n| 0..(n as usize).min(2)).collect();
            let _ = v.get_values::<f64, _>(corner.as_slice());
        }
    }
});
