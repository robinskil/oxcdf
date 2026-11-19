//! Fuzz the netCDF writer.
//!
//! The input is a list of calls: add a dimension, a variable or a group, set
//! a definition, write values and put attributes. A panic is a defect. A file
//! that differs from the model is a defect.
//!
//! ```bash
//! cargo +nightly fuzz run netcdf_write -- -max_total_time=60
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../netcdf_ops.rs"]
mod netcdf_ops;

fuzz_target!(|data: &[u8]| netcdf_ops::run(data));
