//! Fuzz the metadata parse.
//!
//! An open walks the superblock, the object headers, the heaps and the B-trees.
//! Every field there comes from the file, so every field is attacker input.
//!
//! Any `Err` is a pass. A panic, a hang or a huge allocation is a defect.

#![no_main]

use std::sync::Arc;

use libfuzzer_sys::fuzz_target;
use oxcdf::netcdf::NetcdfFile;
use oxcdf::MemorySource;

fuzz_target!(|data: &[u8]| {
    // An empty or tiny buffer cannot reach the parser. Skip it and keep the
    // corpus focused on inputs that do.
    if data.len() < 8 {
        return;
    }
    let _ = NetcdfFile::from_source(Arc::new(MemorySource::new(data.to_vec())));
});
