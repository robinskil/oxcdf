//! Fuzz the writer against a model.
//!
//! The input bytes select writer calls: groups, datasets, writes and
//! attributes, many with wrong arguments. The target closes the file and reads
//! it with the reader.
//!
//! An `Err` from a writer call is a pass. A panic, a file that the reader
//! rejects, or a value that differs from the model is a defect.

#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../write_ops.rs"]
mod write_ops;

fuzz_target!(|data: &[u8]| write_ops::run(data));
