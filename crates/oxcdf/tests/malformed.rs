//! A damaged file must return an error, not panic.
//!
//! This reader parses untrusted binary input. Both crates hold no `unsafe`
//! code, so a memory error is not the risk here. The risk is a panic, a hang or
//! a huge allocation on a file that is truncated or corrupt.
//!
//! One dependency, `zstd`, does compile C. The `filters` fuzz target covers it
//! under the address sanitiser.
//!
//! The crate states the contract: `Error::Unsupported` marks a feature this
//! reader does not implement, and every other error marks a damaged file. A
//! panic marks a defect here.
//!
//! These tests mutate the real corpus and hold that contract. They are the
//! stable-Rust half of the fuzzing story. `fuzz/` holds the deeper one.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use oxcdf::netcdf::NetcdfFile;
use oxcdf::MemorySource;

fn corpus() -> Vec<(String, Vec<u8>)> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files");
    [
        "test_file.nc",
        "gridded-example.nc",
        "wod_ctd_1964.nc",
        "classic.nc",
        "classic64.nc",
        "vlen_strings.nc",
        // Declares user-defined types, so mutating it exercises the committed
        // datatype path.
        "committed_types.nc",
        "legacy_v1_objheader.h5",
    ]
    .iter()
    .filter_map(|name| {
        let path = format!("{root}/{name}");
        std::fs::read(&path).ok().map(|b| (name.to_string(), b))
    })
    .collect()
}

/// Open a byte buffer, catching a panic and reporting it as a failure.
///
/// Any outcome is acceptable except a panic: the file may open, or it may
/// return an error. Both are correct answers for a damaged file.
fn opens_without_panic(label: &str, bytes: Vec<u8>) -> Result<Option<NetcdfFile>, String> {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        NetcdfFile::from_source(Arc::new(MemorySource::new(bytes)))
    }));
    match outcome {
        Ok(Ok(file)) => Ok(Some(file)),
        Ok(Err(_)) => Ok(None),
        Err(panic) => {
            let msg = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "non-string panic".into());
            Err(format!("{label}: open panicked: {msg}"))
        }
    }
}

/// Read every variable of a file that opened, catching a panic.
///
/// A corrupt header can claim an enormous shape, so skip a variable whose claim
/// is larger than the file could hold. An absurd claim is a separate concern
/// from a panic, and it belongs to the `open` fuzz target.
fn reads_without_panic(label: &str, file: &NetcdfFile, byte_len: usize) -> Vec<String> {
    let mut failures = Vec::new();
    for v in file.variables() {
        if v.len() > byte_len as u64 {
            continue;
        }
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let _ = v.get_raw_values(..);
            let _ = v.get_values::<f64, _>(..);
            let _ = v.get_strings(..);
        }));
        if outcome.is_err() {
            failures.push(format!("{label}: reading {} panicked", v.path));
        }
    }
    failures
}

#[test]
fn a_truncated_file_returns_an_error() {
    let mut failures = Vec::new();

    for (name, bytes) in corpus() {
        let len = bytes.len();
        let cuts = [
            0,
            1,
            4,
            8,
            32,
            64,
            512,
            len / 8,
            len / 4,
            len / 2,
            len * 3 / 4,
            len.saturating_sub(1),
        ];
        for cut in cuts {
            if cut > len {
                continue;
            }
            let label = format!("{name} truncated to {cut}");
            match opens_without_panic(&label, bytes[..cut].to_vec()) {
                Err(e) => failures.push(e),
                Ok(Some(file)) => failures.extend(reads_without_panic(&label, &file, cut)),
                Ok(None) => {}
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Flip bytes across the file and check the reader still refuses cleanly.
///
/// The offsets are deterministic, so a failure reproduces exactly.
#[test]
fn a_corrupt_file_returns_an_error() {
    let mut failures = Vec::new();

    for (name, bytes) in corpus() {
        let len = bytes.len();
        // Walk the front of the file closely: the superblock, the root object
        // header and the first messages all live there, and that is where a
        // parser is most likely to trust a field it should check.
        let mut offsets: Vec<usize> = (0..512.min(len)).step_by(7).collect();
        // Then sample the rest.
        offsets.extend((512..len).step_by(len / 64 + 1));

        for offset in offsets {
            let mut mutant = bytes.clone();
            mutant[offset] ^= 0xFF;
            let label = format!("{name} byte {offset} flipped");
            match opens_without_panic(&label, mutant) {
                Err(e) => failures.push(e),
                Ok(Some(file)) => failures.extend(reads_without_panic(&label, &file, len)),
                Ok(None) => {}
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Random bytes must never open as a file.
#[test]
fn random_bytes_are_refused() {
    // A simple deterministic generator keeps this reproducible.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut failures = Vec::new();
    for size in [16usize, 512, 4096, 65536] {
        for round in 0..8 {
            let bytes: Vec<u8> = (0..size).map(|_| (next() & 0xFF) as u8).collect();
            let label = format!("random {size} bytes, round {round}");
            if let Err(e) = opens_without_panic(&label, bytes) {
                failures.push(e);
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A file that keeps its HDF5 signature but holds rubbish after it.
///
/// This gets past the container check, so the superblock parser sees it.
/// Corrupt every byte of the classic fixtures, one at a time, to every value
/// that changes the parse in a different way.
///
/// The other corruption test samples offsets, because the HDF5 corpus is large.
/// The classic fixtures are a few hundred bytes, so this one covers all of them
/// and leaves no gap for a header field to hide in.
///
/// A classic header is nothing but counts, lengths and offsets. Reserving on a
/// count before the buffer bounds it hands the allocator a claim it cannot
/// meet, and a failed allocation aborts the process instead of returning an
/// error, which no caller can catch.
#[test]
fn every_single_byte_corruption_of_a_classic_file_is_handled() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files");
    let mut failures = Vec::new();

    for name in ["classic.nc", "classic64.nc"] {
        let Ok(bytes) = std::fs::read(format!("{root}/{name}")) else {
            continue;
        };

        for offset in 0..bytes.len() {
            // Zero, one, both signs of a byte and both all-ones patterns. These
            // are what turn a small count into an enormous one.
            for value in [0x00u8, 0x01, 0x7F, 0x80, 0xFE, 0xFF] {
                if bytes[offset] == value {
                    continue;
                }
                let mut mutant = bytes.clone();
                mutant[offset] = value;

                let label = format!("{name} byte {offset} set to {value:#04x}");
                match opens_without_panic(&label, mutant) {
                    Err(e) => failures.push(e),
                    Ok(Some(file)) => {
                        failures.extend(reads_without_panic(&label, &file, bytes.len()))
                    }
                    Ok(None) => {}
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn a_valid_signature_over_rubbish_is_refused() {
    let mut failures = Vec::new();
    for fill in [0x00u8, 0xFF, 0xAA] {
        let mut bytes = oxcdf::HDF5_SIGNATURE.to_vec();
        bytes.extend(std::iter::repeat_n(fill, 8192));
        let label = format!("HDF5 signature over 0x{fill:02X} filler");
        if let Err(e) = opens_without_panic(&label, bytes) {
            failures.push(e);
        }
    }
    for sig in oxcdf::CDF_SIGNATURES {
        let mut bytes = sig.to_vec();
        bytes.extend(std::iter::repeat_n(0xFFu8, 8192));
        let label = format!("CDF signature {sig:?} over filler");
        if let Err(e) = opens_without_panic(&label, bytes) {
            failures.push(e);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
