//! Fuzz the filter pipeline directly.
//!
//! A chunk arrives compressed. The reader inflates it, unshuffles it and checks
//! it. `zstd` is a C library, so this target is the one that most needs the
//! address sanitiser that `cargo fuzz` turns on by default.
//!
//! Reaching this code through a whole file needs a valid superblock, a valid
//! object header and a valid chunk index. The fuzzer rarely builds all three,
//! so this target feeds the decoders directly.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxcdf_hdf5::filters;

/// Keep an output claim inside a size the fuzzer can afford.
const MAX_OUTPUT: usize = 1 << 22;

fuzz_target!(|data: &[u8]| {
    // The first byte picks a decoder. The rest is the compressed block.
    let Some((&selector, block)) = data.split_first() else {
        return;
    };
    if block.is_empty() {
        return;
    }

    // A real chunk names its decoded size. Derive a plausible one rather than
    // trusting a fuzzer-chosen number, which would only allocate.
    let expected = (block.len() * 4).min(MAX_OUTPUT);
    // A type size of 0 is not a valid filter argument, so keep it in range.
    let type_size = (block[0] % 8) as usize + 1;

    match selector % 6 {
        0 => {
            let _ = filters::inflate(block, expected);
        }
        1 => {
            let _ = filters::unzstd(block, expected);
        }
        2 => {
            let _ = filters::unblosc(block);
        }
        3 => {
            let _ = filters::unblosclz(block, expected);
        }
        4 => {
            let _ = filters::unshuffle(block, type_size);
            let _ = filters::unbitshuffle(block, type_size);
        }
        _ => {
            let _ = filters::verify_and_strip_fletcher32(block.to_vec());
        }
    }
});
