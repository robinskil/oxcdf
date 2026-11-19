#![cfg(feature = "write")]
//! Random calls of the netCDF writer give the file of a model.
//!
//! The test shares its code with the `netcdf_write` fuzz target, so it also
//! runs where `cargo fuzz` does not.
//!
//! Run with:
//!
//! ```bash
//! cargo test -p oxcdf --features write --test write_calls
//! ```

#[path = "../../../fuzz/netcdf_ops.rs"]
mod netcdf_ops;

/// The number of random inputs.
const INPUTS: u64 = 20_000;

/// The next value of a xorshift generator. The state is never zero.
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn random_calls_give_the_file_of_the_model() {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    for input in 0..INPUTS {
        let len = (next(&mut state) % 200) as usize;
        let data: Vec<u8> = (0..len).map(|_| next(&mut state) as u8).collect();
        let run = std::panic::catch_unwind(|| netcdf_ops::run(&data));
        assert!(run.is_ok(), "input {input} fails: {data:?}");
    }
}
