# The workspace split

The crate divides into two layers. This note records the layout, the one problem
that had to be solved first, and what the split changed.

## The layers

`oxcdf-hdf5` reads an HDF5 container. It knows nothing about netCDF.

```
container  cursor    checksum   error      source
async_source          object_store_source   replay
btree1     btree2     chunk_index           fractal
heap       message/   objheader  superblock symbol_table
context    dense      cache      filters    io
dtype      index      read
```

`oxcdf` applies the netCDF conventions on top.

```
netcdf     classic    extent     async_netcdf  lib
```

`oxcdf` depends on `oxcdf-hdf5` and re-exports it, under the name `hdf5`. A
caller therefore needs one dependency, as before.

## The problem, now solved

`read.rs` sits in the lower layer. It decoded values through `netcdf::Element`
and `netcdf::DType`, which sat in the upper layer. The two layers referred to
each other, so no split was possible.

`Element` and `DType` moved to `dtype.rs`, below the netCDF layer. `netcdf`
re-exports both names, so nothing changed for a caller.

The crate boundary now enforces this. The lower crate cannot name the upper one.
A future cycle fails to compile instead of passing unnoticed.

## What crosses the boundary

`classic.rs` sits in the upper crate. It uses `oxcdf_hdf5::message::Datatype`.
That direction is correct: `Datatype` is the canonical element descriptor, not
an HDF5 detail. `NcType::to_datatype` maps a classic type onto it, and one
decoder then serves both containers.

`Container` and `detect_container` sit in the lower crate, in `container.rs`.
Both `classic.rs` and `netcdf.rs` use them.

## Where the parts went

The HDF5 modules moved from `src/hdf5/` to the root of the lower crate. A path
reads `oxcdf_hdf5::message::Datatype`, not `oxcdf_hdf5::hdf5::message::Datatype`.

`replay` was `pub(crate)`. `async_netcdf.rs` calls it across the boundary, so it
is now `pub`. `ReplaySource` stays private.

Each test file moved to the crate it exercises. A test that names a netCDF type
went to `oxcdf`, even where it also reads the HDF5 layer. `open_cost.rs`
measures `Hdf5File`, but it compares against netcdf-c, so it went to `oxcdf`
with the other `diff-tests` files.

## Watch these

- `test_files/` stays at the workspace root. Both crates read it through
  `env!("CARGO_MANIFEST_DIR")` and `../../test_files`. A published `.crate`
  therefore holds no corpus, so the tests find no files and skip.
- `crates/oxcdf/README.md` is a symbolic link to the root `README.md`. Cargo
  follows it when it packages the crate.
- `[package.metadata.docs.rs]` and `rustdoc-args` sit in both manifests. Without
  them the lower crate's docs.rs page loses the asynchronous items.
- `#![cfg_attr(docsrs, feature(doc_cfg))]` is needed in both crate roots.
- `rust-toolchain.toml` stays at the workspace root.
- The `netcdf` git dependency belongs to the upper crate only.
- A feature of the upper crate turns on the matching feature of the lower one.
  `oxcdf/async` implies `oxcdf-hdf5/async`.

## Verify

```bash
cargo test --workspace --features "oxcdf/diff-tests,oxcdf/object-store,oxcdf/ndarray,oxcdf/async"
```

396 tests pass. Clippy reports nothing across the feature matrix. The doc build
reports no broken link.
