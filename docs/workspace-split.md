# The workspace split

The crate divides into two layers. This note records the plan, the one problem
that had to be solved first, and the order of the work.

## The layers

`oxcdf-hdf5` reads an HDF5 container. It knows nothing about netCDF.

```
error      source      async_source   cursor    checksum
filters    cache       io             replay    dtype
hdf5/      index       read
```

`oxcdf` applies the netCDF conventions on top.

```
netcdf     classic     extent         async_file    lib
```

`oxcdf` depends on `oxcdf-hdf5` and re-exports what a caller needs. A caller
therefore sees one crate, as it does now.

## The problem, now solved

`read.rs` sits in the lower layer. It decoded values through
`netcdf::Element` and `netcdf::DType`, which sit in the upper layer. The two
layers referred to each other, so no split was possible.

`Element` and `DType` now live in `src/dtype.rs`, below the netCDF layer.
`netcdf` re-exports both names, so nothing changed for a caller.

Check the split stays possible:

```bash
grep -rn "crate::netcdf\|crate::classic\|crate::extent" \
  src/read.rs src/index.rs src/cache.rs src/io.rs src/replay.rs \
  src/source.rs src/async_source.rs src/dtype.rs src/hdf5/
```

That command must print nothing.

## What crosses the boundary

`classic.rs` sits in the upper crate. It uses `hdf5::message::Datatype`, which
sits in the lower one. That direction is correct: `Datatype` is the crate's
canonical element descriptor, not an HDF5 detail. `NcType::to_datatype` maps a
classic type onto it, and one decoder then serves both containers.

`Container` and `detect_container` live in `lib.rs` today. Both belong in the
lower crate. `classic.rs` and `netcdf.rs` both use them.

## The order of the work

1. Add a workspace `Cargo.toml`. Move the crate to `crates/oxcdf`.
2. Create `crates/oxcdf-hdf5`. Move the lower-layer files with `git mv`, so the
   history follows them.
3. Move `Container` and `detect_container` into the lower crate.
4. Replace `crate::` with `oxcdf_hdf5::` in the upper crate.
5. Re-export from `oxcdf::lib` until the public surface matches what it is now.
6. Split the features. `ndarray` belongs to the upper crate. `async`,
   `object-store` and the caches belong to both.
7. Move the tests. Most use `oxcdf::` only, so they should need no change.
   `src/hdf5/**` unit tests move with their files.

## Watch these

- `[package.metadata.docs.rs]` and `rustdoc-args` must move to the upper crate.
  Add them to the lower one as well, or its docs.rs page loses the async items.
- `#![cfg_attr(docsrs, feature(doc_cfg))]` is needed in both crate roots.
- `rust-toolchain.toml` stays at the workspace root.
- The `netcdf` git dependency is a dev-dependency of the upper crate only.
- `tests/` at the workspace root does not run. Tests live under each crate.
