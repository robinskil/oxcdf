# oxcdf-hdf5

A pure-Rust, parallel-safe reader for the HDF5 container format.

This crate reads the file as HDF5. It knows nothing about netCDF. It never calls
the C library, so it has no global mutex. Many threads read one file at the same
time.

For netCDF-4 and netCDF classic files, use [`oxcdf`], which applies the netCDF
conventions on top of this crate and re-exports it.

[`oxcdf`]: https://crates.io/crates/oxcdf

```toml
[dependencies]
oxcdf-hdf5 = { version = "0.3", features = ["async", "object-store"] }
```

| Feature | Purpose |
|---|---|
| `async` | The asynchronous engine |
| `object-store` | Read from S3, GCS, Azure or HTTP |

## Read

```rust
use oxcdf_hdf5::index::Hdf5File;
use oxcdf_hdf5::read::{read_hyperslab, Hyperslab};

let file = Hdf5File::open("data.h5")?;
let dataset = file.dataset("/temperature").unwrap();
dataset.prepare(file.ctx())?;

let slab = Hyperslab::all(&dataset.shape);
let raw = read_hyperslab(file.ctx(), dataset, &slab)?;
let values = raw.get::<f64>(dataset)?;
```

## Read asynchronously

`AsyncHdf5File` mirrors `Hdf5File`. An open awaits. A read of values awaits.
Every other call answers at once.

```rust
let file = oxcdf_hdf5::AsyncHdf5File::open_store(store, path).await?;

let temp = file.dataset("/temperature").unwrap();
println!("{:?}", temp.shape);                    // no await: metadata is in memory

let all = temp.read().await?.get::<f64>(&temp)?;
let part = temp.read_selection(&slab).await?;
```

`open_store` reads byte ranges. It needs no local copy. Use
`AsyncHdf5File::open(source)` for any other `AsyncByteSource`.

## Design

An open parses the metadata once. The result is an immutable index. The index is
`Send + Sync`. A read is a pure function of the index and a request. A read holds
no lock.

All input goes through `ByteSource`. Its methods take `&self`. Its methods
address bytes by absolute offset. There is no file position to share.

Both engines share every pure part. Only the fetch differs. Decode stays
synchronous, because decompression uses the processor.

## Scope

The reader targets the subset of HDF5 that netcdf-c writes. It does not target
the whole specification. A feature outside that subset returns
`Error::Unsupported`. Every other error marks a damaged file or a defect here.

The crate does not write files.

See the [repository] for the full state and the test matrix.

[repository]: https://github.com/robinskil/oxcdf
