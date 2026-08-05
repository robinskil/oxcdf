# oxcdf

Iron netCDF. A pure-Rust reader for netCDF-4 and netCDF classic files.

The reader parses the HDF5 container directly. It never calls netcdf-c. That C
library is not thread safe, so its Rust bindings put one process-global mutex
around every call. This crate has no mutex. Many threads read one file at the
same time.

```toml
[dependencies]
oxcdf = "0.1"
```

| Feature | Purpose |
|---|---|
| `async` | The asynchronous engine |
| `object-store` | Read from S3, GCS, Azure or HTTP |
| `ndarray` | Return `ArrayD` instead of `Vec` |
| `diff-tests` | Compare against netcdf-c in tests |

## Read a file

```rust
use oxcdf::netcdf::NetcdfFile;

let file = NetcdfFile::open("argo.nc")?;

// Metadata.
for d in file.dimensions() {
    println!("{} = {}", d.name, d.len);
}
for v in file.variables() {
    println!("{} {:?} {:?}", v.name, v.shape, v.dimensions);
}

// Values. A leading slash is optional.
let temp = file.variable("TEMP").unwrap();
let all = temp.read()?.to_f64()?;
let part = temp.read_slice(&[0..8, 10..30])?.to_f64()?;

// Attributes.
let units = temp.attribute("units").unwrap();
println!("{:?}", units.value.as_text());
```

`Values` also gives `to_i64`, `to_strings` and `to_sequences_f64`.

## Read one chunk at a time

Each chunk is a separate byte range with its own filters. Read chunks in
parallel. No coordination is necessary.

```rust
use rayon::prelude::*;

let blocks: Vec<_> = temp
    .chunks()
    .par_iter()
    .map(|c| temp.read_chunk(c))
    .collect();
```

Chunks are clipped to the variable. They cover it exactly once.

## Read asynchronously

Turn on the `async` feature.

The async engine does not walk chunk indexes. Call `prepare` first. An open is
synchronous, because it walks B-trees.

```rust
use std::sync::Arc;
use oxcdf::async_source::{AsyncByteSource, SyncAsAsync};
use oxcdf::index::Hdf5File;
use oxcdf::read::{read_hyperslab_async, Hyperslab};
use oxcdf::source::FileSource;

let file = Hdf5File::open("argo.nc")?;
let temp = file.dataset("/TEMP").unwrap();
temp.prepare(file.ctx())?;                       // Resolve the chunk index.

let source: Arc<dyn AsyncByteSource> =
    Arc::new(SyncAsAsync(FileSource::open("argo.nc")?));

let values = read_hyperslab_async(
    source.as_ref(),
    file.superblock(),
    file.cache(),
    temp,
    &Hyperslab::all(&temp.shape),
)
.await?
.to_f64(temp)?;
```

## Read from object storage

Turn on the `object-store` and `async` features. The reader reads byte ranges.
It needs no local copy.

```rust
use std::sync::Arc;
use object_store::{aws::AmazonS3Builder, path::Path, ObjectStore};
use oxcdf::async_source::AsyncObjectStoreSource;
use oxcdf::index::{Hdf5File, OpenOptions};
use oxcdf::object_store_source::ObjectStoreSource;
use oxcdf::read::{read_hyperslab_async, Hyperslab};

let store: Arc<dyn ObjectStore> =
    Arc::new(AmazonS3Builder::from_env().with_bucket_name("argo").build()?);
let path = Path::from("dac/aoml/13857_prof.nc");

// 1. Open. The open is synchronous, so run it on a blocking thread.
let blocking = ObjectStoreSource::new(store.clone(), path.clone()).await?;
let file = tokio::task::spawn_blocking(move || {
    Hdf5File::from_source_with(Arc::new(blocking), OpenOptions::remote())
})
.await??;

// 2. Prepare the variables you read.
let temp = file.dataset("/TEMP").unwrap();
temp.prepare(file.ctx())?;

// 3. Read. This source never blocks.
let source = AsyncObjectStoreSource::new(store, path).await?;
let values = read_hyperslab_async(
    &source,
    file.superblock(),
    file.cache(),
    temp,
    &Hyperslab::all(&temp.shape),
)
.await?
.to_f64(temp)?;
```

`OpenOptions::remote()` sets a 256 KiB request size and a 128 MiB byte cache.
Set them yourself with `io_request_size` and `io_cache_bytes`.

## Design

The two engines share every pure part. They differ only in how bytes arrive.

```text
plan     shared    Decide which byte ranges the read needs.
fetch    differs   ByteSource (sync) or AsyncByteSource (async).
decode   shared    Decompress, unshuffle and assemble the values.
```

The decode step stays synchronous. Decompression uses the processor. An async
decode would block the runtime.

Three caches remove repeated work. All three use `moka`. A hit takes no lock.

| Cache | Content |
|---|---|
| `IoCache` | Raw file bytes, in pages |
| `ChunkCache` | Decoded chunks, plus read ahead |
| Chunk index | One index for each variable, resolved on first use |

## Fall back to netcdf-c

`Error::Unsupported` marks a valid HDF5 feature that this reader does not
implement. Every other error marks a damaged file or a defect here.

Match on `Error::is_fallback_worthy()`. `DatasetIndex::is_readable()` answers the
same question before any read.

## Performance

Measured on an Argo file with 68 datasets. See `tests/bench.rs`.

| Threads | netcdf-c | oxcdf |
|---|---|---|
| 1 | 11.9 ms | 8.8 ms |
| 8 | 38.6 ms | 3.7 ms |

netcdf-c gets slower with each added thread. oxcdf gets faster. The mutex causes
the difference. An open costs 277 microseconds against 1.40 milliseconds.

The chunk cache helps these numbers. All measurements use a local file.

## State

Supported: netCDF-4 and classic CDF-1 and CDF-2; all five version 4 chunk
indexes; shuffle, deflate, fletcher32, zstd and blosc; fixed strings, variable
strings and variable sequences; big-endian values; fill values.

Missing: szip; blosc with snappy; extensible arrays that need secondary blocks;
CDF-5 (written, unverified); async open; batched reads across variables.

The crate does not write files. Keep netcdf-c for writes.

## Tests

```bash
cargo test --features "diff-tests,object-store,ndarray,async"
```

294 tests. `differential.rs` compares every value against netcdf-c.
`netcdf_layer.rs` compares variables, dimensions and axes against `ncdump`.
Floats compare by bit pattern, not by tolerance.
