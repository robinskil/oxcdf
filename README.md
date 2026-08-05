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
let file = oxcdf::open("argo.nc")?;

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
println!("{:?}", temp.attribute("units").unwrap().value.as_text());
```

`Values` also gives `to_i64`, `to_strings` and `to_sequences_f64`.

## Read a file asynchronously

Turn on the `async` feature. The API matches the synchronous one. An open
awaits. A read of values awaits. Everything else answers at once.

```rust
let file = oxcdf::open_async(source).await?;

for d in file.dimensions() {
    println!("{} = {}", d.name, d.len);
}

let temp = file.variable("TEMP").unwrap();
println!("{:?}", temp.attribute("units").unwrap().value.as_text());

let all = temp.read().await?.to_f64()?;
let part = temp.read_slice(&[0..8, 10..30]).await?.to_f64()?;
```

`source` is any `Arc<dyn AsyncByteSource>`. Use `SyncAsAsync(FileSource::open(
path)?)` for a local file.

## Read from object storage

Turn on the `object-store` and `async` features. The reader reads byte ranges.
It needs no local copy.

```rust
use object_store::{aws::AmazonS3Builder, path::Path};

let store = Arc::new(AmazonS3Builder::from_env().with_bucket_name("argo").build()?);
let file = oxcdf::AsyncFile::open_store(store, Path::from("13857_prof.nc")).await?;

let temp = file.variable("TEMP").unwrap();
let values = temp.read().await?.to_f64()?;
```

`open_store` uses `OpenOptions::remote()`: a 256 KiB request size and a 128 MiB
byte cache. Pass your own with `open_store_with`.

Every file in the test corpus opens in **one request**.

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

## Design

The two engines share every pure part. They differ only in how bytes arrive.

```text
plan     shared    Decide which byte ranges the read needs.
fetch    differs   ByteSource (sync) or AsyncByteSource (async).
decode   shared    Decompress, unshuffle and assemble the values.
```

The decode step stays synchronous. Decompression uses the processor. An async
decode would block the runtime.

### The asynchronous open

An open walks the file metadata. The walk is a chain of dependent reads. A
superblock names an object header. That header names a heap. That heap names
more headers.

The crate does not hold a second copy of that parser. It runs the same one over
pages held in memory.

1. Fetch a window of pages.
2. Run the synchronous walk over those pages.
3. A read outside the pages records what it wants. The walk stops.
4. Fetch every recorded page in one batch. Go to step 2.

One round is one batch of requests. The count of rounds equals the depth of the
dependent reads. A second parser needs the same count. A round repeats the parse
work, which is pure processor work over memory, so the repeat costs microseconds.

netCDF writes its metadata near the front of the file, so the first window
normally covers the whole walk. A chunk index and a string heap resolve the same
way, on first use.

### Caches

Three caches remove repeated work. All three use `moka`. A hit takes no lock.

| Cache | Content |
|---|---|
| `IoCache` | Raw file bytes, in pages |
| `ChunkCache` | Decoded chunks, plus read ahead |
| Chunk index | One index for each variable, resolved on first use |

## Fall back to netcdf-c

`Error::Unsupported` marks a valid HDF5 feature that this reader does not
implement. Every other error marks a damaged file or a defect here.

Match on `Error::is_fallback_worthy()`. `Variable::is_readable()` answers the
same question before any read.

## Performance

Measured on an Argo file with 68 datasets. See `tests/bench.rs`.

| Threads | netcdf-c | oxcdf |
|---|---|---|
| 1 | 11.9 ms | 8.8 ms |
| 8 | 38.6 ms | 3.7 ms |

netcdf-c gets slower with each added thread. oxcdf gets faster. The mutex causes
the difference. An open costs 277 microseconds against 1.40 milliseconds.

The chunk cache helps these numbers. All measurements use a local file. No test
bucket is available, so the gain on object storage is not measured.

## State

Supported: netCDF-4 and classic CDF-1 and CDF-2; all five version 4 chunk
indexes; shuffle, deflate, fletcher32, zstd and blosc; fixed strings, variable
strings and variable sequences; big-endian values; fill values; asynchronous
open and read.

Missing: szip; blosc with snappy; extensible arrays that need secondary blocks;
CDF-5 (written, unverified); batched reads across variables.

The crate does not write files. Keep netcdf-c for writes.

## Tests

```bash
cargo test --features "diff-tests,object-store,ndarray,async"
```

317 tests. `differential.rs` compares every value against netcdf-c.
`netcdf_layer.rs` compares variables, dimensions and axes against `ncdump`.
`async_open.rs` compares the two engines, file by file and value by value.
Floats compare by bit pattern, not by tolerance.
