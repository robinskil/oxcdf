# oxcdf

Iron netCDF. A pure-Rust reader for netCDF-4 and netCDF classic files.

The reader parses the HDF5 container directly. It never calls netcdf-c. That C
library is not thread safe, so its Rust bindings put one process-global mutex
around every call. This crate has no mutex. Many threads read one file at the
same time.

```toml
[dependencies]
oxcdf = { version = "0.1", features = ["async", "object-store"] }
```

| Feature | Purpose |
|---|---|
| `async` | The asynchronous engine |
| `object-store` | Read from S3, GCS, Azure or HTTP |
| `ndarray` | Return `ArrayD` instead of `Vec` |
| `diff-tests` | Compare against netcdf-c in tests |

## The interface

The two engines expose the same interface. An open awaits. A read of values
awaits. Every other call answers at once, because the open reads all the
metadata.

| | Synchronous | Asynchronous |
|---|---|---|
| Open a local file | `oxcdf::open(path)?` | `oxcdf::open_async(source).await?` |
| Open from a store | | `AsyncFile::open_store(store, path).await?` |
| Dimensions | `file.dimensions()` | `file.dimensions()` |
| Global attributes | `file.attributes()` | `file.attributes()` |
| Every variable | `file.variables()` | `file.variables()` |
| One variable | `file.variable("TEMP")` | `file.variable("TEMP")` |
| A subgroup | `file.group("/forecast")` | `file.group("/forecast")` |
| Variable attributes | `var.attribute("units")` | `var.attribute("units")` |
| Read all | `var.read()?` | `var.read().await?` |
| Read a slice | `var.read_slice(&[0..8])?` | `var.read_slice(&[0..8]).await?` |
| Read an array | `var.read_array_f64()?` | `var.read_array_f64().await?` |
| List the chunks | `var.chunks()` | `var.chunks().await?` |
| Read one chunk | `var.read_chunk(&c)?` | `var.read_chunk(&c).await?` |

## Read a local file

```rust
let file = oxcdf::open("argo.nc")?;

// Dimensions.
for d in file.dimensions() {
    let mark = if d.is_unlimited { " (unlimited)" } else { "" };
    println!("{} = {}{}", d.name, d.len, mark);
}

// Global attributes.
for a in file.attributes() {
    println!("{} = {:?}", a.name, a.value);
}

// Variables.
for v in file.variables() {
    println!("{} {:?} {:?} {:?}", v.name, v.dtype(), v.shape, v.dimensions);
}

// One variable. A leading slash is optional.
let temp = file.variable("TEMP").unwrap();

// Its attributes.
println!("{:?}", temp.attribute("units").unwrap().value.as_text());
println!("{:?}", temp.attribute("_FillValue").unwrap().value.as_f64());

// Its values.
let all = temp.read()?.to_f64()?;
let part = temp.read_slice(&[0..8, 10..30])?.to_f64()?;

// A variable in a subgroup.
let nested = file.variable("/forecast/TEMP").unwrap();
```

`Values` also gives `to_i64`, `to_strings`, `to_sequences_f64` and `as_bytes`.

## Read into an ndarray

Turn on the `ndarray` feature. The result is an `ArrayD`. Its shape is the
variable's shape. Its axes follow the variable's dimensions, in order.

```rust
let temp = file.variable("TEMP").unwrap();

// The whole variable.
let a = temp.read_array_f64()?;      // ArrayD<f64>, shape [8, 6]
assert_eq!(a.shape(), &[8, 6]);
println!("{}", a[[0, 0]]);           // Row-major.

// One row.
let row = a.index_axis(ndarray::Axis(0), 0);

// A slice. Read first, then convert.
let b = temp.read_slice(&[5..15, 2..5])?.to_array_f64()?;
assert_eq!(b.shape(), &[10, 3]);

// An integer variable.
let counts = file.variable("N_LEVELS").unwrap().read_array_i64()?;

// Strings.
let names = file.variable("PLATFORM_NUMBER").unwrap().read()?.to_array_strings()?;
```

The asynchronous form is the same, with an await.

```rust
let a = temp.read_array_f64().await?;
let b = temp.read_slice(&[5..15, 2..5]).await?.to_array_f64()?;
```

`read_array_f64` widens any integer or float variable to `f64`. `read_array_i64`
takes integers only: it refuses a float rather than round it.
`to_array_strings` needs a string variable, fixed length or variable length.

## Read from object storage

Turn on the `async` and `object-store` features. The reader reads byte ranges.
It needs no local copy. The interface does not change.

```rust
use std::sync::Arc;
use object_store::{aws::AmazonS3Builder, path::Path, ObjectStore};
use oxcdf::AsyncFile;

let store: Arc<dyn ObjectStore> =
    Arc::new(AmazonS3Builder::from_env().with_bucket_name("argo").build()?);

let file = AsyncFile::open_store(store, Path::from("dac/aoml/13857_prof.nc")).await?;

// Dimensions.
for d in file.dimensions() {
    let mark = if d.is_unlimited { " (unlimited)" } else { "" };
    println!("{} = {}{}", d.name, d.len, mark);
}

// Global attributes.
for a in file.attributes() {
    println!("{} = {:?}", a.name, a.value);
}

// Variables.
for v in file.variables() {
    println!("{} {:?} {:?} {:?}", v.name, v.dtype(), v.shape, v.dimensions);
}

// One variable.
let temp = file.variable("TEMP").unwrap();

// Its attributes. The open read them, so this needs no request.
println!("{:?}", temp.attribute("units").unwrap().value.as_text());

// Its values.
let all = temp.read().await?.to_f64()?;
let part = temp.read_slice(&[0..8, 10..30]).await?.to_f64()?;
```

`open_store` uses `OpenOptions::remote()`: a 256 KiB request size and a 128 MiB
byte cache. Pass your own with `open_store_with`.

Every file in the test corpus opens in **one request**.

## Read a local file asynchronously

Wrap a local file to get the same interface without a store.

```rust
use std::sync::Arc;
use oxcdf::{FileSource, SyncAsAsync};

let source = Arc::new(SyncAsAsync(FileSource::open("argo.nc")?));
let file = oxcdf::open_async(source).await?;

let temp = file.variable("TEMP").unwrap();
let values = temp.read().await?.to_f64()?;
```

`open_async` takes any `Arc<dyn AsyncByteSource>`. Implement that trait for any
other backend.

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

The asynchronous form reads them together.

```rust
let chunks = temp.chunks().await?;
let blocks = futures::future::try_join_all(
    chunks.iter().map(|c| temp.read_chunk(c))
).await?;
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

325 tests. `differential.rs` compares every value against netcdf-c.
`netcdf_layer.rs` compares variables, dimensions and axes against `ncdump`.
`async_open.rs` compares the two engines, file by file and value by value.
`readme_api.rs` runs every example on this page, so the page cannot drift from
the API. Floats compare by bit pattern, not by tolerance.
