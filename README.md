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
| Read values | `var.get_values::<f32,_>(..)?` | `var.get_values::<f32,_>(..).await?` |
| Read one value | `var.get_value::<f32,_>([0,3])?` | `var.get_value::<f32,_>([0,3]).await?` |
| Read strings | `var.get_strings(..)?` | `var.get_strings(..).await?` |
| Read one string | `var.get_string([0])?` | `var.get_string([0]).await?` |
| Read an array | `var.get::<f32,_>(..)?` | `var.get::<f32,_>(..).await?` |
| List the chunks | `var.chunks()` | `var.chunks().await?` |
| Read one chunk | `var.read_chunk(&c)?` | `var.read_chunk(&c).await?` |

`get_values`, `get_value` and the selection forms match the `netcdf` crate. A
program that uses that crate reads the same values here.

## Read a local file

```rust
use oxcdf::{Extent, Extents};

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
    println!("{} {:?} {:?} {:?}", v.name, v.vartype(), v.shape, v.dimensions);
}

// One variable. A leading slash is optional.
let temp = file.variable("TEMP").unwrap();

// Its attributes.
println!("{:?}", temp.attribute("units").unwrap().value.as_text());
println!("{:?}", temp.attribute("_FillValue").unwrap().value.as_f64());

// Its values. Ask for the stored type to copy them without a conversion.
let all = temp.get_values::<f32, _>(Extents::All)?;
let part = temp.get_values::<f32, _>([0..8, 10..30])?;
let one = temp.get_value::<f32, _>([0, 3])?;

// A variable in a subgroup.
let nested = file.variable("/forecast/TEMP").unwrap();
```

## Select the elements

`get_values` takes the same selection forms as the `netcdf` crate.

```rust
var.get_values::<f32, _>(Extents::All)?;     // the whole variable
var.get_values::<f32, _>(..)?;               // the same
var.get_values::<f32, _>([0..8, 10..30])?;   // one range for each axis
var.get_values::<f32, _>([0, 3])?;           // one element
var.get_values::<f32, _>([2.., 5..])?;       // to the end of each axis
var.get_values::<f32, _>([..8, ..30])?;      // from the start of each axis

// A Rust array holds one type, so mix an index and a range through `Extent`.
var.get_values::<f32, _>([Extent::Index(3), (0..6).into()])?;

// A start and a count for each axis.
var.get_values::<f32, _>(([0usize, 10].as_slice(), [8usize, 20].as_slice()))?;
```

A stride is the one form this reader does not read yet. It returns
`Error::Unsupported` rather than read the wrong elements.

## Types

A read converts between any two numeric types, which is what the `netcdf` crate
does. A read of a string as a number returns `Error::TypeMismatch`, which names
the stored type.

```rust
let temp = file.variable("TEMP").unwrap();
println!("{:?}", temp.vartype());               // Float(4)

let exact = temp.get_values::<f32, _>(..)?;   // copied, no conversion
let wide = temp.get_values::<f64, _>(..)?;    // converted
```

Ask for the type `vartype()` reports and the read copies the values. Any other
numeric type converts, and a conversion can lose information: `f64` to `f32`
loses precision, `i64` to `f64` loses integers above 2^53, and a float to an
integer truncates toward zero.

`vartype()` returns a `DType`, which mirrors the `netcdf` crate's
`NcVariableType`.

| `vartype()` | netCDF | Read it with |
|---|---|---|
| `Int(1)` `Int(2)` `Int(4)` `Int(8)` | `byte` `short` `int` `int64` | `get_values::<i8>` … `::<i64>` |
| `Uint(1)` `Uint(2)` `Uint(4)` `Uint(8)` | `ubyte` `ushort` `uint` `uint64` | `get_values::<u8>` … `::<u64>` |
| `Float(4)` `Float(8)` | `float` `double` | `get_values::<f32>` `::<f64>` |
| `Char` | `char` | `get_strings` or `as_bytes` |
| `String` | `string` | `get_strings` |
| `Vlen(..)` | ragged array | `to_sequences::<T>` |
| `FixedString(n)` | not netCDF | `get_strings` |

## Strings

netCDF stores text two ways, and this reader keeps them apart.

A `string` variable holds one variable-length string in each element. The value
lives in the global heap, which the reader follows for you.

```rust
let names = file.variable("station_name").unwrap();
assert_eq!(names.vartype(), DType::String);

let all = names.get_strings(..)?;      // one string for each element
let one = names.get_string([0])?;
```

A `char` variable holds one **byte** in each element. Its last dimension is the
string length, so `char country(casts, strnlensmall)` holds one string for each
cast. The reader reports the elements as the file stores them. Join the last
axis yourself.

```rust
let v = file.variable("country").unwrap();
assert_eq!(v.vartype(), DType::Char);
assert_eq!(v.shape, vec![47, 40]);     // 40 is the string length

let width = *v.shape.last().unwrap() as usize;
let joined: Vec<String> = v
    .get_strings(..)?
    .chunks(width)
    .map(|row| row.concat().trim_end_matches('\0').to_string())
    .collect();
assert_eq!(joined[0], "GREAT BRITAIN");
```

`get_raw_values(..)` gives the same data as raw bytes, which is often the
simpler route for a caller that builds its own string array.

## Beyond the netCDF interface

Everything above matches the `netcdf` crate. Two reads have no netCDF name, so
they go through `read()`, which returns a `Values`.

```rust
// A ragged array.
let seqs = file.variable("profiles").unwrap().read()?.to_sequences::<f32>()?;

// The type and the shape, before deciding how to decode.
let values = temp.read()?;
println!("{:?} {:?}", values.dtype(), values.shape());
let numbers: Vec<f32> = values.get()?;
```

Chunks are the other addition. Each chunk is a separate byte range with its own
filters, so they are the natural unit of parallel work.

## Read into an ndarray

Turn on the `ndarray` feature. `get` takes the same selections and the
same types as `get_values`. The result is an `ArrayD` whose shape is the
selection's shape. Its axes follow the variable's dimensions, in order.

```rust
let temp = file.variable("TEMP").unwrap();     // stored f32

// The whole variable.
let a = temp.get::<f32, _>(..)?;         // ArrayD<f32>, shape [8, 6]
assert_eq!(a.shape(), &[8, 6]);
println!("{}", a[[0, 0]]);                     // Row-major.

// One row.
let row = a.index_axis(ndarray::Axis(0), 0);

// A block.
let b = temp.get::<f32, _>([5..15, 2..5])?;
assert_eq!(b.shape(), &[10, 3]);

// An integer variable.
let counts = file.variable("N_LEVELS").unwrap().get::<i32, _>(..)?;

// Strings.
let names = file.variable("PLATFORM_NUMBER").unwrap().read()?.to_array_strings()?;
```

The asynchronous form is the same, with an await.

```rust
let a = temp.get::<f32, _>(..).await?;
let b = temp.get::<f32, _>([5..15, 2..5]).await?;
```

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
    println!("{} {:?} {:?} {:?}", v.name, v.vartype(), v.shape, v.dimensions);
}

// One variable.
let temp = file.variable("TEMP").unwrap();

// Its attributes. The open read them, so this needs no request.
println!("{:?}", temp.attribute("units").unwrap().value.as_text());

// Its values.
let all = temp.get_values::<f32, _>(Extents::All).await?;
let part = temp.get_values::<f32, _>([0..8, 10..30]).await?;
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
let values = temp.get_values::<f32, _>(..).await?;
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
CDF-5 (written, unverified); strided selections; batched reads across variables.

The crate does not write files. Keep netcdf-c for writes.

## Tests

```bash
cargo test --features "diff-tests,object-store,ndarray,async"
```

372 tests. `differential.rs` compares every value against netcdf-c.
`typed_reads.rs` reads every corpus variable as its stored type through both
this crate and the `netcdf` crate, and compares element by element.
`netcdf_layer.rs` compares variables, dimensions and axes against `ncdump`.
`async_open.rs` compares the two engines, file by file and value by value.
`readme_api.rs` runs every example on this page, so the page cannot drift from
the API. Floats compare by bit pattern, not by tolerance.
