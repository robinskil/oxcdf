# oxcdf

A pure-Rust reader for netCDF-4 and netCDF classic files.

The reader parses the HDF5 container. It never calls netcdf-c. That C library is
not thread safe. Its Rust bindings put one process-global mutex around every
call. This crate has no mutex. Many threads read one file at the same time.

`oxcdf::open` reads the magic bytes and picks the container. netCDF-4 and
classic then use the same interface.

The API matches the `netcdf` crate. A program moves across with few changes.

```toml
[dependencies]
oxcdf = { version = "0.1", features = ["async", "object-store"] }
```

| Feature | Purpose |
|---|---|
| `async` | The asynchronous engine |
| `object-store` | Read from S3, GCS, Azure or HTTP |
| `ndarray` | Return `ArrayD` |
| `diff-tests` | Compare against netcdf-c in tests |

## Read

```rust
let file = oxcdf::open("argo.nc")?;

for d in file.dimensions() { println!("{} = {}", d.name, d.len); }
for a in file.attributes() { println!("{} = {:?}", a.name, a.value); }

let temp = file.variable("TEMP").unwrap();
println!("{:?} {:?} {:?}", temp.vartype(), temp.shape, temp.dimensions);

let all = temp.get_values::<f32, _>(..)?;
let part = temp.get_values::<f32, _>([0..8, 10..30])?;
let one = temp.get_value::<f32, _>([0, 3])?;
let units = temp.attribute("units").unwrap().value.as_text();
```

With the `ndarray` feature, `get` returns an `ArrayD` of the selection's shape.

## Read asynchronously

An open awaits. A read of values awaits. Every other call answers at once.

```rust
let file = oxcdf::AsyncFile::open_store(store, path).await?;

for d in file.dimensions() { println!("{} = {}", d.name, d.len); }

let temp = file.variable("TEMP").unwrap();
let all = temp.get_values::<f32, _>(..).await?;
let part = temp.get_values::<f32, _>([0..8, 10..30]).await?;
```

`open_store` reads byte ranges. It needs no local copy. It uses a 256 KiB
request size and a 128 MiB byte cache. Use `oxcdf::open_async(source)` for any
other `AsyncByteSource`.

## Select

`get_values` takes the same forms as the `netcdf` crate.

```text
..                                        the whole variable
[0..8, 10..30]                            one range for each axis
[0, 3]                                    one element
[2.., 5..]                                to the end of each axis
[..8, ..30]                               from the start of each axis
[Extent::Index(3), (0..6).into()]         one index and one range
([0, 10].as_slice(), [8, 20].as_slice())  a start and a count
```

A stride returns `Error::Unsupported`. This reader reads contiguous selections
only.

## Types

`vartype()` returns a `DType`. It mirrors `netcdf::NcVariableType`.

| `DType` | netCDF | Read it with |
|---|---|---|
| `Int(1)`…`Int(8)` `Uint(1)`…`Uint(8)` | `byte`…`uint64` | `get_values::<i8>`…`::<u64>` |
| `Float(4)` `Float(8)` | `float` `double` | `get_values::<f32>` `::<f64>` |
| `Char` | `char` | `get_strings` or `get_raw_values` |
| `String` | `string` | `get_strings` |
| `Vlen(..)` | ragged array | `read()?.to_sequences::<T>()` |

A classic file has no ragged array and no variable-length string. Every other
row applies to both containers.

A read converts between any two numeric types, as the `netcdf` crate does. Ask
for the type `vartype()` reports. The read then copies the values.

A conversion loses information. `f64` to `f32` loses precision. `i64` to `f64`
loses integers above 2^53. A float to an integer truncates toward zero.

A read of text as a number returns `Error::TypeMismatch`. The message names the
stored type.

## Strings

A `string` variable holds one value in each element. `get_strings` returns one
string for each element.

A `char` variable holds one byte in each element. Its last dimension is the
string length. `get_strings` returns one string for each character. Join that
axis yourself.

```rust
let v = file.variable("country").unwrap();   // char country(casts, strnlen)
let width = *v.shape.last().unwrap() as usize;
let names: Vec<String> = v.get_strings(..)?
    .chunks(width)
    .map(|row| row.concat().trim_end_matches('\0').to_string())
    .collect();
```

## Attributes

An attribute keeps its stored type. `AttributeValue` mirrors
`netcdf::AttributeValue`. One value gets the singular variant. Several values
get the plural one.

```rust
match &temp.attribute("_FillValue").unwrap().value {
    AttributeValue::Float(v) => println!("f32 {v}"),
    AttributeValue::Double(v) => println!("f64 {v}"),
    other => println!("{other:?}"),
}
let scale = temp.attribute("scale_factor").and_then(|a| a.value.as_f64());
```

## Chunks

A chunk is a separate byte range with its own filters. Read chunks in parallel.
No coordination is necessary.

```rust
let blocks: Vec<_> = temp.chunks().par_iter().map(|c| temp.read_chunk(c)).collect();
```

A chunk is clipped to the variable. The chunks cover it exactly once.

## Containers

`oxcdf::open` and `oxcdf::open_async` both accept either container. The methods
above behave the same way for each.

| | netCDF-4 | classic |
|---|---|---|
| Container | HDF5 | CDF-1 and CDF-2 |
| Groups | nested | one root group |
| Storage | contiguous, chunked or compact | contiguous |
| `chunks()` | the stored chunk grid | one chunk for the whole variable |
| `hdf5()` | the HDF5 index | `None` |
| `classic()` | `None` | the classic file |

`container()` reports which one. `vartype()`, `get_values`, `get_strings`,
`Extents` and `AttributeValue` do not change.

## Design

Both engines share every pure part. Only the fetch differs.

```text
plan     shared    Decide which byte ranges the read needs.
fetch    differs   ByteSource (sync) or AsyncByteSource (async).
decode   shared    Decompress, unshuffle and assemble the values.
```

Decode stays synchronous. Decompression uses the processor. An async decode
blocks the runtime.

### The asynchronous open

An open walks a chain of dependent reads. The crate holds one parser, not two.

1. Fetch a window of pages.
2. Run the synchronous walk over those pages.
3. A read outside the pages records what it wants. The walk stops.
4. Fetch every recorded page in one batch. Go to step 2.

One round is one batch of requests. netCDF writes its metadata near the front of
the file. Every corpus file opens in one request.

### Caches

Three caches remove repeated work. All three use `moka`. A hit takes no lock.

| Cache | Content |
|---|---|
| `IoCache` | Raw file bytes, in pages |
| `ChunkCache` | Decoded chunks, plus read ahead |
| Chunk index | One index for each variable. It resolves on first use. |

## State

Supported:

- netCDF-4, and classic CDF-1 and CDF-2, through one interface
- All five version 4 chunk indexes
- Filters: shuffle, deflate, fletcher32, zstd and blosc
- Every netCDF type, big-endian values and fill values
- Asynchronous open and read

Not supported:

- szip, and blosc with snappy
- Extensible arrays that need secondary blocks
- CDF-5. The code exists. No test file exists.
- Strided selections
- Batched reads across variables

The crate does not write files. Keep netcdf-c for writes.

`Error::Unsupported` marks a valid HDF5 feature this reader does not implement.
Every other error marks a damaged file or a defect here. Match on
`Error::is_fallback_worthy()` to send one variable to netcdf-c.

## Tests

```bash
cargo test --features "diff-tests,object-store,ndarray,async"
```

394 tests.

| File | Compares against |
|---|---|
| `differential.rs` | netcdf-c, every value |
| `typed_reads.rs` | the `netcdf` crate, every variable as its stored type |
| `attributes.rs` | the `netcdf` crate, 421 attribute values |
| `netcdf_layer.rs` | `ncdump`, variables and dimensions and axes |
| `async_open.rs` | the synchronous engine, file by file |
| `classic_interface.rs` | the `netcdf` crate, and the two engines, on classic files |
| `readme_api.rs` | this page, every example |

Floats compare by bit pattern, not by tolerance.
