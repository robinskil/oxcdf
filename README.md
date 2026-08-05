# oxcdf

Iron netCDF. A pure-Rust reader for netCDF-4 files. The reader parses the HDF5 container
directly. It does not call the netcdf-c library.

Many threads can read one file at the same time.

## Purpose

netCDF-4 files are HDF5 files. The netcdf-c library is not thread safe. The Rust
bindings put one process-global mutex around every library call.

That mutex covers the expensive work. Chunk input, decompression and type
conversion all run while the mutex is held. A query engine that reads many files
gets no parallel reads.

This crate removes the mutex. It removes the C library.

## Design

The reader separates one parse from many reads.

1. An open parses the file metadata once. The result is an immutable index.
2. The index is `Send + Sync`. Share it with `Arc`.
3. A read is a pure function of the index and a request.
4. A read holds no lock. A read changes no shared state.

All input goes through `ByteSource`. Its methods take `&self`. Its methods
address bytes by absolute offset. There is no file position to share.

The crate does not write files. A read is a parser. A write needs B-tree
insertion and free space management. Keep netcdf-c for writes.

## Two engines

The crate has a synchronous engine and an asynchronous engine. Both engines
share every pure part: the parsers, the filters, the chunk arithmetic and the
netCDF layer.

The engines differ only in how bytes arrive.

```text
plan     shared    Decide which byte ranges the read needs.
fetch    differs   ByteSource (sync) or AsyncByteSource (async).
decode   shared    Decompress, unshuffle and assemble the values.
```

The decode step stays synchronous. Decompression uses the processor. An async
decode would block the runtime.

## Usage

### Open a file

```rust
use oxcdf::netcdf::NetcdfFile;

let file = NetcdfFile::open("argo.nc")?;
```

To control the input size and the cache size, use `OpenOptions`.

```rust
use oxcdf::index::OpenOptions;

let file = NetcdfFile::open_with(
    "argo.nc",
    OpenOptions::new()
        .io_request_size(256 * 1024)   // Read 256 KiB for each cache miss.
        .io_cache_bytes(128 << 20),    // Hold 128 MiB of file bytes.
)?;
```

For object storage, use the preset. It sets the request size, the cache size and
the merge policy.

```rust
let file = NetcdfFile::open_with("argo.nc", OpenOptions::remote())?;
```

### Read the metadata

```rust
for d in file.dimensions() {
    println!("{} = {}", d.name, d.len);
}

for a in file.attributes() {          // Global attributes.
    println!("{} = {:?}", a.name, a.value);
}

for v in file.variables() {
    println!("{} {:?} {:?}", v.name, v.shape, v.dimensions);
    for a in &v.attributes {          // Variable attributes.
        println!("  {} = {:?}", a.name, a.value);
    }
}
```

An attribute value has one of four types: `Text`, `Floats`, `Ints` or `Raw`. The
reader converts big-endian values to native order.

### Read the values

Give the variable name. A leading slash is optional.

```rust
let temp = file.variable("TEMP").unwrap();

let all = temp.read()?.to_f64()?;                  // The whole variable.
let part = temp.read_slice(&[0..8, 10..30])?;      // One range for each axis.
```

`Values` holds its own type. Call `to_f64`, `to_i64` or `to_strings` on it.

The reader does not replace fill values. A read returns the stored value. The
`_FillValue` attribute states what that value means.

### Read one chunk at a time

A chunked variable holds independent compressed blocks. Each block is a separate
byte range with its own filters. Read blocks in parallel. No coordination is
necessary.

```rust
for chunk in temp.chunks() {
    let block = temp.read_chunk(&chunk)?;
}
```

A chunk is clipped to the variable. An edge chunk never returns padding. The
chunks cover the variable exactly once. A contiguous variable reports one chunk.

### Read as an ndarray

Turn on the `ndarray` feature.

```rust
let a = temp.read_array_f64()?;       // ArrayD<f64>, shape [8, 6].
a[[0, 0]];                            // Row-major. Axes follow the dimensions.
```

### Read from object storage

Turn on the `object-store` feature. The reader reads byte ranges. It needs no
local copy.

```rust
let source = AsyncObjectStoreSource::new(store, path).await?;
let file = Hdf5File::from_source(Arc::new(SyncAsAsync(source)))?;
```

### Read asynchronously

Turn on the `async` feature.

```rust
let values = read_hyperslab_async(source, file.superblock(), file.cache(), d, &slab).await?;
```

The async engine does not walk chunk indexes. Call `prepare` first.

```rust
d.prepare(file.ctx())?;
```

## Caches

The crate holds three caches. All three use `moka`. A cache hit takes no lock.
Readers do not block each other.

| Cache | Content | Purpose |
|---|---|---|
| `IoCache` | Raw file bytes, in pages | Remove repeated input requests |
| `ChunkCache` | Decoded chunks | Remove repeated decompression |
| Chunk index | One index for each variable | Remove repeated B-tree walks |

### The byte cache

`IoCache` holds raw file bytes in pages of a fixed size. The page size is the
input request size. Every miss reads one whole page.

The reader uses pages, not exact ranges. A page lookup is a hash lookup. An
exact range lookup needs an interval search. Pages also merge neighbours: two
reads in one page cost one request.

A larger page gives fewer requests and more waste. Use 256 KiB for object
storage. Use the 64 KiB default for local files.

### The chunk cache

`ChunkCache` holds decoded chunks. It also reads ahead. A read fetches the
chunks it needs, plus a few more. Set the count with `readahead`. Zero stops the
read ahead.

### The chunk index

A chunk index resolves on first use. An open does not walk it. A query that
reads 2 of 55 variables walks 2 indexes.

Call `prepare` to resolve an index early. A DataFusion reader knows its
projection. It prepares only the variables it reads.

## Performance

Measurements use the Argo file in this repository. The file holds 68 datasets.
55 datasets use chunks.

| Threads | netcdf-c | this crate | ratio |
|---|---|---|---|
| 1 | 11.9 ms | 8.8 ms | 1.35x |
| 2 | 17.2 ms | 6.9 ms | 2.48x |
| 4 | 27.6 ms | 5.1 ms | 5.42x |
| 8 | 38.6 ms | 3.7 ms | 10.5x |

netcdf-c gets slower with each added thread. This crate gets faster. The mutex
causes the difference.

An open costs 277 microseconds. netcdf-c costs 1.40 milliseconds.

Two limits apply to these numbers.

1. The chunk cache helps these numbers. The benchmark reads one file 200 times.
   A cold scan gains less.
2. All measurements use a local file. The gain on object storage is not
   measured. No test bucket is available.

Run the benchmarks:

```bash
cargo test --release -p oxcdf --features diff-tests --test bench -- --ignored --nocapture
```

## Fall back to netcdf-c

`Error::Unsupported` marks a valid HDF5 feature that this reader does not
implement. Every other error marks a damaged file or a defect in this crate.

Match on `Error::is_fallback_worthy()`. A fall back then never hides a defect.

`DatasetIndex::is_readable()` answers the same question before any read. It
checks the type and the filters. It reads nothing.

The reader never passes an undecodable filter through. A filter flag marks what
a writer may skip. It does not mark what a chunk skipped. The chunk filter mask
states that. A pass through would return compressed bytes as values.

## Supported features

| Area | State |
|---|---|
| Superblock version 0 to 3 | Done |
| Object header version 1 and 2 | Done |
| Messages: dataspace, datatype, layout, filters, attribute, link, symbol table | Done |
| Old groups: version 1 B-tree, symbol table, local heap | Done |
| New groups: compact links, dense links in a fractal heap | Done |
| Version 2 B-tree, for heaps with free space | Done |
| Chunk index: version 1 B-tree | Done |
| Chunk index version 4: single, implicit, fixed array, extensible array, B-tree | Done |
| Filters: shuffle, deflate, fletcher32, zstd | Done |
| Filters: blosc, with blosclz, lz4, zlib, zstd and both shuffles | Done |
| Values: integer, float, fixed string, variable string, variable sequence | Done |
| Big-endian values | Done |
| Contiguous, chunked and compact storage | Done |
| Fill values | Done |
| netCDF layer: dimensions, variables, axes, attributes | Done |
| netCDF classic: CDF-1 and CDF-2 | Done |
| Cloud reads, ndarray output, async engine | Done |

## Missing features

- **Szip.** The filter uses Rice coding. A decoder needs several hundred lines.
  No test validates it yet. The reader reports `Unsupported`.
  `test_files/generate_szip.c` makes a test file. A decoder can then join the
  netcdf-c comparison tests.
- **Blosc with snappy.** The reader decodes the other four sub-codecs.
- **Extensible arrays that need secondary blocks.** The reader reads the index
  block and the data blocks it addresses. A larger array reports `Unsupported`.
  The dataset stays in the list. A read of it fails. Other datasets still work.
- **Paged data blocks** for the array indexes.
- **CDF-5.** The code exists. No test file exists. Treat it as unverified.
- **Async open.** An open walks B-trees. The async engine has no B-tree walk.
  Open the file synchronously, then read asynchronously.
- **Batched reads across variables.** A read merges ranges inside one read. A
  scan of many small variables still makes one request for each variable.

## Tests

Run every test:

```bash
cargo test -p oxcdf --features "diff-tests,object-store,ndarray,async"
```

| Test file | Purpose |
|---|---|
| `differential.rs` | Compare values against netcdf-c, element by element |
| `netcdf_layer.rs` | Compare variables, dimensions and axes against `ncdump` |
| `read_values.rs` | Compare values against arithmetic |
| `chunk_indexes.rs` | Read the same values through all five version 4 indexes |
| `async_engine.rs` | Compare the async engine against the sync engine |
| `interface.rs` | Slices, chunks and attributes |
| `new_features.rs` | Variable strings, filters and classic files |
| `open_options.rs` | Request size and cache size |

The differential tests need the netcdf-c library. They stay off by default. The
crate exists to avoid that library.

Float values compare by bit pattern, not by tolerance. The reader moves bytes.
It computes nothing. A bit comparison also catches `NaN` and negative zero.

## Test files

`test_files/legacy_v1_objheader.h5` uses version 1 object headers, symbol table
groups and local heaps. The netCDF files never use that layout. Build it with
`generate_legacy.c`.

`test_files/latest_v4_layout.h5` uses all five version 4 chunk indexes. Build it
with `generate_latest.c`.

Rebuild a test file:

```bash
cd test_files && h5cc -o generate generate_legacy.c && ./generate
```
