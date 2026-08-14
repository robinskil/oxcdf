//! The read path for a hyperslab.
//!
//! Every function here is a pure function of an immutable [`DatasetIndex`], a
//! byte source and a request. There are no locks. There is no shared mutable
//! state. There is no file position.
//!
//! Two threads that read two hyperslabs of one dataset share only immutable
//! data. That is the purpose of this crate.
//!
//! Output is row-major. Output is in native byte order. An Arrow builder or an
//! `ndarray` accepts it directly.

use crate::context::Ctx;
use crate::error::{Error, Result};
use crate::filters;
use crate::index::DatasetIndex;
use crate::message::{ByteOrder, DatatypeClass, Layout, StringPad};

/// A rectangular selection of a dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hyperslab {
    /// Index of the first element along each axis.
    pub start: Vec<u64>,
    /// Number of elements along each axis.
    pub count: Vec<u64>,
}

impl Hyperslab {
    /// The selection covering a whole dataset.
    pub fn all(shape: &[u64]) -> Self {
        Self {
            start: vec![0; shape.len()],
            count: shape.to_vec(),
        }
    }

    /// Build a selection, checking it against `shape`.
    pub fn new(start: Vec<u64>, count: Vec<u64>, shape: &[u64]) -> Result<Self> {
        let slab = Self { start, count };
        slab.validate(shape)?;
        Ok(slab)
    }

    /// Build a selection from one range for each axis.
    ///
    /// `what` names the variable in an error message.
    pub fn from_ranges(what: &str, shape: &[u64], ranges: &[std::ops::Range<u64>]) -> Result<Self> {
        if ranges.len() != shape.len() {
            return Err(Error::bad_request(format!(
                "variable {what} has rank {} but {} ranges were given",
                shape.len(),
                ranges.len()
            )));
        }
        let mut start = Vec::with_capacity(ranges.len());
        let mut count = Vec::with_capacity(ranges.len());
        for (axis, r) in ranges.iter().enumerate() {
            if r.end < r.start {
                return Err(Error::bad_request(format!(
                    "range on axis {axis} of variable {what} is reversed"
                )));
            }
            start.push(r.start);
            count.push(r.end - r.start);
        }
        Ok(Self { start, count })
    }

    /// Total number of elements selected.
    pub fn element_count(&self) -> u64 {
        element_count(&self.count)
    }

    /// Check the selection lies inside `shape`.
    pub fn validate(&self, shape: &[u64]) -> Result<()> {
        if self.start.len() != shape.len() || self.count.len() != shape.len() {
            return Err(Error::bad_request(format!(
                "selection has rank {}/{} but the dataset has rank {}",
                self.start.len(),
                self.count.len(),
                shape.len()
            )));
        }
        for (axis, ((&s, &c), &dim)) in self
            .start
            .iter()
            .zip(self.count.iter())
            .zip(shape.iter())
            .enumerate()
        {
            let end = s
                .checked_add(c)
                .ok_or_else(|| Error::bad_request(format!("selection on axis {axis} overflows")))?;
            if end > dim {
                return Err(Error::bad_request(format!(
                    "selection [{s}, {end}) on axis {axis} runs past the dimension size {dim}"
                )));
            }
        }
        Ok(())
    }
}

/// How many elements a shape holds, saturating rather than overflowing.
///
/// Every dimension here comes from the file. A damaged one multiplies past
/// `u64`, and the answer a reader wants for that is "more than anything can
/// hold", which is what `u64::MAX` says. Wrapping would say "a handful", and
/// the read would go on to trust it.
///
/// Saturating is safe because of what callers do with the count: they compare
/// it against a limit, or multiply it into a byte count with
/// [`u64::checked_mul`] and refuse the read when that fails. Neither turns a
/// saturated count into a wrong answer.
pub fn element_count(dims: &[u64]) -> u64 {
    saturating_product(dims.iter().copied())
}

/// The product of an iterator of dimensions, saturating at [`u64::MAX`].
fn saturating_product(dims: impl Iterator<Item = u64>) -> u64 {
    dims.fold(1u64, |total, dim| total.saturating_mul(dim))
}

/// How many bytes one chunk decodes to.
///
/// A chunk shape and an element width both come from the file, so their product
/// can exceed anything this machine could hold. That is a damaged file rather
/// than a large one: no chunk of that size was ever written. Say so, instead of
/// wrapping the width and decoding into a buffer of the wrong size.
fn decoded_chunk_len(chunk_shape: &[u64], element_size: usize, what: &str) -> Result<usize> {
    element_count(chunk_shape)
        .checked_mul(element_size as u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| {
            Error::malformed(format!(
                "one chunk of dataset {what} claims more bytes than this reader can hold"
            ))
        })
}

/// Raw values read from a dataset, in native byte order and row-major layout.
#[derive(Debug, Clone)]
pub struct RawData {
    /// The bytes.
    pub bytes: Vec<u8>,
    /// Width of one element.
    pub element_size: usize,
    /// Shape of the returned block.
    pub shape: Vec<u64>,
}

impl RawData {
    /// Number of elements.
    pub fn len(&self) -> usize {
        if self.element_size == 0 {
            0
        } else {
            self.bytes.len() / self.element_size
        }
    }

    /// Whether the block is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Decode as `T`.
    pub fn get<T: crate::dtype::Element>(&self, dataset: &DatasetIndex) -> Result<Vec<T>> {
        self.get_of(&dataset.datatype, &dataset.path)
    }

    /// Decode as `T`, given the datatype directly.
    ///
    /// A stored type equal to `T` copies the values. Any other numeric type
    /// converts, which is what the `netcdf` crate does. A stored string or
    /// compound returns [`Error::TypeMismatch`], naming the stored type.
    ///
    /// A conversion can lose information. See [`crate::dtype::Element`].
    pub fn get_of<T: crate::dtype::Element>(
        &self,
        datatype: &crate::message::Datatype,
        what: &str,
    ) -> Result<Vec<T>> {
        let size = self.element_size;

        // The stored type is the asked-for type: copy, do not convert. That is
        // what a read of a netCDF variable normally is, and
        // [`crate::dtype::Element::from_ne_slice`] makes it the copy it should
        // be. Walking the buffer element by element instead cost a third of a
        // whole scan in issue #2.
        if crate::dtype::DType::of(datatype) == T::DTYPE && size == std::mem::size_of::<T>() {
            return Ok(T::from_ne_slice(&self.bytes));
        }

        // `len` is already zero for a zero-width element, which a damaged file
        // can declare. `chunks_exact` would panic on that width, so ask it for
        // one byte and take nothing.
        let elements = self.bytes.chunks_exact(size.max(1)).take(self.len());
        match &datatype.class {
            DatatypeClass::FixedPoint { signed: true, .. } => elements
                .map(|b| Ok(T::from_i64(read_integer(b, true)?)))
                .collect(),
            DatatypeClass::FixedPoint { signed: false, .. } => elements
                .map(|b| Ok(T::from_u64(read_unsigned(b)?)))
                .collect(),
            DatatypeClass::FloatingPoint { .. } => elements
                .map(|b| {
                    let v = match size {
                        4 => f32::from_ne_bytes(b.try_into().unwrap()) as f64,
                        8 => f64::from_ne_bytes(b.try_into().unwrap()),
                        other => {
                            return Err(Error::unsupported(format!("{other}-byte floating point")))
                        }
                    };
                    Ok(T::from_f64(v))
                })
                .collect(),
            _ => Err(Error::TypeMismatch {
                stored: crate::dtype::DType::of(datatype).name(),
                asked: T::NAME,
                what: what.to_string(),
            }),
        }
    }

    /// Decode fixed-length string elements.
    pub fn to_strings(&self, dataset: &DatasetIndex) -> Result<Vec<String>> {
        self.to_strings_of(&dataset.datatype)
    }

    /// Decode fixed-length string elements, given the datatype directly.
    pub fn to_strings_of(&self, datatype: &crate::message::Datatype) -> Result<Vec<String>> {
        let DatatypeClass::String { pad, .. } = &datatype.class else {
            return Err(Error::unsupported(format!(
                "decoding {:?} as strings",
                datatype.class
            )));
        };

        // One string for each element. A netCDF `char` variable stores one byte
        // for each element, so it decodes to one string for each character. The
        // last dimension is its string length. This reader reports the elements
        // as stored and leaves that join to the caller.
        let size = self.element_size;
        let mut out = Vec::with_capacity(self.len());

        for i in 0..self.len() {
            let raw = &self.bytes[i * size..(i + 1) * size];
            let end = match pad {
                // A NUL ends the value whatever the declared width.
                StringPad::NullTerminate | StringPad::NullPad => {
                    raw.iter().position(|&b| b == 0).unwrap_or(raw.len())
                }
                // Fortran-style: trailing spaces are padding, not content.
                StringPad::SpacePad => {
                    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                    raw[..end]
                        .iter()
                        .rposition(|&b| b != b' ')
                        .map(|p| p + 1)
                        .unwrap_or(0)
                }
            };
            out.push(String::from_utf8_lossy(&raw[..end]).into_owned());
        }

        Ok(out)
    }
}

/// Whether the data a read is about to copy covers every byte of its output.
///
/// A read starts from the fill value, because storage that was never written
/// must read as it. When the copies that follow cover the whole selection,
/// every one of those bytes is overwritten and the fill is a wasted pass over
/// the buffer. A netCDF4 variable stored as one chunk is exactly that case, and
/// the pass cost 8.5% of the profile in issue #2.
///
/// `chunks` is the resolved chunk index, needed only for a chunked dataset.
fn selection_is_covered(
    dataset: &DatasetIndex,
    chunks: Option<&[crate::btree1::ChunkRecord]>,
    slab: &Hyperslab,
) -> bool {
    match &dataset.layout {
        // The whole image is in memory and every run of the selection is
        // copied out of it.
        Layout::Compact { .. } => true,
        // A written contiguous dataset is one run per row of the selection,
        // and the runs tile the output exactly. An unwritten one has no bytes
        // at all, so the fill value is the answer.
        Layout::Contiguous { address, .. } => address.is_some(),
        Layout::Chunked { chunk_dims, .. } => {
            let rank = dataset.shape.len();
            if rank == 0 || chunk_dims.len() != rank {
                return false;
            }
            let chunk_shape: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
            chunks.is_some_and(|chunks| chunks_cover(chunks, &chunk_shape, slab, rank))
        }
    }
}

/// Whether the stored chunks cover every element of the selection.
///
/// The test marks each cell of the chunk grid the selection touches. Summing
/// the intersection volumes would be shorter and wrong: a damaged index that
/// names one chunk twice would then stand in for a chunk that is missing, and
/// the read would return zeros where it owes the fill value.
///
/// A grid holding more cells than the index holds chunks is missing one, so the
/// marker allocation never grows past the chunk count.
fn chunks_cover(
    chunks: &[crate::btree1::ChunkRecord],
    chunk_shape: &[u64],
    slab: &Hyperslab,
    rank: usize,
) -> bool {
    if slab.element_count() == 0 {
        return true; // nothing to fill
    }

    // The block of grid cells the selection touches.
    let mut first = vec![0u64; rank];
    let mut span = vec![0u64; rank];
    let mut cells = 1u64;
    for axis in 0..rank {
        let width = chunk_shape[axis];
        if width == 0 {
            return false;
        }
        let lo = slab.start[axis] / width;
        let hi = (slab.start[axis] + slab.count[axis] - 1) / width;
        first[axis] = lo;
        span[axis] = hi - lo + 1;
        match cells.checked_mul(span[axis]) {
            // More cells than chunks means at least one cell has no chunk.
            Some(n) if n <= chunks.len() as u64 => cells = n,
            _ => return false,
        }
    }

    let mut seen = vec![false; cells as usize];
    let mut missing = cells;
    for record in chunks {
        if record.size == 0 || record.offset.len() != rank {
            continue; // never written, or not this dataset's shape
        }
        let mut cell = 0u64;
        let mut inside = true;
        for axis in 0..rank {
            let width = chunk_shape[axis];
            let offset = record.offset[axis];
            // A chunk off the grid is not this reader's idea of a chunk.
            if !offset.is_multiple_of(width) {
                inside = false;
                break;
            }
            let index = offset / width;
            if index < first[axis] || index >= first[axis] + span[axis] {
                inside = false;
                break;
            }
            cell = cell * span[axis] + (index - first[axis]);
        }
        if inside && !seen[cell as usize] {
            seen[cell as usize] = true;
            missing -= 1;
            if missing == 0 {
                return true;
            }
        }
    }
    false
}

/// Whether a chunk overlaps the selection.
///
/// A chunk offset comes from the file and a damaged one can sit anywhere in the
/// address space, so its far edge saturates. A chunk that claims to reach past
/// the end of the address space does reach past the selection, which is inside
/// the dataset's shape, so saturating gives the right answer as well as a safe
/// one. Every other place that takes a chunk's far edge does the same.
fn intersects(
    record: &crate::btree1::ChunkRecord,
    chunk_shape: &[u64],
    slab: &Hyperslab,
    rank: usize,
) -> bool {
    (0..rank).all(|axis| {
        let chunk_lo = record.offset[axis];
        let chunk_hi = chunk_lo.saturating_add(chunk_shape[axis]);
        let sel_lo = slab.start[axis];
        let sel_hi = sel_lo + slab.count[axis];
        chunk_lo.max(sel_lo) < chunk_hi.min(sel_hi)
    })
}

/// Fetch and decode the chunks a read needs, plus a few beyond them.
///
/// One batched call fetches everything the cache lacks. A remote source merges
/// neighbouring ranges and sends the rest together. The whole read then costs a
/// few round trips, not one for each chunk.
///
/// The read-ahead reaches past the last needed chunk. A scan walks chunks in
/// order, so those are the likely next need.
///
/// A random-access pattern wastes that work. Turn it off with
/// `ChunkCache::with_readahead(0)`.
fn prefetch_chunks(
    ctx: Ctx<'_>,
    dataset: &DatasetIndex,
    chunks: &[crate::btree1::ChunkRecord],
    needed: &[usize],
    decoded_len: usize,
) -> Result<()> {
    let Some(cache) = ctx.cache else {
        return Ok(());
    };
    let Some(&last) = needed.last() else {
        return Ok(());
    };

    // The chunks this read needs, then the read-ahead window after them.
    let lookahead_end = (last + 1 + cache.readahead()).min(chunks.len());
    let candidates = needed.iter().copied().chain((last + 1)..lookahead_end);

    let mut wanted = Vec::new();
    for i in candidates {
        let record = &chunks[i];
        // Skip what is already decoded, and anything unwritten.
        if record.size == 0 || cache.contains(record.address) {
            continue;
        }
        wanted.push((record.address, record.size as usize, record.filter_mask));
    }
    if wanted.is_empty() {
        return Ok(());
    }

    let ranges: Vec<(u64, usize)> = wanted.iter().map(|&(a, n, _)| (a, n)).collect();
    let blobs = ctx.read_ranges(&ranges)?;

    for ((address, _, filter_mask), raw) in wanted.into_iter().zip(blobs) {
        // A prefetched chunk that fails to decode is not this read's problem
        // unless the read actually needs it, and then the main loop reports it.
        if let Ok(decoded) = filters::decode_chunk(&dataset.pipeline, filter_mask, raw, decoded_len)
        {
            cache.insert(address, bytes::Bytes::from(decoded));
        }
    }

    Ok(())
}

/// Read a little-endian-normalised integer of 1 to 8 bytes.
/// Read a native-order unsigned integer of 1 to 8 bytes.
fn read_unsigned(bytes: &[u8]) -> Result<u64> {
    let mut buf = [0u8; 8];
    let n = bytes.len();
    if n == 0 || n > 8 {
        return Err(Error::unsupported(format!("{n}-byte integer")));
    }
    #[cfg(target_endian = "little")]
    buf[..n].copy_from_slice(bytes);
    #[cfg(target_endian = "big")]
    buf[8 - n..].copy_from_slice(bytes);
    Ok(u64::from_ne_bytes(buf))
}

fn read_integer(bytes: &[u8], signed: bool) -> Result<i64> {
    let mut buf = [0u8; 8];
    let n = bytes.len();
    if n == 0 || n > 8 {
        return Err(Error::unsupported(format!("{n}-byte integer")));
    }
    // Values arrive in native order; copy into the low bytes and sign-extend.
    #[cfg(target_endian = "little")]
    buf[..n].copy_from_slice(bytes);
    #[cfg(target_endian = "big")]
    buf[8 - n..].copy_from_slice(bytes);

    let raw = u64::from_ne_bytes(buf);
    if !signed {
        return Ok(raw as i64);
    }
    // Sign-extend from the stored width.
    let shift = 64 - (n as u32 * 8);
    Ok(((raw << shift) as i64) >> shift)
}

/// Resolve variable-length string elements.
///
/// An inline vlen element is only a descriptor. It holds a length, the address
/// of a global heap collection, and an index into it. The characters live in
/// that collection.
///
/// A decode therefore needs the file, not just the bytes. That is why this is
/// not a method on [`RawData`].
///
/// One call caches the collections it reads. A variable's strings normally
/// share very few of them.
pub fn resolve_vlen_strings(
    ctx: Ctx<'_>,
    dataset: &DatasetIndex,
    raw: &RawData,
) -> Result<Vec<String>> {
    let DatatypeClass::VariableLength { kind, .. } = &dataset.datatype.class else {
        return Err(Error::unsupported(format!(
            "dataset {} is not a variable-length type",
            dataset.path
        )));
    };
    if *kind != crate::message::VlenKind::String {
        return Err(Error::unsupported(
            "variable-length sequences are not decoded as strings",
        ));
    }
    resolve_vlen_strings_of(ctx, &dataset.path, raw)
}

/// Follow variable-length string descriptors into the global heap.
///
/// This takes the descriptors directly, so an attribute uses it as well as a
/// variable. `what` names the source in an error message.
pub fn resolve_vlen_strings_of(ctx: Ctx<'_>, what: &str, raw: &RawData) -> Result<Vec<String>> {
    with_vlen_descriptors(ctx, what, raw, |bytes, length| {
        let len = (length as usize).min(bytes.len());
        let bytes = &bytes[..len];
        // A stored string may still carry a terminator inside its length.
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
    })
}

/// Resolve variable-length sequence elements.
///
/// Each element is a descriptor naming a run in the global heap. Unlike a vlen
/// string, the run holds values of a base type, so the bytes are byte-swapped
/// into native order here just as a fixed-size read would be.
///
/// Returns one buffer per element. An empty sequence yields an empty buffer,
/// which is a real value and not an error.
pub fn resolve_vlen_sequences(
    ctx: Ctx<'_>,
    dataset: &DatasetIndex,
    raw: &RawData,
) -> Result<Vec<Vec<u8>>> {
    let DatatypeClass::VariableLength { kind, base, .. } = &dataset.datatype.class else {
        return Err(Error::unsupported(format!(
            "dataset {} is not a variable-length type",
            dataset.path
        )));
    };
    if *kind != crate::message::VlenKind::Sequence {
        return Err(Error::unsupported(
            "this variable holds variable-length strings, not sequences",
        ));
    }

    let base_size = base.size as usize;
    let swap = matches!(base.byte_order(), Some(ByteOrder::Big)) && base_size > 1;
    let what = dataset.path.as_str();

    with_vlen_descriptors(ctx, what, raw, |bytes, length| {
        let want = (length as usize).saturating_mul(base_size);
        let take = want.min(bytes.len());
        let mut values = bytes[..take].to_vec();
        if swap {
            for chunk in values.chunks_exact_mut(base_size) {
                chunk.reverse();
            }
        }
        Ok(values)
    })
}

/// Walk the vlen descriptors of a read, handing each element's heap bytes to
/// `decode`.
///
/// Global heap collections are cached for the call: one variable's elements
/// normally share very few of them.
fn with_vlen_descriptors<T, F>(
    ctx: Ctx<'_>,
    what: &str,
    raw: &RawData,
    mut decode: F,
) -> Result<Vec<T>>
where
    F: FnMut(&[u8], u32) -> Result<T>,
{
    use crate::heap::{GlobalHeap, VlenDescriptor};
    use std::collections::HashMap;

    let sizes = ctx.sizes();
    let width = VlenDescriptor::encoded_len(sizes);
    if raw.element_size < width {
        return Err(Error::malformed(format!(
            "variable-length element is {} bytes, too small for a {width}-byte descriptor",
            raw.element_size
        )));
    }

    let mut collections: HashMap<u64, GlobalHeap> = HashMap::new();
    let mut out = Vec::with_capacity(raw.len());

    for i in 0..raw.len() {
        let start = i * raw.element_size;
        let descriptor = VlenDescriptor::parse(&raw.bytes[start..start + width], sizes)?;

        if descriptor.length == 0 {
            out.push(decode(&[], 0)?);
            continue;
        }

        let heap = match collections.entry(descriptor.collection_address) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(GlobalHeap::read(ctx, descriptor.collection_address)?)
            }
        };

        let object = heap.object(descriptor.object_index as u16).ok_or_else(|| {
            Error::malformed(format!(
                "variable-length element {i} of {} names global heap object {} which is absent",
                what, descriptor.object_index
            ))
        })?;

        out.push(decode(&object.data, descriptor.length)?);
    }

    Ok(out)
}

/// Read a hyperslab of a dataset.
///
/// The dataset must come from the same file as `ctx`.
pub fn read_hyperslab(ctx: Ctx<'_>, dataset: &DatasetIndex, slab: &Hyperslab) -> Result<RawData> {
    slab.validate(&dataset.shape)?;

    if !dataset.datatype.is_decodable() {
        return Err(Error::unsupported(format!(
            "datatype {:?} of dataset {}",
            dataset.datatype.class, dataset.path
        )));
    }

    let element_size = dataset.element_size();
    let total = slab
        .element_count()
        .checked_mul(element_size as u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::bad_request("selection is too large to hold in memory"))?;

    // Resolve the chunk index first: whether the fill value is needed at all
    // depends on which chunks are stored.
    let chunks = match &dataset.layout {
        Layout::Chunked { .. } => Some(
            dataset
                .chunks(ctx)?
                .ok_or_else(|| Error::malformed("chunked dataset has no chunk index"))?,
        ),
        _ => None,
    };

    // Start from the fill value, not from zeros. A dataset that was never
    // written, or a chunk that was never allocated, must read as its fill
    // value; netCDF's default for a 32-bit integer is -2147483647, not 0.
    //
    // Stored data that covers the whole selection overwrites every one of those
    // bytes, so the pass is skipped there.
    let mut out = vec![0u8; total];
    if !selection_is_covered(dataset, chunks, slab) {
        dataset.fill_value.fill(&mut out, element_size);
    }

    match &dataset.layout {
        Layout::Compact { data } => {
            copy_from_dense(&mut out, data, &dataset.shape, slab, element_size)?;
        }
        Layout::Contiguous { address, .. } => match address {
            // Never written: the fill value already in the buffer is the answer.
            None => {}
            Some(address) => {
                read_contiguous(ctx, *address, &mut out, &dataset.shape, slab, element_size)?;
            }
        },
        Layout::Chunked { chunk_dims, .. } => {
            let chunks =
                chunks.ok_or_else(|| Error::malformed("chunked dataset has no chunk index"))?;
            read_chunked(
                ctx,
                dataset,
                chunks,
                chunk_dims,
                &mut out,
                slab,
                element_size,
            )?;
        }
    }

    // Variable-length elements are descriptors, whose fields are always stored
    // little-endian. Swapping them would corrupt the heap addresses.
    let is_vlen = matches!(dataset.datatype.class, DatatypeClass::VariableLength { .. });
    if !is_vlen {
        if let Some(ByteOrder::Big) = dataset.datatype.byte_order() {
            swap_bytes_in_place(&mut out, element_size);
        }
    }

    Ok(RawData {
        bytes: out,
        element_size,
        shape: slab.count.clone(),
    })
}

/// Byte-swap every element in place, turning big-endian values into native.
fn swap_bytes_in_place(buf: &mut [u8], element_size: usize) {
    if element_size <= 1 {
        return;
    }
    for chunk in buf.chunks_exact_mut(element_size) {
        chunk.reverse();
    }
}

/// Row-major strides, in elements, for a shape.
fn strides(shape: &[u64]) -> Vec<u64> {
    let mut out = vec![1u64; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        out[axis] = out[axis + 1] * shape[axis + 1];
    }
    out
}

/// Walk every contiguous run of the selection.
///
/// A run is the largest block that is contiguous in both source and
/// destination. That is not only the innermost axis. A trailing axis selected
/// in full merges into the run above it.
///
/// A whole-variable read is therefore one run, not one for each row. That turns
/// one `pread` and one allocation for each row into one pair for the read.
///
/// `visit` gets the source element offset, the destination element offset and
/// the run length in elements.
fn for_each_run<F>(shape: &[u64], slab: &Hyperslab, mut visit: F) -> Result<()>
where
    F: FnMut(u64, u64, u64) -> Result<()>,
{
    let rank = shape.len();
    if rank == 0 {
        // A scalar dataset is a single element.
        return visit(0, 0, 1);
    }
    if slab.element_count() == 0 {
        return Ok(());
    }

    let src_strides = strides(shape);
    let dst_strides = strides(&slab.count);

    // Grow the run outwards while each trailing axis is taken in full.
    let mut first = rank - 1;
    while first > 0 && slab.start[first] == 0 && slab.count[first] == shape[first] {
        first -= 1;
    }
    let run: u64 = element_count(&slab.count[first..]);
    if run == 0 {
        return Ok(());
    }

    // Iterate whatever axes remain outside the run.
    let outer: u64 = element_count(&slab.count[..first]);
    let mut index = vec![0u64; first];

    for _ in 0..outer {
        let mut src = 0u64;
        let mut dst = 0u64;
        for axis in 0..first {
            src += (slab.start[axis] + index[axis]) * src_strides[axis];
            dst += index[axis] * dst_strides[axis];
        }
        src += slab.start[first] * src_strides[first];

        visit(src, dst, run)?;

        // Odometer increment over the outer axes.
        for axis in (0..first).rev() {
            index[axis] += 1;
            if index[axis] < slab.count[axis] {
                break;
            }
            index[axis] = 0;
        }
    }

    Ok(())
}

/// Copy a selection out of an in-memory image of the whole dataset.
fn copy_from_dense(
    out: &mut [u8],
    src: &[u8],
    shape: &[u64],
    slab: &Hyperslab,
    element_size: usize,
) -> Result<()> {
    for_each_run(shape, slab, |s, d, run| {
        let sb = s as usize * element_size;
        let db = d as usize * element_size;
        let len = run as usize * element_size;
        if sb + len > src.len() {
            return Err(Error::malformed(
                "stored data is shorter than the dataset's shape implies",
            ));
        }
        out[db..db + len].copy_from_slice(&src[sb..sb + len]);
        Ok(())
    })
}

/// Read a selection out of a contiguous dataset, one run per read.
fn read_contiguous(
    ctx: Ctx<'_>,
    address: u64,
    out: &mut [u8],
    shape: &[u64],
    slab: &Hyperslab,
    element_size: usize,
) -> Result<()> {
    for_each_run(shape, slab, |s, d, run| {
        let len = run as usize * element_size;
        let bytes = ctx.read(address + s * element_size as u64, len)?;
        let db = d as usize * element_size;
        out[db..db + len].copy_from_slice(&bytes);
        Ok(())
    })
}

/// Read a selection out of a chunked dataset.
///
/// Only chunks that intersect the selection are fetched and decoded. Each chunk
/// is independent, which is where a caller gets its parallelism: this loop can
/// become a parallel iterator without any further synchronisation.
fn read_chunked(
    ctx: Ctx<'_>,
    dataset: &DatasetIndex,
    chunks: &[crate::btree1::ChunkRecord],
    chunk_dims: &[u32],
    out: &mut [u8],
    slab: &Hyperslab,
    element_size: usize,
) -> Result<()> {
    let rank = dataset.shape.len();
    if chunk_dims.len() != rank {
        return Err(Error::malformed(format!(
            "dataset {} has rank {rank} but its chunk shape has rank {}",
            dataset.path,
            chunk_dims.len()
        )));
    }

    let dst_strides = strides(&slab.count);
    let chunk_shape: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
    let decoded_len = decoded_chunk_len(&chunk_shape, element_size, &dataset.path)?;
    let chunk_strides = strides(&chunk_shape);

    // Which chunks this read touches. Computing them up front turns the reads
    // into one batched call instead of one per chunk, and gives the read-ahead
    // something to extend.
    let mut needed = Vec::new();
    for (i, record) in chunks.iter().enumerate() {
        if record.offset.len() != rank {
            return Err(Error::malformed("chunk index entry has the wrong rank"));
        }
        if intersects(record, &chunk_shape, slab, rank) {
            needed.push(i);
        }
    }

    prefetch_chunks(ctx, dataset, chunks, &needed, decoded_len)?;

    for &i in &needed {
        let record = &chunks[i];

        // Intersect this chunk with the selection.
        let mut lo = vec![0u64; rank];
        let mut hi = vec![0u64; rank];
        let mut empty = false;
        for axis in 0..rank {
            let chunk_lo = record.offset[axis];
            let chunk_hi = chunk_lo.saturating_add(chunk_shape[axis]);
            let sel_lo = slab.start[axis];
            let sel_hi = slab.start[axis] + slab.count[axis];
            lo[axis] = chunk_lo.max(sel_lo);
            hi[axis] = chunk_hi.min(sel_hi);
            if lo[axis] >= hi[axis] {
                empty = true;
                break;
            }
        }
        if empty {
            continue;
        }

        // Decoding is a read plus an inflate plus an unshuffle. Cache the
        // result so a second hyperslab over the same chunk, or another thread,
        // does not repeat it.
        let decoded = match ctx.cache {
            Some(cache) => cache.get_or_decode(record.address, || {
                let raw = ctx.read(record.address, record.size as usize)?;
                filters::decode_chunk(&dataset.pipeline, record.filter_mask, raw, decoded_len)
            })?,
            None => {
                let raw = ctx.read(record.address, record.size as usize)?;
                bytes::Bytes::from(filters::decode_chunk(
                    &dataset.pipeline,
                    record.filter_mask,
                    raw,
                    decoded_len,
                )?)
            }
        };
        if decoded.len() < decoded_len {
            return Err(Error::malformed(format!(
                "chunk at {:?} of dataset {} decoded to {} bytes, expected {decoded_len}",
                record.offset,
                dataset.path,
                decoded.len()
            )));
        }

        // Copy the intersection, innermost axis at a time.
        let run = hi[rank - 1] - lo[rank - 1];
        let outer: u64 = saturating_product((0..rank - 1).map(|a| hi[a] - lo[a]));
        let mut index = vec![0u64; rank.saturating_sub(1)];

        for _ in 0..outer.max(if rank == 1 { 1 } else { 0 }) {
            let mut src = 0u64;
            let mut dst = 0u64;
            for axis in 0..rank - 1 {
                let coord = lo[axis] + index[axis];
                src += (coord - record.offset[axis]) * chunk_strides[axis];
                dst += (coord - slab.start[axis]) * dst_strides[axis];
            }
            src += (lo[rank - 1] - record.offset[rank - 1]) * chunk_strides[rank - 1];
            dst += (lo[rank - 1] - slab.start[rank - 1]) * dst_strides[rank - 1];

            let sb = src as usize * element_size;
            let db = dst as usize * element_size;
            let len = run as usize * element_size;
            out[db..db + len].copy_from_slice(&decoded[sb..sb + len]);

            for axis in (0..rank.saturating_sub(1)).rev() {
                index[axis] += 1;
                if index[axis] < hi[axis] - lo[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
    }

    Ok(())
}

// ─── the asynchronous engine ───────────────────────────────────────────────

/// Read a hyperslab through an [`crate::async_source::AsyncByteSource`].
///
/// This is the asynchronous twin of [`read_hyperslab`]. It shares every pure
/// part: the chunk index, the run coalescing, the filters, the assembly and the
/// byte order. Only the fetch differs.
///
/// The shape is deliberate:
///
/// 1. Work out every byte range the read needs. This is pure, with no input.
/// 2. Fetch them all in **one** batched `await`.
/// 3. Decode and assemble synchronously.
///
/// Step 3 uses the processor: inflate, unshuffle and copy. It stays synchronous
/// on purpose. An asynchronous step 3 runs on runtime workers and stalls the
/// reactor. That is the mistake this engine exists to avoid.
///
/// For a heavy decode, hand step 3 to `spawn_blocking`. This function awaits
/// before that work starts.
///
/// This function fetches data, not metadata. Resolve the dataset's chunk index
/// before the call. Most callers use [`crate::async_hdf5::AsyncDataset::read`],
/// which resolves the index itself.
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub async fn read_hyperslab_async(
    source: &dyn crate::async_source::AsyncByteSource,
    superblock: &crate::superblock::Superblock,
    cache: Option<&crate::cache::ChunkCache>,
    dataset: &DatasetIndex,
    slab: &Hyperslab,
) -> Result<RawData> {
    read_hyperslab_async_with(
        source,
        superblock,
        cache,
        None,
        crate::io::IoConfig::REMOTE,
        dataset,
        slab,
    )
    .await
}

/// As [`read_hyperslab_async`], with an explicit byte-range merging policy.
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
#[allow(clippy::too_many_arguments)]
pub async fn read_hyperslab_async_with(
    source: &dyn crate::async_source::AsyncByteSource,
    superblock: &crate::superblock::Superblock,
    cache: Option<&crate::cache::ChunkCache>,
    io_cache: Option<&crate::cache::IoCache>,
    io: crate::io::IoConfig,
    dataset: &DatasetIndex,
    slab: &Hyperslab,
) -> Result<RawData> {
    slab.validate(&dataset.shape)?;
    if !dataset.datatype.is_decodable() {
        return Err(Error::unsupported(format!(
            "datatype {:?} of dataset {}",
            dataset.datatype.class, dataset.path
        )));
    }

    let element_size = dataset.element_size();
    let total = slab
        .element_count()
        .checked_mul(element_size as u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::bad_request("selection is too large to hold in memory"))?;

    // The chunk index is metadata, which this function does not fetch, so a
    // chunked dataset arrives with it resolved. Whether the fill value is
    // needed at all depends on which chunks are stored.
    let mut out = vec![0u8; total];
    if !selection_is_covered(dataset, dataset.resolved_chunks(), slab) {
        dataset.fill_value.fill(&mut out, element_size);
    }

    match &dataset.layout {
        Layout::Compact { data } => {
            copy_from_dense(&mut out, data, &dataset.shape, slab, element_size)?;
        }

        Layout::Contiguous { address, .. } => {
            if let Some(address) = address {
                // Gather the runs first, then fetch them all at once. With run
                // coalescing a whole-variable read is a single range.
                let mut runs = Vec::new();
                for_each_run(&dataset.shape, slab, |src, dst, run| {
                    runs.push((src, dst, run));
                    Ok(())
                })?;

                let ranges: Vec<(u64, usize)> = runs
                    .iter()
                    .map(|&(src, _, run)| {
                        (
                            superblock.resolve(address + src * element_size as u64),
                            run as usize * element_size,
                        )
                    })
                    .collect();
                let blobs = fetch_bytes(source, io_cache, &ranges, io).await?;

                for ((_, dst, run), bytes) in runs.into_iter().zip(blobs) {
                    let db = dst as usize * element_size;
                    let len = run as usize * element_size;
                    if bytes.len() < len {
                        return Err(Error::malformed(
                            "contiguous read returned fewer bytes than requested",
                        ));
                    }
                    out[db..db + len].copy_from_slice(&bytes[..len]);
                }
            }
        }

        Layout::Chunked { chunk_dims, .. } => {
            // This function fetches data, not metadata. A chunk index is
            // metadata, so the caller resolves it first.
            // `crate::async_hdf5::AsyncDataset` does that. Say so plainly
            // rather than silently reading nothing.
            let chunks = dataset.resolved_chunks().ok_or_else(|| {
                Error::bad_request(format!(
                    "dataset {} has no resolved chunk index; read it through \
                     `AsyncNetcdfFile`, which resolves the index itself",
                    dataset.path
                ))
            })?;

            let rank = dataset.shape.len();
            if chunk_dims.len() != rank {
                return Err(Error::malformed(format!(
                    "dataset {} has rank {rank} but its chunk shape has rank {}",
                    dataset.path,
                    chunk_dims.len()
                )));
            }
            let chunk_shape: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
            let decoded_len = decoded_chunk_len(&chunk_shape, element_size, &dataset.path)?;

            // Which chunks this read touches.
            let needed: Vec<usize> = chunks
                .iter()
                .enumerate()
                .filter(|(_, r)| r.offset.len() == rank && intersects(r, &chunk_shape, slab, rank))
                .map(|(i, _)| i)
                .collect();

            // Fetch everything not already decoded, plus the read-ahead window,
            // in one batched call.
            let mut wanted = Vec::new();
            if let Some(&last) = needed.last() {
                let lookahead = cache.map(|c| c.readahead()).unwrap_or(0);
                let end = (last + 1 + lookahead).min(chunks.len());
                for i in needed.iter().copied().chain((last + 1)..end) {
                    let r = &chunks[i];
                    if r.size == 0 || cache.is_some_and(|c| c.contains(r.address)) {
                        continue;
                    }
                    wanted.push((r.address, r.size as usize, r.filter_mask));
                }
            }

            let mut fetched: std::collections::HashMap<u64, bytes::Bytes> =
                std::collections::HashMap::new();
            if !wanted.is_empty() {
                let ranges: Vec<(u64, usize)> = wanted
                    .iter()
                    .map(|&(a, n, _)| (superblock.resolve(a), n))
                    .collect();
                let blobs = fetch_bytes(source, io_cache, &ranges, io).await?;

                for ((address, _, mask), raw) in wanted.into_iter().zip(blobs) {
                    match filters::decode_chunk(&dataset.pipeline, mask, raw.to_vec(), decoded_len)
                    {
                        Ok(decoded) => {
                            let decoded = bytes::Bytes::from(decoded);
                            if let Some(c) = cache {
                                c.insert(address, decoded.clone());
                            }
                            fetched.insert(address, decoded);
                        }
                        Err(e) => {
                            // Only a chunk this read actually needs is fatal;
                            // a read-ahead failure is not.
                            if chunks[..].iter().any(|r| {
                                r.address == address && intersects(r, &chunk_shape, slab, rank)
                            }) {
                                return Err(e);
                            }
                        }
                    }
                }
            }

            let dst_strides = strides(&slab.count);
            let chunk_strides = strides(&chunk_shape);

            for &i in &needed {
                let record = &chunks[i];
                let decoded = match cache.and_then(|c| c.get(record.address)) {
                    Some(hit) => hit,
                    None => fetched.get(&record.address).cloned().ok_or_else(|| {
                        Error::malformed(format!(
                            "chunk at {:?} of {} was neither cached nor fetched",
                            record.offset, dataset.path
                        ))
                    })?,
                };
                if decoded.len() < decoded_len {
                    return Err(Error::malformed(format!(
                        "chunk at {:?} of dataset {} decoded to {} bytes, expected {decoded_len}",
                        record.offset,
                        dataset.path,
                        decoded.len()
                    )));
                }

                copy_chunk(
                    &mut out,
                    &decoded,
                    record,
                    &chunk_shape,
                    &chunk_strides,
                    slab,
                    &dst_strides,
                    element_size,
                    rank,
                );
            }
        }
    }

    let is_vlen = matches!(dataset.datatype.class, DatatypeClass::VariableLength { .. });
    if !is_vlen {
        if let Some(ByteOrder::Big) = dataset.datatype.byte_order() {
            swap_bytes_in_place(&mut out, element_size);
        }
    }

    Ok(RawData {
        bytes: out,
        element_size,
        shape: slab.count.clone(),
    })
}

/// Fetch ranges, through the page cache when there is one.
///
/// With a cache, paging already merges neighbours and reuses whatever earlier
/// reads pulled in, so it subsumes explicit coalescing. Without one, ranges are
/// merged into a single batch and sliced back apart zero-copy.
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
async fn fetch_bytes(
    source: &dyn crate::async_source::AsyncByteSource,
    io_cache: Option<&crate::cache::IoCache>,
    ranges: &[(u64, usize)],
    io: crate::io::IoConfig,
) -> Result<Vec<bytes::Bytes>> {
    if let Some(cache) = io_cache {
        let mut out = Vec::with_capacity(ranges.len());
        for &(offset, len) in ranges {
            out.push(cache.read_async(source, offset, len).await?);
        }
        return Ok(out);
    }

    let plan = crate::io::plan(ranges, io);
    let merged: Vec<(u64, usize)> = plan.iter().map(|r| (r.offset, r.len)).collect();
    let fetched = source.read_ranges(&merged).await?;
    crate::io::scatter(&plan, fetched, ranges.len())
}

/// Copy one decoded chunk's intersection with the selection into `out`.
///
/// The copy is pure arithmetic over already-fetched bytes, so it does not care
/// how they arrived. Only the asynchronous engine calls it: the synchronous one
/// decodes and copies in one pass, because it holds the bytes already.
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
#[allow(clippy::too_many_arguments)]
fn copy_chunk(
    out: &mut [u8],
    decoded: &[u8],
    record: &crate::btree1::ChunkRecord,
    chunk_shape: &[u64],
    chunk_strides: &[u64],
    slab: &Hyperslab,
    dst_strides: &[u64],
    element_size: usize,
    rank: usize,
) {
    let mut lo = vec![0u64; rank];
    let mut hi = vec![0u64; rank];
    for axis in 0..rank {
        lo[axis] = record.offset[axis].max(slab.start[axis]);
        hi[axis] = record.offset[axis]
            .saturating_add(chunk_shape[axis])
            .min(slab.start[axis] + slab.count[axis]);
    }

    let run = hi[rank - 1] - lo[rank - 1];
    let outer: u64 = saturating_product((0..rank - 1).map(|a| hi[a] - lo[a]));
    let mut index = vec![0u64; rank.saturating_sub(1)];

    for _ in 0..outer.max(if rank == 1 { 1 } else { 0 }) {
        let mut src = 0u64;
        let mut dst = 0u64;
        for axis in 0..rank - 1 {
            let coord = lo[axis] + index[axis];
            src += (coord - record.offset[axis]) * chunk_strides[axis];
            dst += (coord - slab.start[axis]) * dst_strides[axis];
        }
        src += (lo[rank - 1] - record.offset[rank - 1]) * chunk_strides[rank - 1];
        dst += (lo[rank - 1] - slab.start[rank - 1]) * dst_strides[rank - 1];

        let sb = src as usize * element_size;
        let db = dst as usize * element_size;
        let len = run as usize * element_size;
        out[db..db + len].copy_from_slice(&decoded[sb..sb + len]);

        for axis in (0..rank.saturating_sub(1)).rev() {
            index[axis] += 1;
            if index[axis] < hi[axis] - lo[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        assert_eq!(strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(strides(&[5]), vec![1]);
        assert_eq!(strides(&[]), Vec::<u64>::new());
    }

    #[test]
    fn a_full_selection_covers_the_shape() {
        let s = Hyperslab::all(&[3, 4]);
        assert_eq!(s.start, vec![0, 0]);
        assert_eq!(s.count, vec![3, 4]);
        assert_eq!(s.element_count(), 12);
    }

    #[test]
    fn a_selection_past_the_edge_is_rejected() {
        let err = Hyperslab::new(vec![2], vec![5], &[6]).unwrap_err();
        assert!(matches!(err, Error::BadRequest(_)), "got {err:?}");
    }

    #[test]
    fn a_selection_of_the_wrong_rank_is_rejected() {
        assert!(Hyperslab::new(vec![0], vec![1], &[4, 4]).is_err());
    }

    #[test]
    fn runs_cover_a_two_dimensional_selection_exactly() {
        // A 4x5 source, selecting rows 1..3 and columns 1..4.
        let shape = [4u64, 5];
        let slab = Hyperslab {
            start: vec![1, 1],
            count: vec![2, 3],
        };
        let mut seen = Vec::new();
        for_each_run(&shape, &slab, |s, d, run| {
            seen.push((s, d, run));
            Ok(())
        })
        .unwrap();

        // Row 1 starts at element 6, row 2 at element 11. Destination rows are
        // 3 elements apart.
        assert_eq!(seen, vec![(6, 0, 3), (11, 3, 3)]);
    }

    #[test]
    fn runs_cover_a_one_dimensional_selection() {
        let mut seen = Vec::new();
        for_each_run(
            &[10],
            &Hyperslab {
                start: vec![3],
                count: vec![4],
            },
            |s, d, run| {
                seen.push((s, d, run));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(seen, vec![(3, 0, 4)]);
    }

    #[test]
    fn copies_a_sub_block_out_of_a_dense_image() {
        // 3x4 of u8 values 0..12.
        let src: Vec<u8> = (0..12).collect();
        let slab = Hyperslab {
            start: vec![1, 1],
            count: vec![2, 2],
        };
        let mut out = vec![0u8; 4];
        copy_from_dense(&mut out, &src, &[3, 4], &slab, 1).unwrap();
        assert_eq!(out, vec![5, 6, 9, 10]);
    }

    #[test]
    fn byte_swapping_reverses_each_element() {
        let mut buf = vec![0x01, 0x02, 0x03, 0x04, 0x11, 0x12, 0x13, 0x14];
        swap_bytes_in_place(&mut buf, 4);
        assert_eq!(buf, vec![0x04, 0x03, 0x02, 0x01, 0x14, 0x13, 0x12, 0x11]);
    }

    #[test]
    fn single_byte_elements_are_never_swapped() {
        let mut buf = vec![1, 2, 3];
        swap_bytes_in_place(&mut buf, 1);
        assert_eq!(buf, vec![1, 2, 3]);
    }

    /// A chunk record at `offset`, stored and non-empty.
    ///
    /// The address is where the bytes live, which none of these tests read, so
    /// one address serves them all.
    fn record(offset: &[u64]) -> crate::btree1::ChunkRecord {
        crate::btree1::ChunkRecord {
            size: 64,
            filter_mask: 0,
            offset: offset.to_vec(),
            address: 4096,
        }
    }

    fn slab(start: &[u64], count: &[u64]) -> Hyperslab {
        Hyperslab {
            start: start.to_vec(),
            count: count.to_vec(),
        }
    }

    #[test]
    fn one_chunk_over_the_whole_variable_covers_every_selection() {
        let chunks = [record(&[0, 0])];
        let shape = [514, 1938];
        assert!(chunks_cover(
            &chunks,
            &shape,
            &slab(&[0, 0], &[514, 1938]),
            2
        ));
        assert!(chunks_cover(&chunks, &shape, &slab(&[100, 7], &[9, 11]), 2));
    }

    #[test]
    fn a_selection_reaching_an_absent_chunk_is_not_covered() {
        // A 2x2 grid of 4x4 chunks with the bottom-right one never written.
        let chunks = [record(&[0, 0]), record(&[0, 4]), record(&[4, 0])];
        let shape = [4, 4];
        assert!(chunks_cover(&chunks, &shape, &slab(&[0, 0], &[4, 8]), 2));
        assert!(!chunks_cover(&chunks, &shape, &slab(&[0, 0], &[8, 8]), 2));
        assert!(!chunks_cover(&chunks, &shape, &slab(&[5, 5], &[2, 2]), 2));
    }

    #[test]
    fn a_chunk_with_no_stored_bytes_covers_nothing() {
        let mut empty = record(&[0, 0]);
        empty.size = 0;
        assert!(!chunks_cover(&[empty], &[4, 4], &slab(&[0, 0], &[4, 4]), 2));
    }

    /// A damaged index can name one chunk twice. Counting how much area the
    /// records add up to would then hide a chunk that is missing, and the read
    /// would return zeros where it owes the fill value.
    #[test]
    fn one_chunk_named_twice_does_not_stand_in_for_a_missing_one() {
        let chunks = [record(&[0, 0]), record(&[0, 0]), record(&[0, 4])];
        assert!(!chunks_cover(&chunks, &[4, 4], &slab(&[0, 0], &[8, 8]), 2));
    }

    #[test]
    fn a_chunk_off_the_grid_covers_nothing() {
        // An offset that is not a multiple of the chunk shape is not a cell.
        let chunks = [record(&[1, 0])];
        assert!(!chunks_cover(&chunks, &[4, 4], &slab(&[0, 0], &[4, 4]), 2));
    }

    /// A damaged chunk index can put a chunk anywhere in the address space and
    /// give it any shape. Adding the two must not overflow, in any of the three
    /// places a read takes a chunk's far edge.
    #[test]
    fn a_chunk_reaching_past_the_address_space_does_not_overflow() {
        let at_zero = record(&[0]);
        assert!(intersects(&at_zero, &[u64::MAX], &slab(&[0], &[4]), 1));

        // Far past the selection, and its far edge wraps. It intersects
        // nothing, and it must say so rather than panic.
        let far = record(&[u64::MAX - 1]);
        assert!(!intersects(&far, &[u64::MAX], &slab(&[0], &[4]), 1));
    }

    #[cfg(feature = "async")]
    #[test]
    fn a_chunk_reaching_past_the_address_space_copies_only_the_selection() {
        // `copy_chunk` clips the same edge, so it must clip it the same way.
        let record = record(&[0]);
        let mut out = vec![0u8; 4];
        let decoded: Vec<u8> = vec![7, 8, 9, 10];
        copy_chunk(
            &mut out,
            &decoded,
            &record,
            &[u64::MAX],
            &[1],
            &slab(&[0], &[4]),
            &[1],
            1,
            1,
        );
        assert_eq!(out, decoded);
    }

    /// A shape whose product runs past `u64` must not wrap into a small,
    /// believable number, and must not panic. It says "more than anything can
    /// hold" instead, and the byte count built from it then refuses the read.
    #[test]
    fn a_shape_that_multiplies_past_u64_saturates() {
        assert_eq!(element_count(&[u64::MAX, u64::MAX]), u64::MAX);
        assert_eq!(element_count(&[1 << 40, 1 << 40, 1 << 40]), u64::MAX);
        assert_eq!(element_count(&[3, 4, 5]), 60);
        assert_eq!(element_count(&[]), 1);
    }

    /// A zero anywhere in the shape means no elements, even next to a
    /// dimension that saturates on its own.
    #[test]
    fn a_zero_dimension_still_gives_an_empty_shape() {
        assert_eq!(element_count(&[u64::MAX, u64::MAX, 0]), 0);
        assert_eq!(element_count(&[0, u64::MAX]), 0);
    }

    #[test]
    fn a_selection_that_multiplies_past_u64_is_refused_not_wrapped() {
        let slab = slab(&[0, 0], &[u64::MAX, u64::MAX]);
        assert_eq!(slab.element_count(), u64::MAX);
        // Which is what `read_hyperslab` turns into a refusal.
        assert!(slab.element_count().checked_mul(4).is_none());
    }

    #[test]
    fn a_chunk_too_large_to_hold_is_reported_not_wrapped() {
        let err = decoded_chunk_len(&[u64::MAX, 8], 4, "/x").unwrap_err();
        assert!(matches!(err, Error::Malformed(_)), "got {err:?}");
        assert_eq!(decoded_chunk_len(&[4, 4], 4, "/x").unwrap(), 64);
    }

    #[test]
    fn an_empty_selection_needs_no_fill() {
        assert!(chunks_cover(&[], &[4, 4], &slab(&[0, 0], &[0, 4]), 2));
    }

    #[test]
    fn integers_sign_extend_from_their_stored_width() {
        // -2 stored in two bytes.
        let bytes = (-2i16).to_ne_bytes();
        assert_eq!(read_integer(&bytes, true).unwrap(), -2);
        // The same bits read as unsigned.
        assert_eq!(read_integer(&bytes, false).unwrap(), 0xFFFE);
    }
}

// ─── stored chunks ─────────────────────────────────────────────────────────

/// One stored chunk of a dataset.
///
/// A chunked variable is stored as independent compressed blocks. Reading them
/// separately is the natural unit of parallel work: each one is its own byte
/// range with its own filter pipeline, so nothing is shared between them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Element offset of the chunk's origin within the variable.
    pub offset: Vec<u64>,
    /// Shape of the region this chunk actually contributes.
    ///
    /// Chunks at the edge of a variable are stored at full size but hang past
    /// the end. This shape is already clipped, so reading a chunk never returns
    /// padding.
    pub shape: Vec<u64>,
    /// Size of the chunk on disk, after compression.
    pub stored_size: u64,
}

impl Chunk {
    /// Number of elements this chunk contributes.
    pub fn element_count(&self) -> u64 {
        element_count(&self.shape)
    }

    /// The selection this chunk covers.
    pub fn selection(&self) -> Hyperslab {
        Hyperslab {
            start: self.offset.clone(),
            count: self.shape.clone(),
        }
    }
}

/// Every stored chunk of a dataset, clipped to its bounds.
///
/// The chunk index must be resolved already. A dataset that is not chunked
/// reports one chunk covering everything, so a caller uses one loop either way.
/// Both engines use this.
pub fn chunks_of(dataset: &DatasetIndex) -> Result<Vec<Chunk>> {
    let shape = &dataset.shape;
    let chunk_dims = match &dataset.layout {
        crate::message::Layout::Chunked { chunk_dims, .. } => {
            Some(chunk_dims.iter().map(|&d| d as u64).collect::<Vec<u64>>())
        }
        _ => None,
    };

    let (Some(chunk_dims), Some(records)) = (chunk_dims, dataset.resolved_chunks()) else {
        return Ok(vec![Chunk {
            offset: vec![0; shape.len()],
            shape: shape.clone(),
            stored_size: dataset
                .element_count()
                .saturating_mul(dataset.element_size() as u64),
        }]);
    };

    Ok(records
        .iter()
        .filter_map(|record| {
            // Clip the stored chunk to the variable's bounds. An edge chunk is
            // stored full size and hangs past the end.
            let mut clipped = Vec::with_capacity(shape.len());
            for axis in 0..shape.len() {
                let origin = *record.offset.get(axis)?;
                let full = *chunk_dims.get(axis)?;
                let dim = *shape.get(axis)?;
                if origin >= dim {
                    return None;
                }
                clipped.push(full.min(dim - origin));
            }
            Some(Chunk {
                offset: record.offset.clone(),
                shape: clipped,
                stored_size: record.size as u64,
            })
        })
        .collect())
}
