//! Datasets with contiguous or chunked storage for the writer.

use std::collections::BTreeMap;
use std::ops::Range;

use super::alloc::Allocator;
use super::btree::write_chunk_index;
use super::filter::{self, encode_filter_pipeline, Filter};
use super::header::message;
use super::message::{
    encode_chunked_fill_value, encode_chunked_layout, encode_contiguous_layout, encode_dataspace,
    encode_datatype, encode_fill_value, numeric_datatype, vlen_string_datatype,
};
use super::sink::ByteSink;
use super::{defect, to_u64};
use crate::btree1::ChunkRecord;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::filters::decode_chunk;
use crate::index::DatasetIndex;
use crate::message::filter::{id, FilterPipeline};
use crate::message::{
    ByteOrder, Dataspace, DataspaceKind, Datatype, DatatypeClass, FillValue, VlenKind,
};
use crate::message::{ChunkIndex, Layout};
use crate::objheader::{HeaderMessage, MessageType};
use crate::read::{swap_bytes_in_place, Hyperslab};
use crate::superblock::{Superblock, DEFAULT_ISTORE_K};

/// Largest block of fill bytes or converted values in one sink write.
///
/// The block sets the memory of one fill pass or one typed write. A smaller
/// block gives more sink calls. The unit tests count the writes of 1 MiB.
const BLOCK_BYTES: usize = 1 << 20;

/// The datatype, shape and fill value of a new dataset.
///
/// The setters accept each value. [`crate::write::Hdf5Writer::add_dataset`]
/// checks the full spec and returns an error for a wrong combination.
///
/// # Examples
///
/// ```
/// use oxcdf_hdf5::message::ByteOrder;
/// use oxcdf_hdf5::write::{DatasetSpec, Filter, Hdf5Writer, MemorySink, WriteOptions};
/// use oxcdf_hdf5::{DType, Hyperslab};
///
/// let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default())?;
/// let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[0, 3])?
///     .chunked(&[10, 3])
///     .with_max_shape(&[None, Some(3)])
///     .with_fill_value((-1i32).to_le_bytes().to_vec())
///     .with_filters(&[Filter::Shuffle, Filter::Deflate(4)]);
/// let rows = w.add_dataset(w.root(), "rows", spec)?;
/// w.resize(rows, &[2, 3])?;
/// w.write(rows, &Hyperslab::all(&[2, 3]), &[1i32, 2, 3, 4, 5, 6])?;
/// w.close()?;
/// # Ok::<(), oxcdf_hdf5::Error>(())
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DatasetSpec {
    /// Type of one element.
    pub datatype: Datatype,
    /// Size of each axis. An empty shape gives a scalar dataset.
    pub shape: Vec<u64>,
    /// One element in the dataset byte order. `None` reads as zeros.
    pub fill_value: Option<Vec<u8>>,
    /// Size of one chunk on each axis. `None` gives contiguous storage.
    pub chunk_shape: Option<Vec<u64>>,
    /// Largest size of each axis. `None` for one axis is unlimited.
    ///
    /// `None` for the field fixes each axis at its size in `shape`.
    pub max_shape: Option<Vec<Option<u64>>>,
    /// The filters of each chunk, in the order of the write. Chunked storage only.
    pub filters: Vec<Filter>,
}

impl DatasetSpec {
    /// A dataset of `datatype` and `shape` with no fill value.
    pub fn new(datatype: Datatype, shape: &[u64]) -> Self {
        Self {
            datatype,
            shape: shape.to_vec(),
            fill_value: None,
            chunk_shape: None,
            max_shape: None,
            filters: Vec::new(),
        }
    }

    /// Apply `filters` in their order to each chunk. The dataset must be chunked.
    ///
    /// The writer compresses a chunk when one write covers it in full. A
    /// chunk that gets a part of its elements stays in memory until
    /// [`crate::write::Hdf5Writer::flush_chunks`] or the close. After that,
    /// the chunk takes only a write that covers it in full.
    pub fn with_filters(mut self, filters: &[Filter]) -> Self {
        self.filters = filters.to_vec();
        self
    }

    /// Store the dataset in chunks of `chunk_shape` elements.
    ///
    /// A chunk gets file space at the first write to one of its elements.
    pub fn chunked(mut self, chunk_shape: &[u64]) -> Self {
        self.chunk_shape = Some(chunk_shape.to_vec());
        self
    }

    /// Set the largest size of each axis. `None` for one axis is unlimited.
    ///
    /// Only a chunked dataset can grow, with [`crate::write::Hdf5Writer::resize`].
    pub fn with_max_shape(mut self, max_shape: &[Option<u64>]) -> Self {
        self.max_shape = Some(max_shape.to_vec());
        self
    }

    /// A numeric dataset with no fill value.
    ///
    /// # Errors
    ///
    /// A type with no numeric HDF5 datatype is [`Error::Unsupported`].
    pub fn numeric(dtype: DType, order: ByteOrder, shape: &[u64]) -> Result<Self> {
        Ok(Self::new(numeric_datatype(dtype, order)?, shape))
    }

    /// A dataset of variable-length UTF-8 strings, the netCDF string type.
    ///
    /// An element with no write reads as an empty string.
    pub fn strings(shape: &[u64]) -> Self {
        Self::new(vlen_string_datatype(), shape)
    }

    /// Set the fill value: one element in the dataset byte order.
    ///
    /// A variable-length dataset takes no fill value.
    pub fn with_fill_value(mut self, bytes: Vec<u8>) -> Self {
        self.fill_value = Some(bytes);
        self
    }
}

/// A dataset in the writer, with its fixed messages encoded at creation.
#[derive(Debug)]
pub(super) struct Dataset {
    datatype: Datatype,
    shape: Vec<u64>,
    fill_value: Option<Vec<u8>>,
    /// Size of one element in bytes, above zero.
    element_size: u64,
    /// Size of the full extent in bytes.
    extent_bytes: u64,
    /// Address of the extent, set by the first write that has bytes.
    address: Option<u64>,
    /// Address of the reserved extent while its fill pass is not complete.
    reserved: Option<u64>,
    /// True after a resize.
    touched: bool,
    /// Largest size of each axis, with `u64::MAX` for unlimited.
    max_dims: Option<Vec<u64>>,
    /// The chunks. `None` for contiguous storage.
    chunks: Option<Chunks>,
    dataspace_body: Vec<u8>,
    datatype_body: Vec<u8>,
    fill_body: Vec<u8>,
    /// The filter pipeline message body. Empty for a dataset with no filters.
    pipeline_body: Vec<u8>,
}

/// The part of a selection that lies in one chunk.
#[derive(Debug)]
struct Part {
    /// The position of the first element of the chunk.
    origin: Vec<u64>,
    /// The first selected position in the chunk on each axis.
    low: Vec<u64>,
    /// The end of the selected positions in the chunk on each axis.
    high: Vec<u64>,
    /// Whether the selection covers the full chunk.
    full: bool,
}

impl Part {
    /// The part of `slab` in the chunk at `origin` with `dims` elements on each axis.
    fn new(origin: Vec<u64>, dims: &[u64], slab: &Hyperslab) -> Self {
        let low: Vec<u64> = origin
            .iter()
            .zip(&slab.start)
            .map(|(&o, &s)| o.max(s))
            .collect();
        let high: Vec<u64> = (0..dims.len())
            .map(|a| {
                let chunk_end = origin[a].saturating_add(dims[a]);
                chunk_end.min(slab.start[a] + slab.count[a])
            })
            .collect();
        let full = (0..dims.len()).all(|a| high[a] - low[a] == dims[a]);
        Self {
            origin,
            low,
            high,
            full,
        }
    }
}

/// The first element of each chunk that `slab` touches, in row-major order.
///
/// `dims` is the size of one chunk on each axis. An empty selection touches no chunk.
fn chunk_origins<'a>(dims: &'a [u64], slab: &Hyperslab) -> impl Iterator<Item = Vec<u64>> + 'a {
    // The index of the first chunk and the end index on each axis.
    let first: Vec<u64> = slab.start.iter().zip(dims).map(|(&s, &d)| s / d).collect();
    let end: Vec<u64> = (0..dims.len())
        .map(|a| (slab.start[a] + slab.count[a]).div_ceil(dims[a]))
        .collect();
    let mut next = (!slab.count.contains(&0)).then(|| first.clone());
    std::iter::from_fn(move || {
        let chunk = next.take()?;
        let origin = chunk.iter().zip(dims).map(|(&c, &d)| c * d).collect();
        let mut after = chunk;
        if advance(&mut after, &first, &end) {
            next = Some(after);
        }
        Some(origin)
    })
}

/// What maps the rows of one [`Part`] to the caller data and to the chunk.
struct Rows<'a> {
    slab_start: &'a [u64],
    /// Element strides of each axis in the selection.
    slab_strides: &'a [u64],
    /// Element strides of each axis in one chunk.
    chunk_strides: &'a [u64],
    part: &'a Part,
    element_size: u64,
}

impl Rows<'_> {
    /// Call `emit` for each row of the part. The chunk starts at `base` in `target`.
    fn emit(
        &self,
        target: &mut dyn ByteSink,
        base: u64,
        emit: &mut impl FnMut(&mut dyn ByteSink, u64, Range<usize>) -> Result<()>,
    ) -> Result<()> {
        let Part {
            origin, low, high, ..
        } = self.part;
        let last = low.len() - 1;
        // One row runs along the last axis. `at` is its first element.
        let row_bytes = to_usize((high[last] - low[last]) * self.element_size)?;
        let mut at = low.clone();
        loop {
            let mut source = 0u64;
            let mut target_element = 0u64;
            for a in 0..=last {
                source = (at[a] - self.slab_start[a])
                    .checked_mul(self.slab_strides[a])
                    .and_then(|n| n.checked_add(source))
                    .ok_or_else(overflow)?;
                target_element += (at[a] - origin[a]) * self.chunk_strides[a];
            }
            let source = source
                .checked_mul(self.element_size)
                .ok_or_else(overflow)
                .and_then(to_usize)?;
            let range = source..source.checked_add(row_bytes).ok_or_else(overflow)?;
            let to = offset(base, target_element * self.element_size)?;
            emit(&mut *target, to, range)?;
            // The next row, on the axes before the last one.
            if !advance(&mut at[..last], low, high) {
                return Ok(());
            }
        }
    }
}

/// One chunk in memory that takes writes at offsets from its start.
///
/// A write past the end of the chunk is a bug and panics.
#[derive(Debug)]
struct BufferSink<'a>(&'a mut [u8]);

impl ByteSink for BufferSink<'_> {
    fn write_all_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        let start = to_usize(offset)?;
        let len = self.0.len();
        let target = start
            .checked_add(data.len())
            .and_then(|end| self.0.get_mut(start..end))
            .ok_or_else(|| {
                defect(format!(
                    "a row of {} bytes at {start} passes a chunk of {len} bytes",
                    data.len()
                ))
            })?;
        target.copy_from_slice(data);
        Ok(())
    }

    fn set_len(&mut self, _len: u64) -> Result<()> {
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        Ok(())
    }
}

/// The chunks of a dataset.
#[derive(Debug)]
struct Chunks {
    /// Size of one chunk on each axis, in elements.
    dims: Vec<u64>,
    /// The same sizes for the layout message.
    dims32: Vec<u32>,
    /// Size of one chunk in bytes.
    bytes: u32,
    /// The filters of each chunk, in the order of the write.
    filters: Vec<Filter>,
    /// The file space of each chunk, by the position of its first element.
    spaces: BTreeMap<Vec<u64>, ChunkSpace>,
    /// The chunks with filters that are in memory and not in the file.
    buffers: BTreeMap<Vec<u64>, Vec<u8>>,
    /// Address of the root node of the chunk index.
    ///
    /// A dataset of a file can start with the index of the file. The close
    /// sets it when `stale` is true.
    index: Option<u64>,
    /// True when `index` does not give each chunk of `spaces` that is ready.
    stale: bool,
}

/// The file space of one chunk.
#[derive(Debug, Clone, Copy)]
struct ChunkSpace {
    address: u64,
    /// Size of the chunk in the file, after its filters.
    size: u32,
    /// False while the fill pass of the chunk is not complete.
    ready: bool,
}

impl Chunks {
    /// The part of `slab` in each chunk that it touches, in row-major order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when `slab` covers a part of a chunk
    /// that is in the file with its filters.
    fn parts(&self, slab: &Hyperslab, readable: bool) -> Result<Vec<Part>> {
        let parts: Vec<Part> = chunk_origins(&self.dims, slab)
            .map(|origin| Part::new(origin, &self.dims, slab))
            .collect();
        for part in &parts {
            // The stored bytes of a filtered chunk cannot take a part of a chunk.
            let stored = !self.filters.is_empty()
                && self.spaces.contains_key(&part.origin)
                && !self.buffers.contains_key(&part.origin);
            if stored && !part.full && !readable {
                return Err(Error::unsupported(format!(
                    "a write to a part of the chunk at {:?}, which is in the file with its filters",
                    part.origin
                )));
            }
        }
        Ok(parts)
    }

    /// Check a chunk shape against the dataset shape and its largest shape.
    fn new(
        dims: Vec<u64>,
        shape: &[u64],
        max: &[u64],
        element_size: u64,
        filters: Vec<Filter>,
    ) -> Result<Self> {
        if dims.len() != shape.len() || dims.is_empty() {
            return Err(Error::bad_request(format!(
                "a chunk shape of rank {} for a dataset of rank {}",
                dims.len(),
                shape.len()
            )));
        }
        let mut dims32 = Vec::with_capacity(dims.len());
        let mut elements = 1u64;
        for (&dim, &max) in dims.iter().zip(max) {
            // The HDF5 library rejects a chunk above the largest size of a fixed axis.
            if dim == 0 || dim > max {
                return Err(Error::bad_request(format!(
                    "a chunk size of {dim} for an axis with a largest size of {max}"
                )));
            }
            dims32.push(u32::try_from(dim).ok());
            elements = elements.saturating_mul(dim);
        }
        let bytes = elements
            .checked_mul(element_size)
            .and_then(|n| u32::try_from(n).ok());
        match (dims32.into_iter().collect::<Option<Vec<u32>>>(), bytes) {
            (Some(dims32), Some(bytes)) => Ok(Self {
                dims,
                dims32,
                bytes,
                filters,
                spaces: BTreeMap::new(),
                buffers: BTreeMap::new(),
                index: None,
                stale: false,
            }),
            _ => Err(Error::bad_request(
                "a chunk needs 4 GiB or more, above the HDF5 limit",
            )),
        }
    }

    /// Apply the filters to `chunk` and write it to new file space.
    ///
    /// The space of an earlier version of the chunk stays unused in the file.
    fn store<S: ByteSink>(
        &mut self,
        sink: &mut S,
        alloc: &mut Allocator,
        origin: &[u64],
        chunk: &[u8],
        element_size: u64,
    ) -> Result<()> {
        let stored = filter::apply(&self.filters, to_usize(element_size)?, chunk)?;
        let size = u32::try_from(stored.len())
            .map_err(|_| Error::unsupported("a chunk needs 4 GiB or more after its filters"))?;
        let address = alloc.alloc_data(u64::from(size))?;
        sink.write_all_at(address, &stored)?;
        self.set_space(
            origin.to_vec(),
            ChunkSpace {
                address,
                size,
                ready: true,
            },
        );
        Ok(())
    }

    /// Set the file space of the chunk at `origin`. The chunk index must change.
    fn set_space(&mut self, origin: Vec<u64>, space: ChunkSpace) {
        self.spaces.insert(origin, space);
        self.stale = true;
    }

    /// Write each chunk in memory to the file. A chunk stays in memory after an error.
    fn flush<S: ByteSink>(
        &mut self,
        sink: &mut S,
        alloc: &mut Allocator,
        element_size: u64,
    ) -> Result<()> {
        while let Some((origin, chunk)) = self.buffers.pop_first() {
            if let Err(e) = self.store(sink, alloc, &origin, &chunk, element_size) {
                self.buffers.insert(origin, chunk);
                return Err(e);
            }
        }
        Ok(())
    }
}

/// The dataspace message body for `shape` and its largest shape.
fn dataspace_body(shape: &[u64], max_dims: Option<&[u64]>) -> Result<Vec<u8>> {
    let kind = if shape.is_empty() {
        DataspaceKind::Scalar
    } else {
        DataspaceKind::Simple
    };
    encode_dataspace(&Dataspace {
        kind,
        dims: shape.to_vec(),
        max_dims: max_dims.map(<[u64]>::to_vec),
    })
}

/// The size in bytes of all elements of `shape`.
fn extent_bytes_of(shape: &[u64], element_size: u64) -> Result<u64> {
    // A zero axis empties the dataset, whatever the size of the other axes.
    let elements = if shape.contains(&0) {
        Some(0)
    } else {
        shape.iter().try_fold(1u64, |n, &dim| n.checked_mul(dim))
    };
    elements
        .and_then(|n| n.checked_mul(element_size))
        .ok_or_else(|| {
            Error::bad_request(format!(
                "shape {shape:?} with {element_size}-byte elements needs more than u64::MAX bytes"
            ))
        })
}

impl Dataset {
    /// Check `spec` and encode its fixed messages, so that close cannot fail on it.
    pub(super) fn new(spec: DatasetSpec) -> Result<Self> {
        Self::with_fill_reference(spec, None)
    }

    /// As [`Self::new`], with the fill value message of a variable-length dataset.
    ///
    /// `reference` is the heap reference that an element with no file space
    /// reads as. Elements in file space do not get it.
    pub(super) fn with_fill_reference(
        spec: DatasetSpec,
        reference: Option<[u8; 16]>,
    ) -> Result<Self> {
        let attribute_only = match &spec.datatype.class {
            DatatypeClass::VariableLength { kind, .. } => *kind != VlenKind::String,
            DatatypeClass::Compound { .. } | DatatypeClass::Reference { .. } => true,
            _ => false,
        };
        if attribute_only {
            return Err(Error::unsupported(format!(
                "a dataset of the datatype class {:?}",
                spec.datatype.class
            )));
        }
        let datatype_body = encode_datatype(&spec.datatype)?;
        let element_size = u64::from(spec.datatype.size);
        let max_dims = match &spec.max_shape {
            None => None,
            Some(max) => {
                let max: Vec<u64> = max.iter().map(|dim| dim.unwrap_or(u64::MAX)).collect();
                let fits = max.len() == spec.shape.len()
                    && max.iter().zip(&spec.shape).all(|(max, dim)| max >= dim);
                if !fits {
                    return Err(Error::bad_request(format!(
                        "a largest shape of {max:?} for a shape of {:?}",
                        spec.shape
                    )));
                }
                if spec.chunk_shape.is_none() && max != spec.shape {
                    return Err(Error::bad_request(
                        "a dataset that can grow needs chunked storage",
                    ));
                }
                Some(max)
            }
        };
        let dataspace_body = dataspace_body(&spec.shape, max_dims.as_deref())?;
        let chunks = match spec.chunk_shape {
            None => None,
            Some(dims) => {
                let max = max_dims.as_deref().unwrap_or(&spec.shape);
                let filters = spec.filters.clone();
                Some(Chunks::new(dims, &spec.shape, max, element_size, filters)?)
            }
        };
        let pipeline_body = match (&chunks, spec.filters.as_slice()) {
            (_, []) => Vec::new(),
            (Some(_), filters) => encode_filter_pipeline(filters, spec.datatype.size)?,
            (None, _) => {
                return Err(Error::bad_request(
                    "a dataset with filters needs chunked storage",
                ))
            }
        };
        let vlen = matches!(spec.datatype.class, DatatypeClass::VariableLength { .. });
        if vlen && spec.fill_value.is_some() {
            return Err(Error::bad_request(
                "a variable-length dataset takes no fill value",
            ));
        }
        if let Some(fill) = &spec.fill_value {
            if u64::try_from(fill.len()).ok() != Some(element_size) {
                return Err(Error::bad_request(format!(
                    "fill value has {} bytes, but one element has {element_size}",
                    fill.len()
                )));
            }
        }
        let fill = FillValue {
            bytes: spec.fill_value,
        };
        let message_fill = match reference.filter(|_| vlen) {
            Some(reference) => FillValue {
                bytes: Some(reference.to_vec()),
            },
            None => FillValue {
                bytes: fill.bytes.clone(),
            },
        };
        let fill_body = match chunks {
            Some(_) => encode_chunked_fill_value(&message_fill)?,
            None => encode_fill_value(&message_fill)?,
        };
        let extent_bytes = extent_bytes_of(&spec.shape, element_size)?;
        Ok(Self {
            datatype: spec.datatype,
            shape: spec.shape,
            fill_value: fill.bytes,
            element_size,
            extent_bytes,
            address: None,
            reserved: None,
            touched: false,
            max_dims,
            chunks,
            dataspace_body,
            datatype_body,
            fill_body,
            pipeline_body,
        })
    }

    /// Make the dataset of `index`, with its values where they are in the file.
    ///
    /// `records` holds each chunk of a chunked dataset. The dataset keeps
    /// its definition, as a dataset with values does. A version 1 B-tree
    /// stays the chunk index while no chunk gets new file space. The close
    /// writes a new index for each other chunked dataset. `superblock` is
    /// the superblock of the file.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] for compact storage, a filter that
    /// the writer does not have, a chunk that skips a filter, or a datatype
    /// that the writer cannot encode.
    pub(super) fn existing(
        index: &DatasetIndex,
        records: &[ChunkRecord],
        superblock: &Superblock,
    ) -> Result<Self> {
        let mut spec = DatasetSpec::new(index.datatype.clone(), &index.shape);
        let address = match &index.layout {
            Layout::Contiguous { address, .. } => *address,
            Layout::Chunked { chunk_dims, .. } => {
                let dims: Vec<u64> = chunk_dims.iter().map(|&dim| u64::from(dim)).collect();
                spec = spec
                    .chunked(&dims)
                    .with_filters(&filters_of(&index.pipeline)?);
                None
            }
            Layout::Compact { .. } => {
                return Err(Error::unsupported(format!(
                    "an edit of the dataset {} with compact storage",
                    index.path
                )))
            }
        };
        if let Some(max) = &index.max_shape {
            let unlimited = |&max: &u64| (max != u64::MAX).then_some(max);
            spec = spec.with_max_shape(&max.iter().map(unlimited).collect::<Vec<_>>());
        }
        let vlen = matches!(index.datatype.class, DatatypeClass::VariableLength { .. });
        let reference = match (&index.fill_value.bytes, vlen) {
            (Some(bytes), true) => <[u8; 16]>::try_from(bytes.as_slice()).ok(),
            (Some(bytes), false) => {
                spec = spec.with_fill_value(bytes.clone());
                None
            }
            (None, _) => None,
        };
        let mut dataset = Self::with_fill_reference(spec, reference)?;
        dataset.touched = true;
        dataset.address = address;
        if let Some(chunks) = &mut dataset.chunks {
            for record in records {
                let origin = record.offset.get(..chunks.dims.len());
                let (Some(origin), 0) = (origin, record.filter_mask) else {
                    return Err(Error::unsupported(format!(
                        "an edit of the dataset {} with a chunk that skips a filter",
                        index.path
                    )));
                };
                let space = ChunkSpace {
                    address: record.address,
                    size: record.size,
                    ready: true,
                };
                chunks.spaces.insert(origin.to_vec(), space);
            }
            chunks.index = kept_index(&index.layout, superblock);
            chunks.stale = chunks.index.is_none() && !records.is_empty();
        }
        Ok(dataset)
    }

    /// The type of one element.
    pub(super) fn datatype(&self) -> &Datatype {
        &self.datatype
    }

    /// The number of axes.
    pub(super) fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Whether a write or a resize changed the dataset after its creation.
    pub(super) fn is_touched(&self) -> bool {
        self.touched
            || self.reserved.is_some()
            || self
                .chunks
                .as_ref()
                .is_some_and(|c| !c.spaces.is_empty() || !c.buffers.is_empty())
    }

    /// Whether each element is a reference to the global heap.
    pub(super) fn is_vlen(&self) -> bool {
        matches!(self.datatype.class, DatatypeClass::VariableLength { .. })
    }

    /// Check that `slab` lies in the shape and selects `count` elements.
    pub(super) fn check_selection(&self, slab: &Hyperslab, count: usize) -> Result<()> {
        slab.validate(&self.shape)?;
        if u64::try_from(count).ok() != Some(slab.element_count()) {
            return Err(Error::bad_request(format!(
                "{count} values for a selection of {} elements",
                slab.element_count()
            )));
        }
        Ok(())
    }

    /// Set the shape of a chunked dataset. No axis can shrink.
    ///
    /// An error changes nothing.
    pub(super) fn resize(&mut self, shape: &[u64]) -> Result<()> {
        if self.chunks.is_none() {
            return Err(Error::bad_request(
                "only a chunked dataset can change its shape",
            ));
        }
        let max = self.max_dims.as_deref().unwrap_or(&self.shape);
        let fits = shape.len() == self.shape.len()
            && shape.iter().zip(&self.shape).all(|(new, old)| new >= old)
            && shape.iter().zip(max).all(|(new, max)| new <= max);
        if !fits {
            return Err(Error::bad_request(format!(
                "shape {shape:?} is not from {:?} up to {max:?} on each axis",
                self.shape
            )));
        }
        let extent_bytes = extent_bytes_of(shape, self.element_size)?;
        self.dataspace_body = dataspace_body(shape, self.max_dims.as_deref())?;
        self.extent_bytes = extent_bytes;
        self.shape = shape.to_vec();
        self.touched = true;
        Ok(())
    }

    /// Write the chunk index of a chunked dataset. The layout message needs its address.
    pub(super) fn write_index<S: ByteSink>(
        &mut self,
        sink: &mut S,
        alloc: &mut Allocator,
    ) -> Result<()> {
        let element_size = self.element_size;
        let Some(chunks) = &mut self.chunks else {
            return Ok(());
        };
        chunks.flush(sink, alloc, element_size)?;
        if !chunks.stale {
            return Ok(());
        }
        // A chunk with no complete fill pass holds no values, so it reads as fill.
        let ready: Vec<(Vec<u64>, u64, u32)> = chunks
            .spaces
            .iter()
            .filter(|(_, space)| space.ready)
            .map(|(origin, space)| (origin.clone(), space.address, space.size))
            .collect();
        chunks.index = write_chunk_index(sink, alloc, &chunks.dims, &ready)?;
        chunks.stale = false;
        Ok(())
    }

    /// Write each chunk that is in memory to the file, with its filters.
    pub(super) fn flush_chunks<S: ByteSink>(
        &mut self,
        sink: &mut S,
        alloc: &mut Allocator,
    ) -> Result<()> {
        let element_size = self.element_size;
        match &mut self.chunks {
            Some(chunks) => chunks.flush(sink, alloc, element_size),
            None => Ok(()),
        }
    }

    /// The layout message body. Its length does not depend on an address.
    fn layout_body(&self) -> Vec<u8> {
        match &self.chunks {
            Some(chunks) => encode_chunked_layout(chunks.index, &chunks.dims32, self.datatype.size),
            None => encode_contiguous_layout(self.address, self.extent_bytes),
        }
    }

    /// The body lengths of the messages that [`Self::take_messages`] returns.
    pub(super) fn message_lens(&self) -> Vec<usize> {
        let mut lens = vec![
            self.dataspace_body.len(),
            self.datatype_body.len(),
            self.fill_body.len(),
            self.layout_body().len(),
        ];
        if !self.pipeline_body.is_empty() {
            lens.push(self.pipeline_body.len());
        }
        lens
    }

    /// Move out the dataspace, datatype, fill value and layout messages, in that order.
    ///
    /// A dataset with filters has the filter pipeline message after them.
    pub(super) fn take_messages(&mut self) -> Vec<HeaderMessage> {
        let mut messages = vec![
            message(
                MessageType::Dataspace,
                std::mem::take(&mut self.dataspace_body),
            ),
            message(
                MessageType::Datatype,
                std::mem::take(&mut self.datatype_body),
            ),
            message(MessageType::FillValue, std::mem::take(&mut self.fill_body)),
            message(MessageType::DataLayout, self.layout_body()),
        ];
        if !self.pipeline_body.is_empty() {
            let body = std::mem::take(&mut self.pipeline_body);
            messages.push(message(MessageType::FilterPipeline, body));
        }
        messages
    }

    /// Check that no chunk refuses a write to `slab`. The slab can pass the shape.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when `slab` covers a part of a chunk
    /// that is in the file with its filters.
    pub(super) fn check_chunks(&self, slab: &Hyperslab, readable: bool) -> Result<()> {
        match &self.chunks {
            Some(chunks) if slab.start.len() == chunks.dims.len() => {
                chunks.parts(slab, readable).map(drop)
            }
            _ => Ok(()),
        }
    }

    /// The first element of each chunk inside the shape that holds no values.
    ///
    /// A chunk with no complete fill pass holds no values. Empty for a contiguous dataset.
    pub(super) fn missing_chunks(&self) -> Vec<Vec<u64>> {
        let Some(chunks) = &self.chunks else {
            return Vec::new();
        };
        chunk_origins(&chunks.dims, &Hyperslab::all(&self.shape))
            .filter(|origin| {
                let stored = chunks.spaces.get(origin).is_some_and(|space| space.ready);
                !stored && !chunks.buffers.contains_key(origin)
            })
            .collect()
    }

    /// The number of elements in one chunk. Zero for a contiguous dataset.
    pub(super) fn chunk_elements(&self) -> u64 {
        let bytes = self.chunks.as_ref().map_or(0, |c| u64::from(c.bytes));
        bytes / self.element_size
    }

    /// Write the full chunk at `origin` with `bytes`, which hold each of its elements.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] for a contiguous dataset, or when `bytes`
    /// does not hold one chunk. Only a defect in the writer causes that.
    pub(super) fn write_chunk<S: ByteSink>(
        &mut self,
        sink: &mut S,
        alloc: &mut Allocator,
        origin: &[u64],
        bytes: &[u8],
    ) -> Result<()> {
        let element_size = self.element_size;
        let Some(chunks) = self.chunks.as_mut() else {
            return Err(defect("a chunk write goes to a contiguous dataset"));
        };
        if to_u64(bytes.len())? != u64::from(chunks.bytes) {
            return Err(defect("a chunk write does not have the bytes of one chunk"));
        }
        chunks.store(sink, alloc, origin, bytes, element_size)
    }

    /// The number of elements that a write to `slab` gives file space with no value.
    ///
    /// These are the elements that get the fill value in that write.
    pub(super) fn fill_elements(&self, slab: &Hyperslab) -> Result<u64> {
        slab.validate(&self.shape)?;
        if slab.element_count() == 0 {
            return Ok(0);
        }
        let Some(chunks) = &self.chunks else {
            let selected = slab.element_count().checked_mul(self.element_size);
            let fills = self.address.is_none() && selected != Some(self.extent_bytes);
            return Ok(if fills {
                self.extent_bytes / self.element_size
            } else {
                0
            });
        };
        let per_chunk = u64::from(chunks.bytes) / self.element_size;
        let mut elements = 0u64;
        for origin in chunk_origins(&chunks.dims, slab) {
            let new = if chunks.filters.is_empty() {
                chunks.spaces.get(&origin).is_none_or(|space| !space.ready)
            } else {
                !chunks.buffers.contains_key(&origin) && !chunks.spaces.contains_key(&origin)
            };
            if new && !Part::new(origin, &chunks.dims, slab).full {
                elements = elements.saturating_add(per_chunk);
            }
        }
        Ok(elements)
    }

    /// Write `bytes` in the dataset byte order to the elements of `slab`.
    ///
    /// `fresh` holds one element for each element of [`Self::fill_elements`].
    /// With `fresh`, those elements get its bytes and not the fill value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Malformed`] when `fresh` holds fewer elements than the
    /// fill pass needs. Only a defect in the writer causes that.
    pub(super) fn write_raw<S: ByteSink>(
        &mut self,
        sink: &mut S,
        alloc: &mut Allocator,
        slab: &Hyperslab,
        bytes: &[u8],
        fresh: Option<&[u8]>,
    ) -> Result<()> {
        self.write_selection(sink, alloc, slab, bytes.len(), fresh, |sink, at, range| {
            let run = bytes.get(range).ok_or_else(overflow)?;
            sink.write_all_at(at, run)
        })
    }

    /// Write `values` to the elements of `slab`, converted in blocks.
    ///
    /// The caller checks that `T` has the size of one dataset element.
    pub(super) fn write_values<S: ByteSink, T: Element>(
        &mut self,
        sink: &mut S,
        alloc: &mut Allocator,
        slab: &Hyperslab,
        values: &[T],
    ) -> Result<()> {
        let size = std::mem::size_of::<T>().max(1);
        let swap = needs_swap(self.datatype.byte_order());
        let per_block = (BLOCK_BYTES / size).max(1);
        let len = std::mem::size_of_val(values);
        let mut block = Vec::with_capacity(len.min(per_block * size));
        self.write_selection(sink, alloc, slab, len, None, |sink, mut at, range| {
            let run = values
                .get(range.start / size..range.end / size)
                .ok_or_else(overflow)?;
            for part in run.chunks(per_block) {
                block.clear();
                T::extend_ne_bytes(part, &mut block);
                if swap {
                    swap_bytes_in_place(&mut block, size);
                }
                sink.write_all_at(at, &block)?;
                at = offset(at, to_u64(block.len())?)?;
            }
            Ok(())
        })
    }

    /// Check a selection of `len` bytes, reserve the extent, and call `emit` for each run.
    ///
    /// `emit` gets the file address of one run and its byte range in the caller data.
    fn write_selection<S: ByteSink>(
        &mut self,
        sink: &mut S,
        alloc: &mut Allocator,
        slab: &Hyperslab,
        len: usize,
        mut fresh: Option<&[u8]>,
        emit: impl FnMut(&mut dyn ByteSink, u64, Range<usize>) -> Result<()>,
    ) -> Result<()> {
        slab.validate(&self.shape)?;
        let selected = slab.element_count().checked_mul(self.element_size);
        if selected != u64::try_from(len).ok() {
            return Err(Error::bad_request(format!(
                "{len} bytes for a selection of {} elements of {} bytes",
                slab.element_count(),
                self.element_size
            )));
        }
        if len == 0 {
            return Ok(());
        }
        if self.chunks.is_some() {
            return self.write_chunks(sink, alloc, slab, fresh, emit);
        }
        let address = match self.address {
            Some(address) => address,
            None => {
                let address = match self.reserved {
                    Some(address) => address,
                    None => alloc.alloc_data(self.extent_bytes)?,
                };
                // A failed fill pass keeps its space for the next write.
                self.reserved = Some(address);
                // Sink bytes that get no write hold zeros, so a zero fill needs no pass.
                let fill = self
                    .fill_value
                    .as_deref()
                    .filter(|fill| fill.iter().any(|&b| b != 0));
                // A selection inside the shape with all its bytes covers the full extent.
                if selected != Some(self.extent_bytes) {
                    match (take_fresh(&mut fresh, self.extent_bytes)?, fill) {
                        (Some(block), _) => sink.write_all_at(address, block)?,
                        (None, Some(fill)) => write_fill(sink, address, self.extent_bytes, fill)?,
                        (None, None) => {}
                    }
                }
                self.address = Some(address);
                address
            }
        };
        self.write_runs(sink, address, slab, len, emit)
    }

    /// Write the part of `slab` in each chunk that it touches, with `emit` for each row.
    fn write_chunks<S: ByteSink>(
        &mut self,
        sink: &mut S,
        alloc: &mut Allocator,
        slab: &Hyperslab,
        mut fresh: Option<&[u8]>,
        mut emit: impl FnMut(&mut dyn ByteSink, u64, Range<usize>) -> Result<()>,
    ) -> Result<()> {
        let element_size = self.element_size;
        let fill = self
            .fill_value
            .as_deref()
            .filter(|fill| fill.iter().any(|&b| b != 0));
        let pipeline_body = &self.pipeline_body;
        let Some(chunks) = &mut self.chunks else {
            return Ok(());
        };
        let rank = chunks.dims.len();
        let last = rank - 1;
        // Element strides of each axis in the selection and in one chunk.
        let mut slab_strides = vec![1u64; rank];
        let mut chunk_strides = vec![1u64; rank];
        for axis in (0..last).rev() {
            slab_strides[axis] = slab_strides[axis + 1]
                .checked_mul(slab.count[axis + 1])
                .ok_or_else(overflow)?;
            chunk_strides[axis] = chunk_strides[axis + 1] * chunks.dims[axis + 1];
        }
        let parts = chunks.parts(slab, sink.can_read())?;

        // Each row of a full chunk writes over its bytes, so the buffer needs no reset.
        let mut full_buffer = Vec::new();
        for part in &parts {
            let rows = Rows {
                slab_start: &slab.start,
                slab_strides: &slab_strides,
                chunk_strides: &chunk_strides,
                part,
                element_size,
            };
            if !chunks.filters.is_empty() {
                // A chunk with filters goes to the sink in one piece.
                if let Some(buffer) = chunks.buffers.get_mut(&part.origin) {
                    rows.emit(&mut BufferSink(buffer), 0, &mut emit)?;
                    continue;
                }
                if part.full {
                    full_buffer.resize(to_usize(u64::from(chunks.bytes))?, 0);
                    rows.emit(&mut BufferSink(&mut full_buffer), 0, &mut emit)?;
                    chunks.store(sink, alloc, &part.origin, &full_buffer, element_size)?;
                    continue;
                }
                if let Some(space) = chunks.spaces.get(&part.origin).copied() {
                    // The chunk is in the file with its filters, so it comes back to memory.
                    let mut stored = vec![0; to_usize(u64::from(space.size))?];
                    sink.read_exact_at(space.address, &mut stored)?;
                    let pipeline = FilterPipeline::parse(pipeline_body)?;
                    let len = to_usize(u64::from(chunks.bytes))?;
                    let mut buffer = decode_chunk(&pipeline, 0, stored, len)?;
                    rows.emit(&mut BufferSink(&mut buffer), 0, &mut emit)?;
                    chunks.buffers.insert(part.origin.clone(), buffer);
                    continue;
                }
                let mut buffer = match (take_fresh(&mut fresh, u64::from(chunks.bytes))?, fill) {
                    (Some(block), _) => block.to_vec(),
                    (None, Some(fill)) => fill.repeat(to_usize(
                        u64::from(chunks.bytes) / to_u64(fill.len().max(1))?,
                    )?),
                    (None, None) => vec![0; to_usize(u64::from(chunks.bytes))?],
                };
                rows.emit(&mut BufferSink(&mut buffer), 0, &mut emit)?;
                // A part of a chunk stays in memory, because later writes can add to it.
                chunks.buffers.insert(part.origin.clone(), buffer);
                continue;
            }
            let space = match chunks.spaces.get(&part.origin) {
                Some(space) => *space,
                None => {
                    let space = ChunkSpace {
                        address: alloc.alloc_data(u64::from(chunks.bytes))?,
                        size: chunks.bytes,
                        ready: false,
                    };
                    chunks.set_space(part.origin.clone(), space);
                    space
                }
            };
            if !space.ready {
                // Sink bytes that get no write hold zeros, so a zero fill needs no pass.
                if !part.full {
                    let bytes = u64::from(chunks.bytes);
                    match (take_fresh(&mut fresh, bytes)?, fill) {
                        (Some(block), _) => sink.write_all_at(space.address, block)?,
                        (None, Some(fill)) => write_fill(sink, space.address, bytes, fill)?,
                        (None, None) => {}
                    }
                }
                if let Some(space) = chunks.spaces.get_mut(&part.origin) {
                    space.ready = true;
                }
                chunks.stale = true;
            }
            rows.emit(&mut *sink, space.address, &mut emit)?;
        }
        Ok(())
    }

    /// Call `emit` for each contiguous run of `slab` in the extent at `address`.
    fn write_runs<S: ByteSink>(
        &self,
        sink: &mut S,
        address: u64,
        slab: &Hyperslab,
        len: usize,
        mut emit: impl FnMut(&mut dyn ByteSink, u64, Range<usize>) -> Result<()>,
    ) -> Result<()> {
        let rank = self.shape.len();
        if rank == 0 {
            return emit(sink, address, 0..len);
        }
        // Element strides of each axis in row-major order.
        let mut strides = vec![1u64; rank];
        for axis in (0..rank - 1).rev() {
            strides[axis] = strides[axis + 1]
                .checked_mul(self.shape[axis + 1])
                .ok_or_else(overflow)?;
        }
        // One run spans `axis` and each later axis that the slab covers in full.
        let mut axis = rank - 1;
        while axis > 0 && slab.start[axis] == 0 && slab.count[axis] == self.shape[axis] {
            axis -= 1;
        }
        let run_bytes = slab.count[axis]
            .checked_mul(strides[axis])
            .and_then(|n| n.checked_mul(self.element_size))
            .filter(|&n| n > 0)
            .ok_or_else(overflow)?;
        let first = slab.start[axis]
            .checked_mul(strides[axis])
            .ok_or_else(overflow)?;
        // Position of the current run in the slab, on the axes before `axis`.
        let mut index = vec![0u64; axis];
        let run_len = to_usize(run_bytes)?;
        for run_start in (0..len).step_by(run_len) {
            let run = run_start..run_start.saturating_add(run_len).min(len);
            let mut element = first;
            for a in 0..axis {
                element = slab.start[a]
                    .checked_add(index[a])
                    .and_then(|i| i.checked_mul(strides[a]))
                    .and_then(|n| n.checked_add(element))
                    .ok_or_else(overflow)?;
            }
            let at = element
                .checked_mul(self.element_size)
                .ok_or_else(overflow)?;
            emit(&mut *sink, offset(address, at)?, run)?;
            for a in (0..axis).rev() {
                index[a] += 1;
                if index[a] < slab.count[a] {
                    break;
                }
                index[a] = 0;
            }
        }
        Ok(())
    }
}

/// The filters of the writer for the pipeline of a dataset in a file.
fn filters_of(pipeline: &FilterPipeline) -> Result<Vec<Filter>> {
    let filter = |filter: &crate::message::filter::Filter| {
        let level = filter.client_data.first();
        let level = level.and_then(|&level| u8::try_from(level).ok());
        match (filter.id, level) {
            (id::SHUFFLE, _) => Ok(Filter::Shuffle),
            (id::FLETCHER32, _) => Ok(Filter::Fletcher32),
            (id::DEFLATE, Some(level)) => Ok(Filter::Deflate(level)),
            (id::ZSTD, Some(level)) => Ok(Filter::Zstd(level)),
            _ => Err(Error::unsupported(format!(
                "an edit of a dataset with the filter {}",
                filter.id
            ))),
        }
    };
    pipeline.filters.iter().map(filter).collect()
}

/// The address of the chunk index in `layout` that the close can keep.
///
/// The writer encodes a version 1 B-tree address only. The new superblock
/// gives no node size, so each reader uses the default node size. An index
/// of another node size, or an unknown node size, gets no address.
fn kept_index(layout: &Layout, superblock: &Superblock) -> Option<u64> {
    let default_k = superblock.indexed_storage_internal_node_k == DEFAULT_ISTORE_K
        && superblock.extension_address.is_none();
    match layout {
        Layout::Chunked {
            address,
            index: ChunkIndex::BtreeV1,
            ..
        } if default_k => *address,
        _ => None,
    }
}

/// Take the next `bytes` bytes from `fresh`. `None` when the caller gave no bytes.
///
/// # Errors
///
/// Returns [`Error::Malformed`] when `fresh` has fewer than `bytes` bytes.
fn take_fresh<'a>(fresh: &mut Option<&'a [u8]>, bytes: u64) -> Result<Option<&'a [u8]>> {
    let Some(rest) = fresh.take() else {
        return Ok(None);
    };
    let bytes = to_usize(bytes)?;
    if rest.len() < bytes {
        return Err(defect(format!(
            "a fill pass of {bytes} bytes gets {} bytes of new elements",
            rest.len()
        )));
    }
    let (block, rest) = rest.split_at(bytes);
    *fresh = Some(rest);
    Ok(Some(block))
}

/// Move `at` to the next position from `low` up to `high` in row-major order.
///
/// Return false after the last position. `at` is then at `low` again.
fn advance(at: &mut [u64], low: &[u64], high: &[u64]) -> bool {
    for axis in (0..at.len()).rev() {
        at[axis] += 1;
        if at[axis] < high[axis] {
            return true;
        }
        at[axis] = low[axis];
    }
    false
}

/// Write the fill value to the `total` bytes at `address`.
///
/// # Errors
///
/// Returns [`Error::Malformed`] when `fill` is empty.
fn write_fill<S: ByteSink>(sink: &mut S, address: u64, total: u64, fill: &[u8]) -> Result<()> {
    if fill.is_empty() {
        return Err(defect("a fill value has no bytes"));
    }
    // A fill value fits in one message, so one element fits in a block.
    let per_block = (BLOCK_BYTES / fill.len()).max(1);
    // At least one element in the block makes each loop pass write bytes.
    let elements = (total / to_u64(fill.len())?).clamp(1, to_u64(per_block)?);
    let block = fill.repeat(to_usize(elements)?);
    let mut done = 0u64;
    while done < total {
        let len = (total - done).min(to_u64(block.len())?);
        sink.write_all_at(offset(address, done)?, &block[..to_usize(len)?])?;
        done += len;
    }
    Ok(())
}

/// Whether values in the host byte order need a swap to get `order`.
pub(super) fn needs_swap(order: Option<ByteOrder>) -> bool {
    let host = if cfg!(target_endian = "big") {
        ByteOrder::Big
    } else {
        ByteOrder::Little
    };
    matches!(order, Some(o) if o != host)
}

/// The address `at` bytes past `address`.
fn offset(address: u64, at: u64) -> Result<u64> {
    address.checked_add(at).ok_or_else(overflow)
}

/// The error for a file offset past `u64::MAX`.
fn overflow() -> Error {
    Error::bad_request("a dataset offset passes u64::MAX")
}

/// Convert a length to `usize`, or return an error when it does not fit.
fn to_usize(n: u64) -> Result<usize> {
    usize::try_from(n).map_err(|_| Error::unsupported(format!("{n} does not fit in usize")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::MemorySink;

    /// A memory sink that records the offset and length of each write.
    #[derive(Debug, Default)]
    struct Recorder {
        inner: MemorySink,
        writes: Vec<(u64, usize)>,
        fail_next: bool,
    }

    impl ByteSink for Recorder {
        fn write_all_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
            if std::mem::take(&mut self.fail_next) {
                return Err(Error::Io(std::io::ErrorKind::Other.into()));
            }
            self.writes.push((offset, data.len()));
            self.inner.write_all_at(offset, data)
        }

        fn set_len(&mut self, len: u64) -> Result<()> {
            self.inner.set_len(len)
        }

        fn sync(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn dataset(shape: &[u64]) -> Dataset {
        Dataset::new(DatasetSpec::numeric(DType::Uint(1), ByteOrder::Little, shape).unwrap())
            .unwrap()
    }

    #[test]
    fn full_trailing_axes_merge_into_one_run() {
        let mut d = dataset(&[4, 3, 5]);
        let mut sink = Recorder::default();
        let mut alloc = Allocator::new(0, 0);
        let slab = Hyperslab::new(vec![1, 0, 0], vec![2, 3, 5], &d.shape).unwrap();
        d.write_raw(&mut sink, &mut alloc, &slab, &[1; 30], None)
            .unwrap();
        // No fill pass, then one run of two full planes.
        assert_eq!(sink.writes, [(15, 30)]);
    }

    #[test]
    fn a_zero_fill_value_gets_no_fill_pass() {
        let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[1 << 40])
            .unwrap()
            .with_fill_value(vec![0; 4]);
        let mut d = Dataset::new(spec).unwrap();
        let mut sink = Recorder::default();
        let mut alloc = Allocator::new(0, 0);
        let slab = Hyperslab::new(vec![2], vec![1], &d.shape).unwrap();
        d.write_raw(&mut sink, &mut alloc, &slab, &[9; 4], None)
            .unwrap();
        assert_eq!(sink.writes, [(8, 4)]);
        assert_eq!(alloc.end_of_file(), 1 << 42);
    }

    #[test]
    fn a_failed_fill_pass_keeps_its_space_for_the_next_write() {
        let spec = DatasetSpec::numeric(DType::Uint(1), ByteOrder::Little, &[8])
            .unwrap()
            .with_fill_value(vec![7]);
        let mut d = Dataset::new(spec).unwrap();
        let mut sink = Recorder::default();
        let mut alloc = Allocator::new(0, 0);
        let slab = Hyperslab::new(vec![3], vec![1], &d.shape).unwrap();
        sink.fail_next = true;
        assert!(d
            .write_raw(&mut sink, &mut alloc, &slab, &[1], None)
            .is_err());
        assert_eq!(d.address, None);
        d.write_raw(&mut sink, &mut alloc, &slab, &[1], None)
            .unwrap();
        assert_eq!(d.address, Some(0));
        assert_eq!(sink.writes, [(0, 8), (3, 1)]);
        assert_eq!(alloc.end_of_file(), 8);
        assert_eq!(sink.inner.into_inner(), [7, 7, 7, 1, 7, 7, 7, 7]);
    }

    #[test]
    fn a_chunk_with_a_failed_fill_pass_stays_missing() {
        let mut d = Dataset::new(DatasetSpec::strings(&[6]).chunked(&[4])).unwrap();
        let mut sink = Recorder::default();
        let mut alloc = Allocator::new(0, 0);
        let slab = Hyperslab::new(vec![1], vec![1], &d.shape).unwrap();
        sink.fail_next = true;
        assert!(d
            .write_raw(&mut sink, &mut alloc, &slab, &[2; 16], Some(&[1; 64]))
            .is_err());
        // The close gives the chunk its empty strings, because its index entry is not there.
        assert_eq!(d.missing_chunks(), [vec![0], vec![4]]);
        d.write_chunk(&mut sink, &mut alloc, &[0], &[3; 64])
            .unwrap();
        assert_eq!(d.missing_chunks(), [vec![4]]);
    }

    #[test]
    fn only_new_chunks_with_a_part_of_the_write_get_new_elements() {
        let mut d = Dataset::new(DatasetSpec::strings(&[5]).chunked(&[2])).unwrap();
        let mut sink = Recorder::default();
        let mut alloc = Allocator::new(0, 0);
        // The write covers the chunk at 2 in full and the chunk at 0 in part.
        let slab = Hyperslab::new(vec![1], vec![3], &d.shape).unwrap();
        assert_eq!(d.fill_elements(&slab).unwrap(), 2);
        d.write_raw(&mut sink, &mut alloc, &slab, &[5; 48], Some(&[6; 32]))
            .unwrap();
        assert_eq!(d.fill_elements(&slab).unwrap(), 0);
        assert_eq!(d.missing_chunks(), [vec![4]]);
        // The edge chunk has one element in the shape, but its fill pass covers two.
        let edge = Hyperslab::new(vec![4], vec![1], &d.shape).unwrap();
        assert_eq!(d.fill_elements(&edge).unwrap(), 2);
        let data = sink.inner.into_inner();
        assert_eq!(data[..32], [[6u8; 16], [5; 16]].concat());
    }

    #[test]
    fn typed_values_go_to_the_sink_in_blocks() {
        let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Big, &[300_000]).unwrap();
        let mut d = Dataset::new(spec).unwrap();
        let mut sink = Recorder::default();
        let mut alloc = Allocator::new(0, 0);
        let values: Vec<i32> = (0..300_000).collect();
        d.write_values(&mut sink, &mut alloc, &Hyperslab::all(&d.shape), &values)
            .unwrap();
        assert_eq!(
            sink.writes,
            [(0, 1 << 20), (1 << 20, 1_200_000 - (1 << 20))]
        );
        let data = sink.inner.into_inner();
        assert_eq!(data[4..8], 1i32.to_be_bytes());
        assert_eq!(data[1_199_996..], 299_999i32.to_be_bytes());
    }

    #[test]
    fn a_partial_last_axis_writes_one_run_per_row() {
        let mut d = dataset(&[2, 3, 5]);
        let mut sink = Recorder::default();
        let mut alloc = Allocator::new(0, 0);
        let slab = Hyperslab::new(vec![0, 1, 1], vec![2, 2, 3], &d.shape).unwrap();
        let bytes: Vec<u8> = (1..=12).collect();
        d.write_raw(&mut sink, &mut alloc, &slab, &bytes, None)
            .unwrap();
        assert_eq!(sink.writes, [(6, 3), (11, 3), (21, 3), (26, 3)]);
        let data = sink.inner.into_inner();
        assert_eq!(data[6..9], [1, 2, 3]);
        assert_eq!(data[26..29], [10, 11, 12]);
    }

    #[test]
    fn the_fill_goes_in_blocks_of_one_mebibyte_at_most() {
        let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[600_000])
            .unwrap()
            .with_fill_value(vec![1, 2, 3, 4]);
        let mut d = Dataset::new(spec).unwrap();
        let mut sink = Recorder::default();
        let mut alloc = Allocator::new(0, 0);
        let slab = Hyperslab::new(vec![0], vec![1], &d.shape).unwrap();
        d.write_raw(&mut sink, &mut alloc, &slab, &[9; 4], None)
            .unwrap();
        let lens: Vec<usize> = sink.writes.iter().map(|w| w.1).collect();
        assert_eq!(lens, [1 << 20, 1 << 20, 2_400_000 - (2 << 20), 4]);
        let data = sink.inner.into_inner();
        assert_eq!(data[..8], [9, 9, 9, 9, 1, 2, 3, 4]);
        assert_eq!(data[2_399_996..], [1, 2, 3, 4]);
    }

    #[test]
    fn only_a_version_1_index_with_the_default_node_size_stays() {
        let superblock = |k: u16, extension: Option<u64>| Superblock {
            version: 1,
            sizes: crate::cursor::Sizes::EIGHT,
            superblock_offset: 0,
            base_address: 0,
            end_of_file_address: 0,
            extension_address: extension,
            root: crate::superblock::RootGroup::ObjectHeader(0),
            group_internal_node_k: 16,
            group_leaf_node_k: 4,
            indexed_storage_internal_node_k: k,
        };
        let chunked = |index: ChunkIndex| Layout::Chunked {
            address: Some(800),
            chunk_dims: vec![2],
            element_size: 4,
            index,
        };
        let btree = chunked(ChunkIndex::BtreeV1);
        assert_eq!(kept_index(&btree, &superblock(32, None)), Some(800));
        assert_eq!(kept_index(&btree, &superblock(64, None)), None);
        // An extension can hold another node size.
        assert_eq!(kept_index(&btree, &superblock(32, Some(96))), None);
        let fixed = chunked(ChunkIndex::FixedArray { page_bits: 10 });
        assert_eq!(kept_index(&fixed, &superblock(32, None)), None);
    }

    #[test]
    fn a_full_first_write_writes_no_fill() {
        let mut d = dataset(&[2, 2]);
        let mut sink = Recorder::default();
        let mut alloc = Allocator::new(0, 0);
        let slab = Hyperslab::all(&d.shape);
        d.write_raw(&mut sink, &mut alloc, &slab, &[1, 2, 3, 4], None)
            .unwrap();
        assert_eq!(sink.writes, [(0, 4)]);
    }

    #[test]
    fn wrong_specs_are_rejected() {
        let int4 = |shape: &[u64]| DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, shape);
        assert!(matches!(
            Dataset::new(int4(&[u64::MAX, 2]).unwrap()),
            Err(Error::BadRequest(_))
        ));
        assert!(matches!(
            Dataset::new(int4(&[1; 33]).unwrap()),
            Err(Error::BadRequest(_))
        ));
        assert!(matches!(
            DatasetSpec::numeric(DType::Char, ByteOrder::Little, &[1]),
            Err(Error::Unsupported(_))
        ));
        let compound = DatasetSpec::new(
            Datatype {
                version: 1,
                size: 8,
                class: crate::message::DatatypeClass::Compound { members: vec![] },
            },
            &[1],
        );
        assert!(matches!(Dataset::new(compound), Err(Error::Unsupported(_))));
        // A zero axis gives an empty extent before the product overflows.
        assert_eq!(
            Dataset::new(int4(&[u64::MAX, u64::MAX, 0]).unwrap())
                .unwrap()
                .extent_bytes,
            0
        );
    }
}
