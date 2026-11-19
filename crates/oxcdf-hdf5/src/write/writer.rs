//! The writer: a tree of objects in memory, written to a sink at close.

use std::collections::{HashMap, HashSet};

use super::alloc::Allocator;
use super::dataset::{needs_swap, Dataset, DatasetSpec};
use super::encode::UNDEFINED_ADDRESS;
use super::gheap::{HeapWriter, DESCRIPTOR_BYTES};
use super::header::{encode_header, encode_superblock, header_len_of, message, SUPERBLOCK_BYTES};
use super::message::{
    dimension_list_datatype, encode_attribute, encode_attribute_info, encode_group_info,
    encode_hard_link, encode_link_info, hard_link_len, numeric_datatype, reference_list_datatype,
    vlen_string_datatype, MAX_MESSAGE_BYTES,
};
use super::sink::ByteSink;
use super::{defect, to_u64};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::message::{
    Attribute, ByteOrder, CharSet, Dataspace, DataspaceKind, Datatype, DatatypeClass, StringPad,
    VlenKind,
};
use crate::objheader::{HeaderMessage, MessageType};
use crate::read::{swap_bytes_in_place, Hyperslab};

mod open;

/// Default size of one metadata block in bytes.
const DEFAULT_METADATA_BLOCK_BYTES: u64 = 65536;

/// Largest size of one metadata block in bytes.
const MAX_METADATA_BLOCK_BYTES: u64 = 1 << 30;

/// Default body size of the `Nil` message at the end of each object header.
const DEFAULT_HEADER_SLACK_BYTES: usize = 128;

/// Index of the root group in the node list.
const ROOT: usize = 0;

/// Name of the attribute that holds the scales of each axis of a dataset.
const DIMENSION_LIST: &str = "DIMENSION_LIST";

/// Name of the attribute that holds the datasets that use a scale.
const REFERENCE_LIST: &str = "REFERENCE_LIST";

/// Size in bytes of one `REFERENCE_LIST` element.
const REFERENCE_LIST_ELEMENT_BYTES: usize = 16;

/// Settings for a new file.
///
/// The setters take any value. [`Hdf5Writer::create`] checks the limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteOptions {
    metadata_block_bytes: u64,
    header_slack_bytes: usize,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            metadata_block_bytes: DEFAULT_METADATA_BLOCK_BYTES,
            header_slack_bytes: DEFAULT_HEADER_SLACK_BYTES,
        }
    }
}

impl WriteOptions {
    /// Make the default settings, as [`WriteOptions::default`] does.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the size of one metadata block in bytes.
    ///
    /// The default is 65536 bytes. The limit is 1 GiB.
    /// [`Hdf5Writer::create`] rejects a larger size.
    pub fn metadata_block_bytes(mut self, bytes: u64) -> Self {
        self.metadata_block_bytes = bytes;
        self
    }

    /// Set the body size of the `Nil` message at the end of each object header.
    ///
    /// The default is 128 bytes. The limit is 65535 bytes.
    /// [`Hdf5Writer::create`] rejects a larger size.
    pub fn header_slack_bytes(mut self, bytes: usize) -> Self {
        self.header_slack_bytes = bytes;
        self
    }
}

/// A handle to a group in one [`Hdf5Writer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(usize);

/// A handle to a dataset in one [`Hdf5Writer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DatasetId(usize);

/// A handle to a group or a dataset in one [`Hdf5Writer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectId {
    /// A group.
    Group(GroupId),
    /// A dataset.
    Dataset(DatasetId),
}

impl From<GroupId> for ObjectId {
    fn from(id: GroupId) -> Self {
        Self::Group(id)
    }
}

impl From<DatasetId> for ObjectId {
    fn from(id: DatasetId) -> Self {
        Self::Dataset(id)
    }
}

/// One object in the tree.
#[derive(Debug)]
struct Node {
    /// Link name in the parent. Empty for the root group.
    name: String,
    /// What the object is.
    kind: NodeKind,
    /// The attributes of the object.
    attributes: Attributes,
    /// The scale node of each axis of a dataset. Empty with no scale.
    scales: Vec<Option<usize>>,
    /// The dataset node and axis of each use of this scale, in attach order.
    uses: Vec<(usize, u32)>,
}

/// The attributes of one object, with each message body encoded at put time.
#[derive(Debug, Default)]
struct Attributes {
    /// Message bodies in creation order. The position is the creation index.
    bodies: Vec<Vec<u8>>,
    /// The position of each attribute name in `bodies`.
    index: HashMap<String, usize>,
}

impl Attributes {
    /// Set the body of attribute `name`. A new name goes last.
    fn set(&mut self, name: &str, body: Vec<u8>) {
        match self.index.get(name) {
            Some(&position) => self.bodies[position] = body,
            None => {
                self.index.insert(name.to_owned(), self.bodies.len());
                self.bodies.push(body);
            }
        }
    }
}

/// The links of one group.
#[derive(Debug, Default)]
struct Links {
    /// Child node indices in insert order. This order gives the output.
    children: Vec<usize>,
    /// Child names, for the duplicate check only.
    names: HashSet<String>,
}

/// The kinds of object the writer holds.
#[derive(Debug)]
enum NodeKind {
    /// A group with its links.
    Group(Links),
    /// A dataset. It has no children.
    Dataset(Box<Dataset>),
}

/// Builds a new HDF5 file in memory and writes it to a sink at close.
///
/// A writer that is dropped with no [`Self::close`] leaves a file with no
/// superblock. No reader opens that file.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use oxcdf_hdf5::message::ByteOrder;
/// use oxcdf_hdf5::write::{DatasetSpec, Hdf5Writer, MemorySink, WriteOptions};
/// use oxcdf_hdf5::{DType, Hdf5File, Hyperslab, MemorySource};
///
/// let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::new())?;
/// let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[3])?;
/// let x = w.add_dataset(w.root(), "x", spec)?;
/// w.write(x, &Hyperslab::all(&[3]), &[1i32, 2, 3])?;
/// let bytes = w.close()?.into_inner();
///
/// let file = Hdf5File::from_source(Arc::new(MemorySource::new(bytes)))?;
/// assert_eq!(file.dataset("/x").unwrap().shape, [3]);
/// # Ok::<(), oxcdf_hdf5::Error>(())
/// ```
#[derive(Debug)]
pub struct Hdf5Writer<S: ByteSink> {
    sink: S,
    alloc: Allocator,
    heap: HeapWriter,
    options: WriteOptions,
    nodes: Vec<Node>,
}

impl<S: ByteSink> Hdf5Writer<S> {
    /// Start a file with an empty root group.
    ///
    /// The sink gets dataset values at each write. It gets the object headers
    /// and the superblock at [`Self::close`].
    ///
    /// # Errors
    ///
    /// A header slack above 65535 bytes, or a metadata block above 1 GiB, is
    /// [`Error::BadRequest`].
    pub fn create(sink: S, options: WriteOptions) -> Result<Self> {
        if options.metadata_block_bytes > MAX_METADATA_BLOCK_BYTES {
            return Err(Error::bad_request(format!(
                "a metadata block of {} bytes is above the limit of {MAX_METADATA_BLOCK_BYTES} bytes",
                options.metadata_block_bytes
            )));
        }
        if options.header_slack_bytes > MAX_MESSAGE_BYTES {
            return Err(Error::bad_request(format!(
                "a header slack of {} bytes is above the {MAX_MESSAGE_BYTES}-byte message limit",
                options.header_slack_bytes
            )));
        }
        let alloc = Allocator::new(SUPERBLOCK_BYTES, options.metadata_block_bytes);
        let root = Node {
            name: String::new(),
            kind: NodeKind::Group(Links::default()),
            attributes: Attributes::default(),
            scales: Vec::new(),
            uses: Vec::new(),
        };
        Ok(Self {
            sink,
            alloc,
            heap: HeapWriter::default(),
            options,
            nodes: vec![root],
        })
    }

    /// The root group.
    pub fn root(&self) -> GroupId {
        GroupId(ROOT)
    }

    /// Add an empty group `name` to `parent` and return its id.
    ///
    /// # Errors
    ///
    /// An unknown parent, a wrong name or a name that exists in `parent` is
    /// [`Error::BadRequest`]. A name is wrong when it is empty, is `.`, or
    /// holds `/` or a NUL byte. A name too long for a link message is
    /// [`Error::Unsupported`]. An error changes nothing.
    pub fn add_group(&mut self, parent: GroupId, name: &str) -> Result<GroupId> {
        self.check_link(parent, name)?;
        let kind = NodeKind::Group(Links::default());
        Ok(GroupId(self.push_child(parent, name, kind)))
    }

    /// Add a dataset `name` to `parent` and return its id.
    ///
    /// `spec` sets the datatype, the shape, the storage and the filters.
    ///
    /// # Errors
    ///
    /// The name errors are those of [`Self::add_group`]. A fill value that is
    /// not one element, a wrong shape, or an extent above `u64::MAX` bytes is
    /// [`Error::BadRequest`]. A datatype with a size of 0 bytes or with bits
    /// outside its size is also [`Error::BadRequest`]. A datatype class that
    /// the writer cannot encode is [`Error::Unsupported`]. An error changes
    /// nothing.
    pub fn add_dataset(
        &mut self,
        parent: GroupId,
        name: &str,
        spec: DatasetSpec,
    ) -> Result<DatasetId> {
        self.check_link(parent, name)?;
        let kind = NodeKind::Dataset(Box::new(self.new_dataset(spec)?));
        Ok(DatasetId(self.push_child(parent, name, kind)))
    }

    /// Give the dataset `dataset` of the group `parent` the link name `name`.
    ///
    /// The dataset keeps its place in the group, its values and its
    /// attributes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] for a dataset that is not in `parent`,
    /// a wrong name or a name that exists in `parent`. Returns
    /// [`Error::Unsupported`] for a name too long for a link message. An
    /// error changes nothing.
    pub fn rename_dataset(
        &mut self,
        parent: GroupId,
        dataset: DatasetId,
        name: &str,
    ) -> Result<()> {
        if !self.links(parent)?.children.contains(&dataset.0) {
            return Err(Error::bad_request(format!(
                "the group holds no dataset with id {}",
                dataset.0
            )));
        }
        self.node_index(dataset.into())?;
        self.check_link(parent, name)?;
        let old = std::mem::replace(&mut self.nodes[dataset.0].name, name.to_owned());
        if let NodeKind::Group(links) = &mut self.nodes[parent.0].kind {
            links.names.remove(&old);
            links.names.insert(name.to_owned());
        }
        Ok(())
    }

    /// Replace the definition of a dataset that holds no values.
    ///
    /// The dataset keeps its name, its place in the group and its
    /// attributes.
    ///
    /// # Errors
    ///
    /// An unknown dataset, or a dataset with a write, a resize or a scale
    /// list, is [`Error::BadRequest`]. The errors of `spec` are those of
    /// [`Self::add_dataset`]. An error changes nothing.
    pub fn redefine_dataset(&mut self, dataset: DatasetId, spec: DatasetSpec) -> Result<()> {
        // Fail on the dataset before the heap gets an object.
        self.blank_dataset(dataset)?;
        let new = self.new_dataset(spec)?;
        *self.blank_dataset(dataset)? = new;
        Ok(())
    }

    /// The dataset `id`, or [`Error::BadRequest`] when it has values or a scale list.
    fn blank_dataset(&mut self, id: DatasetId) -> Result<&mut Dataset> {
        let Some(Node {
            kind: NodeKind::Dataset(d),
            scales,
            uses,
            ..
        }) = self.nodes.get_mut(id.0)
        else {
            return Err(no_dataset(id));
        };
        if d.is_touched() || !scales.is_empty() || !uses.is_empty() {
            return Err(Error::bad_request(
                "a dataset with values or scales keeps its definition",
            ));
        }
        Ok(&mut **d)
    }

    /// Check `spec` and make its dataset.
    ///
    /// A contiguous string dataset with no file space reads as empty
    /// strings, as in a file from netcdf-c. A chunked string dataset gets no
    /// fill value: the HDF5 library cannot read a chunk with no file space
    /// when a string fill value is there. The close gives each chunk space.
    fn new_dataset(&mut self, spec: DatasetSpec) -> Result<Dataset> {
        if !is_vlen_string(&spec.datatype) || spec.chunk_shape.is_some() {
            return Dataset::new(spec);
        }
        // Fail on a wrong definition before the heap gets an object.
        Dataset::new(spec.clone())?;
        let reference = self.heap.put_empty(&mut self.sink, &mut self.alloc)?;
        Dataset::with_fill_reference(spec, Some(reference))
    }

    /// Make `scale` the dimension scale of axis `axis` of `dataset`.
    ///
    /// At close, `dataset` gets a `DIMENSION_LIST` attribute and `scale` gets
    /// a `REFERENCE_LIST` attribute, as the HDF5 dimension scale interface
    /// writes them.
    ///
    /// # Errors
    ///
    /// An unknown dataset, a dataset as its own scale, an axis outside the
    /// rank, an axis that has a scale, or an attribute with one of the two
    /// names is [`Error::BadRequest`]. A list above the message limit, or an
    /// object with too many attributes, is [`Error::Unsupported`]. An error
    /// changes nothing.
    pub fn attach_scale(
        &mut self,
        dataset: DatasetId,
        axis: usize,
        scale: DatasetId,
    ) -> Result<()> {
        let rank = match self.nodes.get(dataset.0).map(|n| &n.kind) {
            Some(NodeKind::Dataset(d)) => d.rank(),
            _ => return Err(no_dataset(dataset)),
        };
        let scale_node = self.node_index(scale.into())?;
        // The HDF5 dimension scale interface also rejects this.
        if scale_node == dataset.0 {
            return Err(Error::bad_request("a dataset is not its own scale"));
        }
        let Some(axis32) = u32::try_from(axis).ok().filter(|_| axis < rank) else {
            return Err(Error::bad_request(format!(
                "axis {axis} is outside a dataset of rank {rank}"
            )));
        };
        let node = &self.nodes[dataset.0];
        if node.scales.get(axis).is_some_and(Option::is_some) {
            return Err(Error::bad_request(format!("axis {axis} has a scale")));
        }
        let scale_attributes = &self.nodes[scale_node].attributes;
        let taken = (node.scales.is_empty() && node.attributes.index.contains_key(DIMENSION_LIST))
            || (self.nodes[scale_node].uses.is_empty()
                && scale_attributes.index.contains_key(REFERENCE_LIST));
        if taken {
            return Err(Error::bad_request(
                "an attribute has the name of a scale list",
            ));
        }
        // The same checks as at close, so that close cannot fail on these lists.
        let uses = self.nodes[scale_node].uses.len() + 1;
        encode_attribute(&reference_list(vec![
            0;
            uses * REFERENCE_LIST_ELEMENT_BYTES
        ])?)?;
        self.check_attribute_room(dataset.0, usize::from(node.scales.is_empty()))?;
        let new_reference_list = self.nodes[scale_node].uses.is_empty();
        self.check_attribute_room(scale_node, usize::from(new_reference_list))?;
        let node = &mut self.nodes[dataset.0];
        node.scales.resize(rank, None);
        node.scales[axis] = Some(scale_node);
        self.nodes[scale_node].uses.push((dataset.0, axis32));
        Ok(())
    }

    /// Write `bytes` in the dataset byte order to the elements of `slab`.
    ///
    /// The first write with bytes reserves the full extent. If it does not
    /// cover the dataset, the fill value goes to the full extent first. With
    /// no fill value, or a fill value of zeros, the sink gets no fill bytes
    /// and the other elements read as zeros.
    ///
    /// # Errors
    ///
    /// An unknown dataset, a slab outside the shape or a wrong byte count is
    /// [`Error::BadRequest`]. A variable-length dataset is also
    /// [`Error::BadRequest`]: use [`Self::write_strings`]. A failed sink call
    /// returns its error, such as [`Error::Io`].
    pub fn write_raw(&mut self, dataset: DatasetId, slab: &Hyperslab, bytes: &[u8]) -> Result<()> {
        let Self {
            sink, alloc, nodes, ..
        } = self;
        match nodes.get_mut(dataset.0).map(|n| &mut n.kind) {
            Some(NodeKind::Dataset(d)) if d.is_vlen() => Err(Error::bad_request(
                "a variable-length dataset takes no raw bytes",
            )),
            Some(NodeKind::Dataset(d)) => d.write_raw(sink, alloc, slab, bytes, None),
            _ => Err(no_dataset(dataset)),
        }
    }

    /// Check that the chunks of `dataset` accept a write to `slab`.
    ///
    /// The slab can pass the current shape. Use this before a
    /// [`Self::resize`] for a write, because no axis can shrink again.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] for an unknown dataset. Returns
    /// [`Error::Unsupported`] when `slab` covers a part of a chunk that is
    /// in the file with its filters.
    pub fn check_write(&self, dataset: DatasetId, slab: &Hyperslab) -> Result<()> {
        match self.nodes.get(dataset.0).map(|n| &n.kind) {
            Some(NodeKind::Dataset(d)) => d.check_chunks(slab, self.sink.can_read()),
            _ => Err(no_dataset(dataset)),
        }
    }

    /// Set the shape of a chunked dataset.
    ///
    /// Each axis keeps its size or grows, up to its largest size. The new
    /// elements read as the fill value.
    ///
    /// # Errors
    ///
    /// An unknown dataset, a contiguous dataset or a wrong shape is
    /// [`Error::BadRequest`]. An error changes nothing.
    pub fn resize(&mut self, dataset: DatasetId, shape: &[u64]) -> Result<()> {
        match self.nodes.get_mut(dataset.0).map(|n| &mut n.kind) {
            Some(NodeKind::Dataset(d)) => d.resize(shape),
            _ => Err(no_dataset(dataset)),
        }
    }

    /// Write each chunk of `dataset` that is in memory to the file, with its filters.
    ///
    /// A dataset with filters keeps a chunk in memory while writes cover only
    /// a part of it. This call frees that memory. After it, such a chunk
    /// takes only a write that covers it in full: a write to a part of it is
    /// [`Error::Unsupported`]. A dataset with no filters has no chunks in
    /// memory.
    ///
    /// # Errors
    ///
    /// An unknown dataset is [`Error::BadRequest`]. A failed sink call
    /// returns its error, such as [`Error::Io`].
    pub fn flush_chunks(&mut self, dataset: DatasetId) -> Result<()> {
        let Self {
            sink, alloc, nodes, ..
        } = self;
        match nodes.get_mut(dataset.0).map(|n| &mut n.kind) {
            Some(NodeKind::Dataset(d)) => d.flush_chunks(sink, alloc),
            _ => Err(no_dataset(dataset)),
        }
    }

    /// Write the strings `values` to the elements of `slab`.
    ///
    /// The dataset must hold variable-length strings. Each string goes to the
    /// global heap, and the element holds a reference. A second write to an
    /// element leaves the first string unused in the file. An element that
    /// gets file space with no value gets an empty string of its own, as the
    /// HDF5 library gives it. The close does the same for each chunk with no
    /// write, so a large string dataset with few writes takes 32 bytes for
    /// each element.
    ///
    /// # Errors
    ///
    /// An unknown dataset, a string with a NUL byte, a slab outside the shape
    /// or a wrong number of values is [`Error::BadRequest`]. Another datatype
    /// is [`Error::TypeMismatch`]. An error before the first heap write
    /// changes nothing.
    pub fn write_strings<T: AsRef<str>>(
        &mut self,
        dataset: DatasetId,
        slab: &Hyperslab,
        values: &[T],
    ) -> Result<()> {
        let Self {
            sink,
            alloc,
            nodes,
            heap,
            ..
        } = self;
        let Some(Node {
            name,
            kind: NodeKind::Dataset(d),
            ..
        }) = nodes.get_mut(dataset.0)
        else {
            return Err(no_dataset(dataset));
        };
        if !is_vlen_string(d.datatype()) {
            return Err(Error::TypeMismatch {
                stored: DType::of(d.datatype()).name(),
                asked: "a variable-length string",
                what: name.clone(),
            });
        }
        d.check_selection(slab, values.len())?;
        let references = string_references(heap, sink, alloc, values)?;
        // Each element that gets file space with no value gets its own empty string.
        let mut fresh = Vec::new();
        for _ in 0..d.fill_elements(slab)? {
            fresh.extend_from_slice(&heap.put_empty(sink, alloc)?);
        }
        d.write_raw(sink, alloc, slab, &references, Some(&fresh))
    }

    /// Write `values` to the elements of `slab`.
    ///
    /// The dataset datatype must have the size and the class of the numeric
    /// datatype of `T` in the dataset byte order. The datatype version is not
    /// compared, because the writer always encodes version 1.
    ///
    /// # Errors
    ///
    /// Any other datatype is [`Error::TypeMismatch`]. The other errors are
    /// those of [`Self::write_raw`].
    pub fn write<T: Element>(
        &mut self,
        dataset: DatasetId,
        slab: &Hyperslab,
        values: &[T],
    ) -> Result<()> {
        let Self {
            sink, alloc, nodes, ..
        } = self;
        let Some(Node {
            name,
            kind: NodeKind::Dataset(d),
            ..
        }) = nodes.get_mut(dataset.0)
        else {
            return Err(no_dataset(dataset));
        };
        let datatype = d.datatype();
        let order = datatype.byte_order();
        let exact = order
            .and_then(|o| numeric_datatype(T::DTYPE, o).ok())
            .is_some_and(|e| e.size == datatype.size && e.class == datatype.class);
        if !exact {
            let mut stored = DType::of(datatype).name();
            if stored == T::NAME {
                // Same class and size byte, but another size, precision or layout.
                stored = format!("a nonstandard {}-byte datatype", datatype.size);
            }
            return Err(Error::TypeMismatch {
                stored,
                asked: T::NAME,
                what: name.clone(),
            });
        }
        d.write_values(sink, alloc, slab, values)
    }

    /// Check that `parent` is a group that can take a new link `name`.
    fn check_link(&self, parent: GroupId, name: &str) -> Result<()> {
        check_name(name)?;
        let links = self.links(parent)?;
        if links.names.contains(name) {
            return Err(Error::bad_request(format!(
                "the group already holds a link {name:?}"
            )));
        }
        // The same check as at close, so that close cannot fail on this name.
        encode_hard_link(name, UNDEFINED_ADDRESS, to_u64(links.children.len())?)?;
        Ok(())
    }

    /// Add a node of `kind` as the last child of the checked group `parent`.
    fn push_child(&mut self, parent: GroupId, name: &str, kind: NodeKind) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            name: name.to_owned(),
            kind,
            attributes: Attributes::default(),
            scales: Vec::new(),
            uses: Vec::new(),
        });
        if let NodeKind::Group(links) = &mut self.nodes[parent.0].kind {
            links.children.push(id);
            links.names.insert(name.to_owned());
        }
        id
    }

    /// The links of group `id`, or [`Error::BadRequest`] when `id` is no group.
    fn links(&self, id: GroupId) -> Result<&Links> {
        match self.nodes.get(id.0).map(|n| &n.kind) {
            Some(NodeKind::Group(links)) => Ok(links),
            _ => Err(Error::bad_request(format!("no group with id {}", id.0))),
        }
    }

    /// Put `attribute` on `object`.
    ///
    /// An attribute with the same name gets the new value and keeps its
    /// creation order. An object holds at most 65535 attributes. This count
    /// includes the scale lists of [`Self::attach_scale`].
    ///
    /// # Errors
    ///
    /// An unknown object, an empty name or a wrong attribute is
    /// [`Error::BadRequest`]. The name of a scale list that the writer makes
    /// for `object` is also [`Error::BadRequest`]. A body above the message
    /// limit or too many attributes is [`Error::Unsupported`]. An error
    /// changes nothing.
    pub fn put_attribute(
        &mut self,
        object: impl Into<ObjectId>,
        attribute: Attribute,
    ) -> Result<()> {
        let node = self.node_index(object.into())?;
        let body = self.check_attribute(node, &attribute)?;
        self.nodes[node].attributes.set(&attribute.name, body);
        Ok(())
    }

    /// Check that `node` can take `attribute`, and return the message body.
    fn check_attribute(&self, node: usize, attribute: &Attribute) -> Result<Vec<u8>> {
        let Node {
            attributes,
            scales,
            uses,
            ..
        } = &self.nodes[node];
        if attribute.name.is_empty() {
            return Err(Error::bad_request("an attribute name is empty"));
        }
        let scale_list = (attribute.name == DIMENSION_LIST && !scales.is_empty())
            || (attribute.name == REFERENCE_LIST && !uses.is_empty());
        if scale_list {
            return Err(Error::bad_request(format!(
                "the writer makes the attribute {:?} of this object",
                attribute.name
            )));
        }
        // The same check as at close, so that close cannot fail on this attribute.
        let body = encode_attribute(attribute)?;
        if !attributes.index.contains_key(&attribute.name) {
            self.check_attribute_room(node, 1)?;
        }
        Ok(body)
    }

    /// Check that `node` can take `added` new attributes, with the scale lists of close.
    fn check_attribute_room(&self, node: usize, added: usize) -> Result<()> {
        let Node {
            attributes,
            scales,
            uses,
            ..
        } = &self.nodes[node];
        // Close adds one attribute for each scale list that is not empty.
        let count = attributes.bodies.len()
            + usize::from(!scales.is_empty())
            + usize::from(!uses.is_empty())
            + added;
        // The attribute info message stores the count in 16 bits.
        if u16::try_from(count).is_err() {
            return Err(Error::unsupported(format!(
                "an object holds at most {} attributes",
                u16::MAX
            )));
        }
        Ok(())
    }

    /// Put a little-endian numeric attribute on `object`.
    ///
    /// The dataspace is simple with rank 1, also for one value, as in
    /// netcdf-c.
    ///
    /// # Errors
    ///
    /// An empty `values` is [`Error::BadRequest`]. The other errors are those
    /// of [`Self::put_attribute`].
    pub fn put_numeric_attribute<T: Element>(
        &mut self,
        object: impl Into<ObjectId>,
        name: &str,
        values: &[T],
    ) -> Result<()> {
        if values.is_empty() {
            return Err(Error::bad_request(format!(
                "attribute {name:?} has no values"
            )));
        }
        let dataspace = Dataspace {
            kind: DataspaceKind::Simple,
            dims: vec![to_u64(values.len())?],
            max_dims: None,
        };
        let mut data = Vec::new();
        T::extend_ne_bytes(values, &mut data);
        if needs_swap(Some(ByteOrder::Little)) {
            swap_bytes_in_place(&mut data, std::mem::size_of::<T>());
        }
        let attribute = Attribute {
            name: name.to_owned(),
            datatype: numeric_datatype(T::DTYPE, ByteOrder::Little)?,
            dataspace,
            data,
        };
        self.put_attribute(object, attribute)
    }

    /// Put a text attribute on `object` as one fixed string, as in netcdf-c.
    ///
    /// The string size is the UTF-8 byte length of `text`, and no NUL byte
    /// is stored. The pad is `NullTerminate` and the dataspace is scalar.
    /// ASCII text gets the ASCII character set, other text UTF-8.
    ///
    /// # Errors
    ///
    /// Empty text or text with a NUL byte is [`Error::BadRequest`]. The other
    /// errors are those of [`Self::put_attribute`].
    pub fn put_text_attribute(
        &mut self,
        object: impl Into<ObjectId>,
        name: &str,
        text: &str,
    ) -> Result<()> {
        if text.is_empty() {
            return Err(Error::bad_request(format!(
                "text attribute {name:?} is empty"
            )));
        }
        if text.contains('\0') {
            return Err(Error::bad_request(format!(
                "text attribute {name:?} holds a NUL byte"
            )));
        }
        let data = text.as_bytes().to_vec();
        let size = u32::try_from(data.len()).map_err(|_| {
            Error::unsupported(format!(
                "text attribute {name:?} of {} bytes is above the 4-byte size limit",
                data.len()
            ))
        })?;
        let charset = if text.is_ascii() {
            CharSet::Ascii
        } else {
            CharSet::Utf8
        };
        let attribute = Attribute {
            name: name.to_owned(),
            datatype: Datatype {
                version: 1,
                size,
                class: DatatypeClass::String {
                    pad: StringPad::NullTerminate,
                    charset,
                },
            },
            dataspace: Dataspace {
                kind: DataspaceKind::Scalar,
                dims: Vec::new(),
                max_dims: None,
            },
            data,
        };
        self.put_attribute(object, attribute)
    }

    /// Put an attribute of variable-length UTF-8 strings on `object`.
    ///
    /// The dataspace is simple with rank 1, also for one string, as netcdf-c
    /// writes the netCDF string type. Each string goes to the global heap. A
    /// second put with the same name leaves the first strings unused in the
    /// file.
    ///
    /// # Errors
    ///
    /// An empty `values`, or a string with a NUL byte, is
    /// [`Error::BadRequest`]. The other errors are those of
    /// [`Self::put_attribute`]. An error before the first heap write changes
    /// nothing.
    pub fn put_string_attribute<T: AsRef<str>>(
        &mut self,
        object: impl Into<ObjectId>,
        name: &str,
        values: &[T],
    ) -> Result<()> {
        if values.is_empty() {
            return Err(Error::bad_request(format!(
                "attribute {name:?} has no values"
            )));
        }
        let node = self.node_index(object.into())?;
        let dataspace = Dataspace {
            kind: DataspaceKind::Simple,
            dims: vec![to_u64(values.len())?],
            max_dims: None,
        };
        let mut attribute = Attribute {
            name: name.to_owned(),
            datatype: vlen_string_datatype(),
            dataspace,
            data: vec![0; values.len().saturating_mul(DESCRIPTOR_BYTES)],
        };
        // Fail on the object and the attribute before a string goes to the heap.
        self.check_attribute(node, &attribute)?;
        attribute.data =
            string_references(&mut self.heap, &mut self.sink, &mut self.alloc, values)?;
        let body = encode_attribute(&attribute)?;
        self.nodes[node].attributes.set(name, body);
        Ok(())
    }

    /// The node index of `object`, or [`Error::BadRequest`] for an unknown object.
    fn node_index(&self, object: ObjectId) -> Result<usize> {
        match object {
            ObjectId::Group(id) => self.links(id).map(|_| id.0),
            ObjectId::Dataset(id) => match self.nodes.get(id.0).map(|n| &n.kind) {
                Some(NodeKind::Dataset(_)) => Ok(id.0),
                _ => Err(no_dataset(id)),
            },
        }
    }

    /// Write the file and return the sink.
    ///
    /// # Errors
    ///
    /// A failed sink call returns its error, such as [`Error::Io`]. A file
    /// end past `u64::MAX - 1` bytes is [`Error::BadRequest`]. When `close`
    /// returns an error, the sink is lost and the file is not valid.
    pub fn close(mut self) -> Result<S> {
        let order = self.parents_first();
        // The lists get their size here and their addresses below.
        self.put_scale_lists(None)?;
        let mut addresses = vec![UNDEFINED_ADDRESS; self.nodes.len()];
        let mut lens = vec![0u64; self.nodes.len()];
        for &node in &order {
            lens[node] = to_u64(self.header_len(node))?;
            addresses[node] = self.alloc.alloc_metadata(lens[node])?;
        }
        self.put_scale_lists(Some(&addresses))?;
        self.fill_string_chunks()?;
        self.heap.flush(&mut self.sink)?;
        // After the headers have their addresses, so that the root group stays first.
        let Self {
            sink, alloc, nodes, ..
        } = &mut self;
        for node in nodes.iter_mut() {
            if let NodeKind::Dataset(d) = &mut node.kind {
                d.write_index(sink, alloc)?;
            }
        }
        for &node in &order {
            let messages = self.take_messages(node, &addresses)?;
            let header = encode_header(&messages, self.options.header_slack_bytes)?;
            // A longer header would overwrite the next allocation.
            if to_u64(header.len())? != lens[node] {
                return Err(defect(format!(
                    "the object header of node {node} differs from its computed length"
                )));
            }
            self.sink.write_all_at(addresses[node], &header)?;
        }
        let end_of_file = self.alloc.end_of_file();
        self.sink
            .write_all_at(0, &encode_superblock(addresses[ROOT], end_of_file))?;
        self.sink.set_len(end_of_file)?;
        self.sink.sync()?;
        Ok(self.sink)
    }

    /// Give each chunk with no file space in a string dataset an empty string for each element.
    ///
    /// The HDF5 library reads an element with no file space as no string,
    /// and some readers fail on that.
    fn fill_string_chunks(&mut self) -> Result<()> {
        let Self {
            sink,
            alloc,
            nodes,
            heap,
            ..
        } = self;
        for node in nodes.iter_mut() {
            let NodeKind::Dataset(d) = &mut node.kind else {
                continue;
            };
            if !is_vlen_string(d.datatype()) {
                continue;
            }
            let elements = d.chunk_elements();
            let mut chunk = Vec::new();
            for origin in d.missing_chunks() {
                chunk.clear();
                for _ in 0..elements {
                    chunk.extend_from_slice(&heap.put_empty(sink, alloc)?);
                }
                d.write_chunk(sink, alloc, &origin, &chunk)?;
            }
        }
        Ok(())
    }

    /// Put the two scale list attributes on each object that needs them.
    ///
    /// With no addresses, each reference is zero and the heap gets nothing.
    /// The size of each attribute is the same in the two cases.
    fn put_scale_lists(&mut self, addresses: Option<&[u64]>) -> Result<()> {
        let Self {
            sink,
            alloc,
            nodes,
            heap,
            ..
        } = self;
        for node in nodes.iter_mut() {
            if !node.scales.is_empty() {
                let mut data = Vec::with_capacity(node.scales.len() * DESCRIPTOR_BYTES);
                for &scale in &node.scales {
                    let reference = match (scale, addresses) {
                        (Some(scale), Some(addresses)) => {
                            heap.put(sink, alloc, &addresses[scale].to_le_bytes(), 1)?
                        }
                        _ => [0; DESCRIPTOR_BYTES],
                    };
                    data.extend_from_slice(&reference);
                }
                let attribute = Attribute {
                    name: DIMENSION_LIST.to_owned(),
                    datatype: dimension_list_datatype(),
                    dataspace: Dataspace {
                        kind: DataspaceKind::Simple,
                        dims: vec![to_u64(node.scales.len())?],
                        max_dims: None,
                    },
                    data,
                };
                let body = encode_attribute(&attribute)?;
                node.attributes.set(DIMENSION_LIST, body);
            }
            if !node.uses.is_empty() {
                let mut data = Vec::with_capacity(node.uses.len() * REFERENCE_LIST_ELEMENT_BYTES);
                for &(dataset, axis) in &node.uses {
                    let address = addresses.map_or(0, |a| a[dataset]);
                    data.extend_from_slice(&address.to_le_bytes());
                    data.extend_from_slice(&axis.to_le_bytes());
                    data.extend_from_slice(&[0; 4]);
                }
                let body = encode_attribute(&reference_list(data)?)?;
                node.attributes.set(REFERENCE_LIST, body);
            }
        }
        Ok(())
    }

    /// Node indices in a walk from the root that puts each parent before its children.
    fn parents_first(&self) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut stack = vec![ROOT];
        while let Some(node) = stack.pop() {
            order.push(node);
            if let NodeKind::Group(links) = &self.nodes[node].kind {
                stack.extend(links.children.iter().rev());
            }
        }
        order
    }

    /// Exact length of the object header of `node`. It needs no address.
    fn header_len(&self, node: usize) -> usize {
        let Node {
            kind, attributes, ..
        } = &self.nodes[node];
        let attribute_info = encode_attribute_info(0).len();
        let attribute_lens = attributes.bodies.iter().map(Vec::len);
        let slack = self.options.header_slack_bytes;
        match kind {
            NodeKind::Group(links) => {
                let fixed = [
                    encode_link_info(0).len(),
                    encode_group_info().len(),
                    attribute_info,
                ];
                let link_lens = links
                    .children
                    .iter()
                    .map(|&child| hard_link_len(&self.nodes[child].name));
                header_len_of(
                    fixed.into_iter().chain(link_lens).chain(attribute_lens),
                    slack,
                )
            }
            NodeKind::Dataset(d) => {
                let fixed = d.message_lens().into_iter().chain([attribute_info]);
                header_len_of(fixed.chain(attribute_lens), slack)
            }
        }
    }

    /// Move out the header messages of `node`, with its children at `addresses`.
    fn take_messages(&mut self, node: usize, addresses: &[u64]) -> Result<Vec<HeaderMessage>> {
        let bodies = std::mem::take(&mut self.nodes[node].attributes.bodies);
        let attribute_info = message(
            MessageType::AttributeInfo,
            encode_attribute_info(to_u16(bodies.len())?),
        );
        let mut messages = match &mut self.nodes[node].kind {
            NodeKind::Group(links) => {
                let children = std::mem::take(&mut links.children);
                let mut messages = Vec::with_capacity(children.len() + bodies.len() + 3);
                messages.push(message(
                    MessageType::LinkInfo,
                    encode_link_info(to_u64(children.len())?),
                ));
                messages.push(message(MessageType::GroupInfo, encode_group_info()));
                messages.push(attribute_info);
                for (position, &child) in children.iter().enumerate() {
                    let body = encode_hard_link(
                        &self.nodes[child].name,
                        addresses[child],
                        to_u64(position)?,
                    )?;
                    messages.push(message(MessageType::Link, body));
                }
                messages
            }
            NodeKind::Dataset(d) => {
                let mut messages = Vec::with_capacity(bodies.len() + 5);
                messages.extend(d.take_messages());
                messages.push(attribute_info);
                messages
            }
        };
        for (index, body) in bodies.into_iter().enumerate() {
            messages.push(HeaderMessage {
                creation_order: Some(to_u16(index)?),
                ..message(MessageType::Attribute, body)
            });
        }
        Ok(messages)
    }
}

/// The `REFERENCE_LIST` attribute with `data`, one element for each use.
fn reference_list(data: Vec<u8>) -> Result<Attribute> {
    Ok(Attribute {
        name: REFERENCE_LIST.to_owned(),
        datatype: reference_list_datatype(),
        dataspace: Dataspace {
            kind: DataspaceKind::Simple,
            dims: vec![to_u64(data.len() / REFERENCE_LIST_ELEMENT_BYTES)?],
            max_dims: None,
        },
        data,
    })
}

/// Whether `datatype` is a variable-length string.
fn is_vlen_string(datatype: &Datatype) -> bool {
    matches!(
        datatype.class,
        DatatypeClass::VariableLength {
            kind: VlenKind::String,
            ..
        }
    )
}

/// Put each string in the global heap and return the references, one after the other.
fn string_references<S: ByteSink, T: AsRef<str>>(
    heap: &mut HeapWriter,
    sink: &mut S,
    alloc: &mut Allocator,
    values: &[T],
) -> Result<Vec<u8>> {
    let mut lengths = Vec::with_capacity(values.len());
    for value in values {
        let text = value.as_ref();
        if text.contains('\0') {
            return Err(Error::bad_request("a string holds a NUL byte"));
        }
        lengths.push(u32::try_from(text.len()).map_err(|_| {
            Error::unsupported(format!(
                "a string of {} bytes is above the 4-byte length limit",
                text.len()
            ))
        })?);
    }
    let mut references = Vec::with_capacity(values.len().saturating_mul(DESCRIPTOR_BYTES));
    for (value, length) in values.iter().zip(lengths) {
        // An empty string is an object. A reference of zeros is no string.
        let reference = match length {
            0 => heap.put_empty(sink, alloc)?,
            _ => heap.put(sink, alloc, value.as_ref().as_bytes(), length)?,
        };
        references.extend_from_slice(&reference);
    }
    Ok(references)
}

/// Reject a link name that HDF5 cannot resolve as one path component.
fn check_name(name: &str) -> Result<()> {
    if name.is_empty() || name == "." || name.contains(['/', '\0']) {
        return Err(Error::bad_request(format!(
            "{name:?} is not a valid link name"
        )));
    }
    Ok(())
}

/// The error for a dataset id that names no dataset.
fn no_dataset(id: DatasetId) -> Error {
    Error::bad_request(format!("no dataset with id {}", id.0))
}

/// Convert a count or index to `u16`, or return an error when it does not fit.
fn to_u16(n: usize) -> Result<u16> {
    u16::try_from(n).map_err(|_| Error::unsupported(format!("{n} does not fit in 16 bits")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Ctx;
    use crate::index::Hdf5File;
    use crate::objheader::ObjectHeader;
    use crate::source::MemorySource;
    use crate::superblock::Superblock;
    use crate::write::message::encode_datatype;
    use crate::write::MemorySink;
    use std::sync::Arc;

    fn reopen(bytes: Vec<u8>) -> Hdf5File {
        Hdf5File::from_source(Arc::new(MemorySource::new(bytes))).unwrap()
    }

    #[test]
    fn a_writer_moves_to_another_thread() {
        fn assert_send<T: Send>() {}
        assert_send::<Hdf5Writer<MemorySink>>();
        assert_send::<Hdf5Writer<crate::write::FileSink>>();
    }

    #[test]
    fn a_rejected_name_leaves_the_writer_usable() {
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        assert!(matches!(
            w.add_group(root, &"n".repeat(70000)),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            w.add_group(GroupId(99), "a"),
            Err(Error::BadRequest(_))
        ));
        w.add_group(root, "a").unwrap();
        let file = reopen(w.close().unwrap().into_inner());
        assert_eq!(file.root().groups.len(), 1);
    }

    #[test]
    fn a_slack_above_the_body_limit_is_rejected_at_create() {
        let above = WriteOptions::default().header_slack_bytes(65536);
        assert!(matches!(
            Hdf5Writer::create(MemorySink::new(), above),
            Err(Error::BadRequest(_))
        ));
        let at = WriteOptions::default().header_slack_bytes(65535);
        let mut w = Hdf5Writer::create(MemorySink::new(), at).unwrap();
        let root = w.root();
        w.add_group(root, "a").unwrap();
        let file = reopen(w.close().unwrap().into_inner());
        assert!(file.root().group("a").is_some());
    }

    #[test]
    fn a_metadata_block_above_the_limit_is_rejected_at_create() {
        let above = WriteOptions::default().metadata_block_bytes((1 << 30) + 1);
        assert!(matches!(
            Hdf5Writer::create(MemorySink::new(), above),
            Err(Error::BadRequest(_))
        ));
        let at = WriteOptions::default().metadata_block_bytes(1 << 30);
        assert!(Hdf5Writer::create(MemorySink::new(), at).is_ok());
    }

    #[test]
    fn a_wide_root_header_is_larger_than_the_metadata_block() {
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        for i in 0..2000 {
            w.add_group(root, &format!("group_{i:04}")).unwrap();
        }
        for i in 0..100i32 {
            w.put_numeric_attribute(root, &format!("attr_{i:03}"), &[i])
                .unwrap();
        }
        let source = MemorySource::new(w.close().unwrap().into_inner());
        let superblock = Superblock::read(&source).unwrap();
        let address = superblock.root_object_header_address().unwrap();
        let header = ObjectHeader::read(Ctx::new(&source, &superblock), address).unwrap();
        let area: usize = header.messages.iter().map(|m| 6 + m.data.len()).sum();
        assert!(area > 65536, "{area}");
    }

    #[test]
    fn a_duplicate_is_rejected_after_many_inserts() {
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        for i in 0..1000 {
            w.add_group(root, &format!("g{i}")).unwrap();
        }
        let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[1]).unwrap();
        assert!(matches!(
            w.add_dataset(root, "g500", spec.clone()),
            Err(Error::BadRequest(_))
        ));
        w.add_dataset(root, "d", spec).unwrap();
        assert!(matches!(w.add_group(root, "d"), Err(Error::BadRequest(_))));
        let file = reopen(w.close().unwrap().into_inner());
        assert_eq!(file.root().groups.len(), 1000);
        assert_eq!(file.datasets().len(), 1);
    }

    #[test]
    fn options_set_the_block_and_the_slack() {
        let options = WriteOptions::default()
            .metadata_block_bytes(0)
            .header_slack_bytes(0);
        let mut w = Hdf5Writer::create(MemorySink::new(), options).unwrap();
        let root = w.root();
        w.add_group(root, "a").unwrap();
        let bytes = w.close().unwrap().into_inner();
        let link = encode_hard_link("a", 0, 0).unwrap().len();
        // Root: three info messages and one link. Child: three info messages.
        let infos = (6 + 26) + (6 + 2) + (6 + 20);
        assert_eq!(bytes.len(), 48 + (14 + infos + 6 + link) + (14 + infos));
        let file = reopen(bytes);
        assert_eq!(file.root().address, 48);
        assert!(file.root().group("a").is_some());
    }

    #[test]
    fn attribute_frames_hold_the_creation_index() {
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        w.add_group(root, "g").unwrap();
        w.put_numeric_attribute(root, "a", &[1u8]).unwrap();
        w.put_numeric_attribute(root, "b", &[2u8]).unwrap();
        w.put_text_attribute(root, "c", "x").unwrap();
        w.put_numeric_attribute(root, "a", &[3u8]).unwrap();
        let source = MemorySource::new(w.close().unwrap().into_inner());
        let superblock = Superblock::read(&source).unwrap();
        let address = superblock.root_object_header_address().unwrap();
        let header = ObjectHeader::read(Ctx::new(&source, &superblock), address).unwrap();

        let mut next = 0;
        for m in &header.messages {
            if m.kind == MessageType::Attribute {
                assert_eq!(m.creation_order, Some(next));
                next += 1;
            } else {
                assert_eq!(m.creation_order, Some(0), "{:?}", m.kind);
            }
        }
        assert_eq!(next, 3);
    }

    #[test]
    fn wrong_attribute_targets_and_names_are_bad_requests() {
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        assert!(matches!(
            w.put_numeric_attribute(GroupId(99), "a", &[1u8]),
            Err(Error::BadRequest(_))
        ));
        assert!(matches!(
            w.put_numeric_attribute(DatasetId(0), "a", &[1u8]),
            Err(Error::BadRequest(_))
        ));
        assert!(matches!(
            w.put_numeric_attribute(root, "", &[1u8]),
            Err(Error::BadRequest(_))
        ));
        assert!(matches!(
            w.put_text_attribute(root, "nul\0", "x"),
            Err(Error::BadRequest(_))
        ));
        let file = reopen(w.close().unwrap().into_inner());
        assert!(file.root().attributes.is_empty());
    }

    #[test]
    fn ids_of_the_wrong_kind_are_bad_requests() {
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[2]).unwrap();
        let d = w.add_dataset(root, "d", spec.clone()).unwrap();
        let as_group = GroupId(d.0);
        let as_dataset = DatasetId(root.0);
        let all = Hyperslab::all(&[2]);
        fn bad<T>(r: Result<T>) -> bool {
            matches!(r, Err(Error::BadRequest(_)))
        }
        assert!(bad(w.add_group(as_group, "g")));
        assert!(bad(w.add_dataset(as_group, "e", spec)));
        assert!(bad(w.put_text_attribute(as_group, "a", "x")));
        assert!(bad(w.put_text_attribute(as_dataset, "a", "x")));
        assert!(bad(w.write(as_dataset, &all, &[1i32, 2])));
        assert!(bad(w.write_raw(as_dataset, &all, &[0; 8])));
        assert!(bad(w.write_raw(DatasetId(99), &all, &[0; 8])));
        let file = reopen(w.close().unwrap().into_inner());
        assert!(file.root().attributes.is_empty());
        assert!(file.dataset("/d").unwrap().attributes.is_empty());
    }

    #[test]
    fn a_wrong_dataset_spec_changes_nothing() {
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        let int4 = |shape: &[u64]| DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, shape);
        assert!(matches!(
            w.add_dataset(root, "d", int4(&[u64::MAX, 2]).unwrap()),
            Err(Error::BadRequest(_))
        ));
        assert!(matches!(
            w.add_dataset(root, "d", int4(&[1; 33]).unwrap()),
            Err(Error::BadRequest(_))
        ));
        assert!(matches!(
            w.add_dataset(root, "a/b", int4(&[1]).unwrap()),
            Err(Error::BadRequest(_))
        ));
        w.add_dataset(root, "d", int4(&[1]).unwrap()).unwrap();
        let file = reopen(w.close().unwrap().into_inner());
        assert_eq!(file.datasets().len(), 1);
    }

    #[test]
    fn a_dataset_header_holds_the_messages_in_order() {
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[2]).unwrap();
        let d = w.add_dataset(root, "d", spec).unwrap();
        w.put_text_attribute(d, "a", "x").unwrap();
        w.put_text_attribute(d, "b", "y").unwrap();
        let file = reopen(w.close().unwrap().into_inner());
        let address = file.dataset("/d").unwrap().address;
        let header = ObjectHeader::read(file.ctx(), address).unwrap();
        let kinds: Vec<_> = header.messages.iter().map(|m| m.kind).collect();
        assert_eq!(
            kinds,
            [
                MessageType::Dataspace,
                MessageType::Datatype,
                MessageType::FillValue,
                MessageType::DataLayout,
                MessageType::AttributeInfo,
                MessageType::Attribute,
                MessageType::Attribute,
            ]
        );
        assert_eq!(header.messages[6].creation_order, Some(1));
        assert_eq!(encode_attribute_info(2), header.messages[4].data);
    }

    #[test]
    fn an_attached_scale_gives_the_two_lists() {
        use crate::heap::{GlobalHeap, VlenDescriptor};
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        let int4 = |shape: &[u64]| DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, shape);
        let scale = w.add_dataset(root, "x", int4(&[3]).unwrap()).unwrap();
        let data = w.add_dataset(root, "d", int4(&[2, 3]).unwrap()).unwrap();
        w.attach_scale(data, 1, scale).unwrap();
        let file = reopen(w.close().unwrap().into_inner());
        let scale = file.dataset("/x").unwrap();
        let data = file.dataset("/d").unwrap();

        let list = data.attribute("DIMENSION_LIST").unwrap();
        assert_eq!(list.dataspace.dims, [2]);
        assert_eq!(&list.data[..16], [0; 16]);
        let sizes = file.superblock().sizes;
        let descriptor = VlenDescriptor::parse(&list.data[16..], sizes).unwrap();
        assert_eq!(descriptor.length, 1);
        let heap = GlobalHeap::read(file.ctx(), descriptor.collection_address).unwrap();
        let object = heap.object(descriptor.object_index as u16).unwrap();
        assert_eq!(object.data[..8], scale.address.to_le_bytes());

        let back = scale.attribute("REFERENCE_LIST").unwrap();
        assert_eq!(back.dataspace.dims, [1]);
        let mut expected = data.address.to_le_bytes().to_vec();
        expected.extend_from_slice(&[1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(back.data, expected);
    }

    #[test]
    fn the_scale_lists_have_the_netcdf_c_datatypes() {
        let list = encode_datatype(&dimension_list_datatype()).unwrap();
        assert_eq!(
            list,
            [0x19, 0, 0, 0, 0x10, 0, 0, 0, 0x17, 0, 0, 0, 8, 0, 0, 0]
        );
        let mut expected = vec![0x36, 2, 0, 0, 0x10, 0, 0, 0];
        expected.extend_from_slice(b"dataset\0\0");
        expected.extend_from_slice(&[0x17, 0, 0, 0, 8, 0, 0, 0]);
        expected.extend_from_slice(b"dimension\0\x08");
        expected.extend_from_slice(&[0x10, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0x20, 0]);
        assert_eq!(
            encode_datatype(&reference_list_datatype()).unwrap(),
            expected
        );
    }

    #[test]
    fn wrong_scale_attachments_are_bad_requests() {
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[3]).unwrap();
        let scale = w.add_dataset(root, "x", spec.clone()).unwrap();
        let data = w.add_dataset(root, "d", spec).unwrap();
        fn bad<T>(r: Result<T>) -> bool {
            matches!(r, Err(Error::BadRequest(_)))
        }
        assert!(bad(w.attach_scale(data, 1, scale)));
        assert!(bad(w.attach_scale(data, 0, DatasetId(99))));
        assert!(bad(w.attach_scale(DatasetId(0), 0, scale)));
        w.attach_scale(data, 0, scale).unwrap();
        assert!(bad(w.attach_scale(data, 0, scale)));
        assert!(bad(w.put_numeric_attribute(data, "DIMENSION_LIST", &[1u8])));
        assert!(bad(w.put_numeric_attribute(
            scale,
            "REFERENCE_LIST",
            &[1u8]
        )));
        let file = reopen(w.close().unwrap().into_inner());
        assert_eq!(file.dataset("/d").unwrap().attributes.len(), 1);
    }

    /// Two writers with one int dataset `d` and one int scale `x` on axis 0.
    fn two_writers_with_a_scale() -> [(Hdf5Writer<MemorySink>, DatasetId, DatasetId); 2] {
        std::array::from_fn(|_| {
            let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
            let root = w.root();
            let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[3]).unwrap();
            let scale = w.add_dataset(root, "x", spec.clone()).unwrap();
            let data = w.add_dataset(root, "d", spec).unwrap();
            w.attach_scale(data, 0, scale).unwrap();
            (w, data, scale)
        })
    }

    #[test]
    fn a_rejected_redefinition_writes_no_heap_object() {
        let [(mut changed, data, _), (same, ..)] = two_writers_with_a_scale();
        let strings = DatasetSpec::strings(&[3]);
        assert!(matches!(
            changed.redefine_dataset(DatasetId(99), strings.clone()),
            Err(Error::BadRequest(_))
        ));
        assert!(matches!(
            changed.redefine_dataset(data, strings),
            Err(Error::BadRequest(_))
        ));
        let changed = changed.close().unwrap().into_inner();
        assert_eq!(changed, same.close().unwrap().into_inner());
    }

    #[test]
    fn a_rejected_string_attribute_writes_no_heap_object() {
        let [(mut changed, data, scale), (same, ..)] = two_writers_with_a_scale();
        assert!(matches!(
            changed.put_string_attribute(data, "DIMENSION_LIST", &["a"]),
            Err(Error::BadRequest(_))
        ));
        assert!(matches!(
            changed.put_string_attribute(scale, "REFERENCE_LIST", &["a"]),
            Err(Error::BadRequest(_))
        ));
        let changed = changed.close().unwrap().into_inner();
        assert_eq!(changed, same.close().unwrap().into_inner());
    }

    #[test]
    fn the_scale_lists_count_toward_the_attribute_limit() {
        let [(mut w, data, scale), _] = two_writers_with_a_scale();
        // Close adds one scale list to each dataset.
        for (object, last) in [(data, "DIMENSION_LIST"), (scale, "REFERENCE_LIST")] {
            for i in 0..usize::from(u16::MAX) - 1 {
                w.put_numeric_attribute(object, &format!("a{i}"), &[0u8])
                    .unwrap();
            }
            assert!(
                matches!(
                    w.put_numeric_attribute(object, "one_more", &[0u8]),
                    Err(Error::Unsupported(_))
                ),
                "{last}"
            );
        }
        let file = reopen(w.close().unwrap().into_inner());
        let count = usize::from(u16::MAX);
        assert_eq!(file.dataset("/d").unwrap().attributes.len(), count);
        assert_eq!(file.dataset("/x").unwrap().attributes.len(), count);
    }

    #[test]
    fn a_dataset_is_not_its_own_scale() {
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[3]).unwrap();
        let d = w.add_dataset(root, "d", spec).unwrap();
        assert!(matches!(w.attach_scale(d, 0, d), Err(Error::BadRequest(_))));
        let file = reopen(w.close().unwrap().into_inner());
        assert!(file.dataset("/d").unwrap().attributes.is_empty());
    }

    #[test]
    fn a_dataset_with_no_values_takes_a_new_spec() {
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        let int4 = |shape: &[u64]| DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, shape);
        let d = w.add_dataset(root, "d", int4(&[2]).unwrap()).unwrap();
        w.put_text_attribute(d, "units", "m").unwrap();
        w.redefine_dataset(d, int4(&[4]).unwrap().chunked(&[2]))
            .unwrap();
        w.write(d, &Hyperslab::all(&[4]), &[1i32, 2, 3, 4]).unwrap();
        assert!(matches!(
            w.redefine_dataset(d, int4(&[2]).unwrap()),
            Err(Error::BadRequest(_))
        ));
        let file = reopen(w.close().unwrap().into_inner());
        let d = file.dataset("/d").unwrap();
        assert_eq!(d.shape, [4]);
        assert_eq!(d.attributes.len(), 1);
        let raw = crate::read::read_hyperslab(file.ctx(), d, &Hyperslab::all(&[4])).unwrap();
        assert_eq!(i32::from_ne_slice(&raw.bytes), [1, 2, 3, 4]);
    }

    #[test]
    fn a_dataset_takes_a_new_link_name() {
        let mut w = Hdf5Writer::create(MemorySink::new(), WriteOptions::default()).unwrap();
        let root = w.root();
        let group = w.add_group(root, "g").unwrap();
        let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[2]).unwrap();
        let d = w.add_dataset(root, "old", spec.clone()).unwrap();
        w.add_dataset(root, "other", spec.clone()).unwrap();
        w.write(d, &Hyperslab::all(&[2]), &[1i32, 2]).unwrap();
        fn bad<T>(r: Result<T>) -> bool {
            matches!(r, Err(Error::BadRequest(_)))
        }
        assert!(bad(w.rename_dataset(root, d, "other")));
        assert!(bad(w.rename_dataset(root, d, "a/b")));
        assert!(bad(w.rename_dataset(group, d, "new")));
        assert!(bad(w.rename_dataset(root, DatasetId(99), "new")));
        w.rename_dataset(root, d, "new").unwrap();
        // The earlier name is free again.
        w.add_dataset(root, "old", spec).unwrap();
        let file = reopen(w.close().unwrap().into_inner());
        let names: Vec<&str> = file
            .root()
            .datasets
            .iter()
            .map(|d| d.name.as_str())
            .collect();
        assert_eq!(names, ["new", "other", "old"]);
        let d = file.dataset("/new").unwrap();
        let raw = crate::read::read_hyperslab(file.ctx(), d, &Hyperslab::all(&[2])).unwrap();
        assert_eq!(i32::from_ne_slice(&raw.bytes), [1, 2]);
    }
}
