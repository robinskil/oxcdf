//! Open a file that exists for edits.
//!
//! The writer makes its tree from the index of the reader. The values stay
//! where they are. The close writes each object header again at the end of
//! the file, and the superblock last. The earlier headers stay unused in
//! the file, so each edit session adds the size of all headers to the file.

use std::collections::HashMap;

use super::{
    Dataset, DatasetId, GroupId, Hdf5Writer, Links, NodeKind, ObjectId, WriteOptions,
    DIMENSION_LIST, REFERENCE_LIST, ROOT,
};
use crate::error::{Error, Result};
use crate::heap::{GlobalHeap, VlenDescriptor};
use crate::index::{GroupIndex, Hdf5File};
use crate::message::{Attribute, Datatype, DatatypeClass};
use crate::objheader::{MessageType, ObjectHeader};
use crate::write::alloc::Allocator;
use crate::write::gheap::DESCRIPTOR_BYTES;
use crate::write::sink::ByteSink;

/// Object header address to node, for each object of the file.
type Nodes = HashMap<u64, usize>;

/// The state of the import of a file.
#[derive(Debug, Default)]
struct Import {
    /// The node of each object of the file.
    nodes: Nodes,
    /// Each `DIMENSION_LIST` attribute, with the node and the path of its dataset.
    lists: Vec<(usize, String, Attribute)>,
    /// The path of the object that the import adds now.
    at: String,
}

impl<S: ByteSink> Hdf5Writer<S> {
    /// Open the file in `sink` for edits. `file` is the index of that file.
    ///
    /// Each group, dataset and attribute of the file is in the writer, as
    /// after the calls that made it. [`Self::find_group`] and
    /// [`Self::find_dataset`] give the ids. A dataset of the file keeps its
    /// definition. Until [`Self::close`] writes the superblock, a reader
    /// sees the earlier groups, datasets and attributes. A write to values
    /// of the file changes them at once.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] for a file that the writer cannot
    /// describe again: a user block, addresses of another size than 8
    /// bytes, compact storage, a filter that is not shuffle, deflate,
    /// Zstandard or Fletcher-32, a datatype that the writer cannot encode,
    /// an object reference outside the scale lists, an object with two
    /// links, or attributes that the reader cannot list in full. A shared
    /// header message, a header message that the close cannot write again,
    /// and a link that the index does not show are also
    /// [`Error::Unsupported`]. A content of the file that the checks of the
    /// writer refuse is [`Error::Unsupported`], not [`Error::BadRequest`].
    /// The message of such an error names the path of the object. The
    /// other errors are those of [`Self::create`] and of the reader. An
    /// error changes no byte of the sink.
    pub fn open(sink: S, file: &Hdf5File, options: WriteOptions) -> Result<Self> {
        let mut writer = Self::create(sink, options)?;
        let superblock = file.superblock();
        let plain = superblock.sizes.offset == 8
            && superblock.sizes.length == 8
            && superblock.base_address == 0
            && superblock.superblock_offset == 0;
        if !plain {
            return Err(Error::unsupported(
                "an edit of a file with a user block or with addresses that are not 8 bytes",
            ));
        }
        // Each header gets its exact space at the close, after the end of the file.
        writer.alloc = Allocator::new(superblock.end_of_file_address, 0);
        let mut import = Import::default();
        writer
            .import_group(file, file.root(), ROOT, &mut import)
            .map_err(|error| in_object(&import.at, error))?;
        let mut heaps: HashMap<u64, GlobalHeap> = HashMap::new();
        for (node, path, list) in &import.lists {
            writer
                .import_scales(file, *node, list, &import.nodes, &mut heaps)
                .map_err(|error| in_object(path, error))?;
        }
        Ok(writer)
    }

    /// The group `name` of the group `parent`.
    pub fn find_group(&self, parent: GroupId, name: &str) -> Option<GroupId> {
        self.find(parent, name)
            .filter(|&child| matches!(self.nodes[child].kind, NodeKind::Group(_)))
            .map(GroupId)
    }

    /// The dataset `name` of the group `parent`.
    pub fn find_dataset(&self, parent: GroupId, name: &str) -> Option<DatasetId> {
        self.find(parent, name)
            .filter(|&child| matches!(self.nodes[child].kind, NodeKind::Dataset(_)))
            .map(DatasetId)
    }

    /// The node of the link `name` in the group `parent`.
    fn find(&self, parent: GroupId, name: &str) -> Option<usize> {
        let links = self.links(parent).ok()?;
        let named = |&&child: &&usize| self.nodes[child].name == name;
        links.children.iter().find(named).copied()
    }

    /// Add the content of `group` to the node `node`.
    ///
    /// At an error, `import.at` holds the path of the object that the error is about.
    fn import_group(
        &mut self,
        file: &Hdf5File,
        group: &GroupIndex,
        node: usize,
        import: &mut Import,
    ) -> Result<()> {
        import.at.clone_from(&group.path);
        claim(&mut import.nodes, group.address, node)?;
        check_messages(file, group.address)?;
        if let Some(name) = group.hidden_links.first() {
            return Err(Error::unsupported(format!(
                "the link {name:?}, which is not a hard link to a group or a dataset"
            )));
        }
        let object = GroupId(node).into();
        self.import_attributes(object, &group.attributes, group.attributes_complete, import)?;
        for dataset in &group.datasets {
            import.at.clone_from(&dataset.path);
            check_messages(file, dataset.address)?;
            let records = dataset.chunks(file.ctx())?.unwrap_or(&[]);
            let existing = Dataset::existing(dataset, records, file.superblock())?;
            let kind = NodeKind::Dataset(Box::new(existing));
            self.check_link(GroupId(node), &dataset.name)?;
            let child = self.push_child(GroupId(node), &dataset.name, kind);
            claim(&mut import.nodes, dataset.address, child)?;
            let complete = dataset.attributes_complete;
            let object = DatasetId(child).into();
            self.import_attributes(object, &dataset.attributes, complete, import)?;
        }
        for child in &group.groups {
            import.at.clone_from(&child.path);
            self.check_link(GroupId(node), &child.name)?;
            let kind = NodeKind::Group(Links::default());
            let child_node = self.push_child(GroupId(node), &child.name, kind);
            self.import_group(file, child, child_node, import)?;
        }
        Ok(())
    }

    /// Put each attribute on `object`. A `DIMENSION_LIST` goes to `import.lists`.
    fn import_attributes(
        &mut self,
        object: ObjectId,
        attributes: &[Attribute],
        complete: bool,
        import: &mut Import,
    ) -> Result<()> {
        if !complete {
            return Err(Error::unsupported(
                "attributes that the reader cannot list in full",
            ));
        }
        for attribute in attributes {
            match attribute.name.as_str() {
                DIMENSION_LIST => {
                    let node = self.node_index(object)?;
                    let path = import.at.clone();
                    import.lists.push((node, path, attribute.clone()));
                }
                // The close makes this list again from each `DIMENSION_LIST`.
                REFERENCE_LIST => {}
                _ if holds_reference(&attribute.datatype) => {
                    // The object headers move, so a stored address is wrong after the close.
                    return Err(Error::unsupported(format!(
                        "the object reference attribute {:?}",
                        attribute.name
                    )));
                }
                _ => {
                    let mut attribute = attribute.clone();
                    let elements = attribute.dataspace.element_count();
                    let bytes = elements.checked_mul(u64::from(attribute.datatype.size));
                    // A version 1 attribute message pads its data to 8 bytes.
                    if let Some(bytes) = bytes.and_then(|bytes| usize::try_from(bytes).ok()) {
                        attribute.data.truncate(bytes);
                    }
                    self.put_attribute(object, attribute)?;
                }
            }
        }
        Ok(())
    }

    /// Attach the scale of each axis that the `DIMENSION_LIST` attribute `list` names.
    fn import_scales(
        &mut self,
        file: &Hdf5File,
        node: usize,
        list: &Attribute,
        nodes: &Nodes,
        heaps: &mut HashMap<u64, GlobalHeap>,
    ) -> Result<()> {
        let wrong = || Error::unsupported("a DIMENSION_LIST of another form");
        if !matches!(list.datatype.class, DatatypeClass::VariableLength { .. }) {
            return Err(wrong());
        }
        let sizes = file.superblock().sizes;
        for (axis, reference) in list.data.chunks_exact(DESCRIPTOR_BYTES).enumerate() {
            let descriptor = VlenDescriptor::parse(reference, sizes)?;
            // An axis with no scale.
            if descriptor.length == 0 {
                continue;
            }
            let heap = match heaps.entry(descriptor.collection_address) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(GlobalHeap::read(file.ctx(), descriptor.collection_address)?)
                }
            };
            let index = u16::try_from(descriptor.object_index).map_err(|_| wrong())?;
            let address = heap
                .object(index)
                .and_then(|object| object.data.get(..8))
                .and_then(|bytes| <[u8; 8]>::try_from(bytes).ok())
                .map(u64::from_le_bytes);
            // One axis with two scales has two addresses in its object.
            let scale = address
                .filter(|_| descriptor.length == 1)
                .and_then(|address| nodes.get(&address))
                .ok_or_else(wrong)?;
            self.attach_scale(DatasetId(node), axis, DatasetId(*scale))?;
        }
        Ok(())
    }
}

/// Record that the object at `address` is `node`.
fn claim(nodes: &mut Nodes, address: u64, node: usize) -> Result<()> {
    match nodes.insert(address, node) {
        None => Ok(()),
        Some(_) => Err(Error::unsupported("a second link to one object")),
    }
}

/// Put the path of the object in an error that the content of the file causes.
///
/// The file is valid, so a refusal of the writer checks is [`Error::Unsupported`].
/// The other errors, such as [`Error::Io`] and [`Error::Malformed`], keep their kind.
fn in_object(path: &str, error: Error) -> Error {
    match error {
        Error::Unsupported(message) | Error::BadRequest(message) | Error::NotFound(message) => {
            Error::unsupported(format!("an edit of {path}: {message}"))
        }
        Error::TypeMismatch { .. } => Error::unsupported(format!("an edit of {path}: {error}")),
        other => other,
    }
}

/// Check that the close can write the object header at `address` again with no loss.
///
/// # Errors
///
/// Returns [`Error::Unsupported`] for a shared message, and for a message
/// that the close does not write again and that a reader can see. The
/// errors of [`ObjectHeader::read`] keep their kind.
fn check_messages(file: &Hdf5File, address: u64) -> Result<()> {
    let header = ObjectHeader::read(file.ctx(), address)?;
    for message in &header.messages {
        if message.is_shared() {
            return Err(Error::unsupported(format!(
                "a shared {:?} message",
                message.kind
            )));
        }
        if !can_rewrite(message.kind) {
            return Err(Error::unsupported(format!(
                "a header message of the type {:?}",
                message.kind
            )));
        }
    }
    Ok(())
}

/// Whether the close can write a header with a message of type `kind` again, with no change that a reader sees.
fn can_rewrite(kind: MessageType) -> bool {
    use MessageType as M;
    match kind {
        // The close writes these again from the index of the reader.
        M::Dataspace
        | M::Datatype
        | M::FillValueOld
        | M::FillValue
        | M::DataLayout
        | M::FilterPipeline
        | M::Attribute
        | M::AttributeInfo
        | M::LinkInfo
        | M::GroupInfo
        | M::Link
        | M::SymbolTable => true,
        // The close drops these. A reader sees no change.
        M::Nil
        | M::ObjectHeaderContinuation
        | M::ObjectModificationTimeOld
        | M::ObjectModificationTime
        | M::BtreeKValues
        | M::ObjectReferenceCount => true,
        M::ExternalDataFiles
        | M::Bogus
        | M::ObjectComment
        | M::SharedMessageTable
        | M::DriverInfo
        | M::Unknown(_) => false,
    }
}

/// Whether a value of `datatype` holds a reference to an object or a region.
fn holds_reference(datatype: &Datatype) -> bool {
    match &datatype.class {
        DatatypeClass::Reference { .. } => true,
        DatatypeClass::Compound { members } => members
            .iter()
            .any(|member| holds_reference(&member.datatype)),
        DatatypeClass::VariableLength { base, .. } | DatatypeClass::Enum { base, .. } => {
            holds_reference(base)
        }
        _ => false,
    }
}
