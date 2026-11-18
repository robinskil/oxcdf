//! The writer: a tree of objects in memory, written to a sink at close.

use super::alloc::Allocator;
use super::encode::UNDEFINED_ADDRESS;
use super::header::{encode_header, encode_superblock, header_len, SUPERBLOCK_BYTES};
use super::message::{
    encode_attribute_info, encode_group_info, encode_hard_link, encode_link_info,
};
use super::sink::ByteSink;
use crate::error::{Error, Result};
use crate::objheader::{HeaderMessage, MessageType};

/// Default size of one metadata block in bytes.
const DEFAULT_METADATA_BLOCK_BYTES: u64 = 65536;

/// Default body size of the `Nil` message at the end of each object header.
const DEFAULT_HEADER_SLACK_BYTES: usize = 128;

/// Index of the root group in the node list.
const ROOT: usize = 0;

/// Settings for a new file.
#[derive(Debug, Clone)]
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
    /// Set the size of one metadata block in bytes.
    pub fn metadata_block_bytes(mut self, bytes: u64) -> Self {
        self.metadata_block_bytes = bytes;
        self
    }

    /// Set the body size of the `Nil` message at the end of each object header.
    pub fn header_slack_bytes(mut self, bytes: usize) -> Self {
        self.header_slack_bytes = bytes;
        self
    }
}

/// A handle to a group in one [`Hdf5Writer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(usize);

/// One object in the tree.
#[derive(Debug)]
struct Node {
    /// Link name in the parent. Empty for the root group.
    name: String,
    /// What the object is.
    kind: NodeKind,
}

/// The kinds of object the writer holds.
#[derive(Debug)]
enum NodeKind {
    /// A group with its children in insert order.
    Group { children: Vec<usize> },
}

/// Builds a new HDF5 file in memory and writes it to a sink at close.
#[derive(Debug)]
pub struct Hdf5Writer<S: ByteSink> {
    sink: S,
    alloc: Allocator,
    options: WriteOptions,
    nodes: Vec<Node>,
}

impl<S: ByteSink> Hdf5Writer<S> {
    /// Start a file with an empty root group. Nothing is written until [`Self::close`].
    pub fn create(sink: S, options: WriteOptions) -> Result<Self> {
        let alloc = Allocator::new(SUPERBLOCK_BYTES, options.metadata_block_bytes);
        let root = Node {
            name: String::new(),
            kind: NodeKind::Group {
                children: Vec::new(),
            },
        };
        Ok(Self {
            sink,
            alloc,
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
    /// An unknown parent, a wrong name or a name that exists in `parent` is
    /// [`Error::BadRequest`]. A name too long for a link message is
    /// [`Error::Unsupported`]. An error changes nothing.
    pub fn add_group(&mut self, parent: GroupId, name: &str) -> Result<GroupId> {
        check_name(name)?;
        let NodeKind::Group { children } = &self
            .nodes
            .get(parent.0)
            .ok_or_else(|| Error::bad_request(format!("no group with id {}", parent.0)))?
            .kind;
        if children.iter().any(|&c| self.nodes[c].name == name) {
            return Err(Error::bad_request(format!(
                "the group already holds a link {name:?}"
            )));
        }
        // The same check as at close, so that close cannot fail on this name.
        encode_hard_link(name, UNDEFINED_ADDRESS, to_u64(children.len())?)?;

        let id = self.nodes.len();
        self.nodes.push(Node {
            name: name.to_owned(),
            kind: NodeKind::Group {
                children: Vec::new(),
            },
        });
        let NodeKind::Group { children } = &mut self.nodes[parent.0].kind;
        children.push(id);
        Ok(GroupId(id))
    }

    /// Write the file and return the sink.
    pub fn close(mut self) -> Result<S> {
        let order = self.parents_first();
        let mut addresses = vec![UNDEFINED_ADDRESS; self.nodes.len()];
        for &node in &order {
            // Message sizes do not depend on addresses, so placeholders give the length.
            let messages = self.messages(node, &addresses)?;
            let len = to_u64(header_len(&messages, self.options.header_slack_bytes))?;
            addresses[node] = self.alloc.alloc_metadata(len)?;
        }
        for &node in &order {
            let messages = self.messages(node, &addresses)?;
            let header = encode_header(&messages, self.options.header_slack_bytes)?;
            self.sink.write_all_at(addresses[node], &header)?;
        }
        let end_of_file = self.alloc.end_of_file();
        self.sink
            .write_all_at(0, &encode_superblock(addresses[ROOT], end_of_file))?;
        self.sink.set_len(end_of_file)?;
        self.sink.sync()?;
        Ok(self.sink)
    }

    /// Node indices in a walk from the root that puts each parent before its children.
    fn parents_first(&self) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut stack = vec![ROOT];
        while let Some(node) = stack.pop() {
            order.push(node);
            let NodeKind::Group { children } = &self.nodes[node].kind;
            stack.extend(children.iter().rev());
        }
        order
    }

    /// The header messages of `node`, with its children at `addresses`.
    fn messages(&self, node: usize, addresses: &[u64]) -> Result<Vec<HeaderMessage>> {
        let NodeKind::Group { children } = &self.nodes[node].kind;
        let mut messages = Vec::with_capacity(children.len() + 3);
        messages.push(message(
            MessageType::LinkInfo,
            encode_link_info(to_u64(children.len())?),
        ));
        messages.push(message(MessageType::GroupInfo, encode_group_info()));
        messages.push(message(
            MessageType::AttributeInfo,
            encode_attribute_info(0),
        ));
        for (position, &child) in children.iter().enumerate() {
            let body =
                encode_hard_link(&self.nodes[child].name, addresses[child], to_u64(position)?)?;
            messages.push(message(MessageType::Link, body));
        }
        Ok(messages)
    }
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

/// A header message with no flags and no creation order.
fn message(kind: MessageType, data: Vec<u8>) -> HeaderMessage {
    HeaderMessage {
        kind,
        flags: 0,
        creation_order: None,
        data,
    }
}

/// Convert a count or length to `u64`, or return an error when it does not fit.
fn to_u64(n: usize) -> Result<u64> {
    u64::try_from(n).map_err(|_| Error::unsupported(format!("{n} does not fit in 64 bits")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Hdf5File;
    use crate::source::MemorySource;
    use crate::write::MemorySink;
    use std::sync::Arc;

    fn reopen(bytes: Vec<u8>) -> Hdf5File {
        Hdf5File::from_source(Arc::new(MemorySource::new(bytes))).unwrap()
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
}
