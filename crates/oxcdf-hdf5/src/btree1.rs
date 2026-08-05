//! Version 1 B-trees.
//!
//! One structure serves two unrelated purposes, distinguished by a node type
//! byte:
//!
//! * Type 0 indexes the children of an old-style group. Leaves point at symbol
//!   table nodes.
//! * Type 1 indexes the chunks of a chunked dataset. Leaves point at the raw
//!   chunk bytes, and each key carries the chunk's stored size, its filter mask
//!   and its position in the dataset.
//!
//! Type 1 is the important one. Walking it once produces the whole chunk index,
//! which is exactly the immutable structure that makes parallel reads possible:
//! afterwards every chunk is just a byte range that any thread may fetch.

use std::collections::HashSet;

use crate::context::Ctx;
use crate::cursor::Cursor;
use crate::error::{Error, Result};

/// Signature of a version 1 B-tree node.
pub const TREE_SIGNATURE: &[u8; 4] = b"TREE";

/// Guard against cycles in a damaged file.
const MAX_NODES: usize = 1 << 20;

/// Node type 0: an old-style group.
pub const NODE_TYPE_GROUP: u8 = 0;
/// Node type 1: chunked dataset storage.
pub const NODE_TYPE_CHUNK: u8 = 1;

/// One chunk of a chunked dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRecord {
    /// Stored size in bytes, after filters.
    pub size: u32,
    /// Which filters were skipped for this chunk. A set bit means the filter at
    /// that position was not applied.
    pub filter_mask: u32,
    /// Position of the chunk's first element, in elements, per dimension.
    pub offset: Vec<u64>,
    /// Address of the chunk's bytes.
    pub address: u64,
}

/// Walk a type 1 B-tree and return every chunk it indexes.
///
/// `rank` is the dataset's rank. The on-disk keys carry `rank + 1` offsets; the
/// trailing one belongs to the element-size axis and is always zero, so it is
/// dropped.
pub fn read_chunk_index(ctx: Ctx<'_>, address: u64, rank: usize) -> Result<Vec<ChunkRecord>> {
    let mut out = Vec::new();
    let mut visited = HashSet::new();
    let mut stack = vec![address];

    while let Some(node_address) = stack.pop() {
        if out.len() > MAX_NODES || visited.len() > MAX_NODES {
            return Err(Error::malformed(
                "chunk B-tree is implausibly large; the file may be cyclic",
            ));
        }
        if !visited.insert(node_address) {
            continue;
        }

        let node = Node::read(ctx, node_address, rank)?;
        if node.node_type != NODE_TYPE_CHUNK {
            return Err(Error::malformed(format!(
                "expected a chunk B-tree node, found node type {}",
                node.node_type
            )));
        }

        if node.level == 0 {
            // Leaf: each child address is a chunk, described by the key that
            // precedes it.
            for (key, child) in node.chunk_keys.iter().zip(node.children.iter()) {
                out.push(ChunkRecord {
                    size: key.size,
                    filter_mask: key.filter_mask,
                    offset: key.offset.clone(),
                    address: *child,
                });
            }
        } else {
            stack.extend(node.children.iter().copied());
        }
    }

    // A deterministic order keeps reads predictable and makes tests stable.
    out.sort_by(|a, b| a.offset.cmp(&b.offset));
    Ok(out)
}

/// Walk a type 0 B-tree and return the addresses of its symbol table nodes.
pub fn read_group_node_addresses(ctx: Ctx<'_>, address: u64) -> Result<Vec<u64>> {
    let mut out = Vec::new();
    let mut visited = HashSet::new();
    let mut stack = vec![address];

    while let Some(node_address) = stack.pop() {
        if visited.len() > MAX_NODES {
            return Err(Error::malformed(
                "group B-tree is implausibly large; the file may be cyclic",
            ));
        }
        if !visited.insert(node_address) {
            continue;
        }

        let node = Node::read(ctx, node_address, 0)?;
        if node.node_type != NODE_TYPE_GROUP {
            return Err(Error::malformed(format!(
                "expected a group B-tree node, found node type {}",
                node.node_type
            )));
        }

        if node.level == 0 {
            out.extend(node.children.iter().copied());
        } else {
            stack.extend(node.children.iter().copied());
        }
    }

    Ok(out)
}

/// A chunk B-tree key.
#[derive(Debug, Clone)]
struct ChunkKey {
    size: u32,
    filter_mask: u32,
    offset: Vec<u64>,
}

/// One B-tree node, with its keys and child pointers.
#[derive(Debug)]
struct Node {
    node_type: u8,
    level: u8,
    children: Vec<u64>,
    chunk_keys: Vec<ChunkKey>,
}

impl Node {
    fn read(ctx: Ctx<'_>, address: u64, rank: usize) -> Result<Self> {
        let sizes = ctx.sizes();
        // Header: signature, type, level, entries used, two sibling addresses.
        let header_len = 8 + 2 * sizes.offset as usize;
        let header = ctx.read(address, header_len)?;
        let mut cur = Cursor::new(&header);

        let sig = cur.take(4)?;
        if sig != TREE_SIGNATURE {
            return Err(Error::malformed(format!(
                "B-tree node at {address} is missing its TREE signature"
            )));
        }
        let node_type = cur.u8()?;
        let level = cur.u8()?;
        let entries_used = cur.u16()? as usize;
        let _left = cur.address(sizes)?;
        let _right = cur.address(sizes)?;

        // A node stores entries_used children, each preceded by a key, and one
        // trailing key. Read the whole body in one go.
        let key_len = match node_type {
            NODE_TYPE_GROUP => sizes.length as usize,
            NODE_TYPE_CHUNK => 4 + 4 + 8 * (rank + 1),
            other => {
                return Err(Error::malformed(format!("B-tree node type {other}")));
            }
        };
        let body_len = (entries_used + 1) * key_len + entries_used * sizes.offset as usize;
        let body = ctx.read(address + header_len as u64, body_len)?;
        let mut cur = Cursor::new(&body);

        let mut children = Vec::with_capacity(entries_used);
        let mut chunk_keys = Vec::with_capacity(entries_used);

        for _ in 0..entries_used {
            match node_type {
                NODE_TYPE_GROUP => {
                    cur.skip(key_len)?;
                }
                _ => {
                    chunk_keys.push(parse_chunk_key(&mut cur, rank)?);
                }
            }
            children.push(cur.address_required(sizes, "B-tree child")?);
        }
        // The trailing key bounds the last child; it names no child of its own.

        Ok(Self {
            node_type,
            level,
            children,
            chunk_keys,
        })
    }
}

fn parse_chunk_key(cur: &mut Cursor<'_>, rank: usize) -> Result<ChunkKey> {
    let size = cur.u32()?;
    let filter_mask = cur.u32()?;
    let mut offset = Vec::with_capacity(rank);
    for _ in 0..rank {
        offset.push(cur.u64()?);
    }
    // The final offset belongs to the element-size axis and is always zero.
    cur.skip(8)?;
    Ok(ChunkKey {
        size,
        filter_mask,
        offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Layout;
    use crate::source::{ByteSource, FileSource};
    use crate::superblock::Superblock;
    use crate::ObjectHeader;

    const LEGACY_FILE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_files/legacy_v1_objheader.h5"
    );

    /// The legacy fixture's `chunked_i32` is 40x6 with 7x4 chunks, so the grid
    /// is ceil(40/7) x ceil(6/4) = 6 x 2 = 12 chunks.
    #[test]
    fn reads_the_chunk_index_of_the_legacy_fixture() {
        let src = FileSource::open(LEGACY_FILE).unwrap();
        let sb = Superblock::read(&src).unwrap();
        let ctx = Ctx::new(&src, &sb);

        let dataset = find_dataset(ctx, &sb, "chunked_i32");
        let layout = dataset.layout(sb.sizes).unwrap().unwrap();

        let (address, chunk_dims) = match &layout {
            Layout::Chunked {
                address,
                chunk_dims,
                ..
            } => (address.unwrap(), chunk_dims.clone()),
            other => panic!("expected a chunked layout, got {other:?}"),
        };
        assert_eq!(chunk_dims, vec![7, 4]);

        let chunks = read_chunk_index(ctx, address, chunk_dims.len()).unwrap();
        assert_eq!(
            chunks.len(),
            12,
            "a 40x6 dataset in 7x4 chunks is a 6x2 grid"
        );

        // Offsets must land on chunk boundaries and cover the grid exactly once.
        let mut seen: Vec<Vec<u64>> = chunks.iter().map(|c| c.offset.clone()).collect();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 12, "every chunk offset must be distinct");

        for c in &chunks {
            assert_eq!(c.offset.len(), 2, "offsets drop the element-size axis");
            assert_eq!(c.offset[0] % 7, 0, "offset must sit on a chunk boundary");
            assert_eq!(c.offset[1] % 4, 0);
            assert!(c.offset[0] < 40 && c.offset[1] < 6);
            assert!(c.size > 0, "a written chunk has bytes");
            assert!(
                c.address > 0 && c.address < src.size(),
                "chunk address must be inside the file"
            );
        }
    }

    /// Chunks are returned in a deterministic order so downstream reads and
    /// tests do not depend on B-tree walk order.
    #[test]
    fn the_chunk_index_is_sorted_by_offset() {
        let src = FileSource::open(LEGACY_FILE).unwrap();
        let sb = Superblock::read(&src).unwrap();
        let ctx = Ctx::new(&src, &sb);
        let dataset = find_dataset(ctx, &sb, "chunked_i32");
        let layout = dataset.layout(sb.sizes).unwrap().unwrap();
        let (address, rank) = match &layout {
            Layout::Chunked {
                address,
                chunk_dims,
                ..
            } => (address.unwrap(), chunk_dims.len()),
            other => panic!("expected chunked, got {other:?}"),
        };

        let chunks = read_chunk_index(ctx, address, rank).unwrap();
        let mut sorted = chunks.clone();
        sorted.sort_by(|a, b| a.offset.cmp(&b.offset));
        assert_eq!(chunks, sorted);
    }

    #[test]
    fn walks_the_group_btree_of_the_legacy_fixture() {
        let src = FileSource::open(LEGACY_FILE).unwrap();
        let sb = Superblock::read(&src).unwrap();
        let ctx = Ctx::new(&src, &sb);

        let root = ObjectHeader::read(ctx, sb.root_object_header_address().unwrap()).unwrap();
        let st = root.symbol_table(sb.sizes).unwrap().unwrap();
        let nodes = read_group_node_addresses(ctx, st.btree_address).unwrap();

        assert!(!nodes.is_empty(), "the root group has children");
        for addr in nodes {
            let buf = src.read_vec(addr, 4).unwrap();
            assert_eq!(&buf, b"SNOD", "group B-tree leaves point at symbol tables");
        }
    }

    #[test]
    fn rejects_a_node_without_a_signature() {
        let src = FileSource::open(LEGACY_FILE).unwrap();
        let sb = Superblock::read(&src).unwrap();
        let ctx = Ctx::new(&src, &sb);
        // Address 0 is the superblock, not a B-tree node.
        assert!(read_chunk_index(ctx, 0, 2).is_err());
    }

    /// Find a dataset by walking the root group's symbol table directly. Group
    /// traversal proper arrives in a later module; this keeps the B-tree tests
    /// self-contained.
    fn find_dataset(ctx: Ctx<'_>, sb: &Superblock, want: &str) -> ObjectHeader {
        use crate::heap::LocalHeap;
        use crate::symbol_table::SymbolTableNode;

        let root = ObjectHeader::read(ctx, sb.root_object_header_address().unwrap()).unwrap();
        let st = root.symbol_table(sb.sizes).unwrap().unwrap();
        let heap = LocalHeap::read(ctx, st.local_heap_address).unwrap();
        let nodes = read_group_node_addresses(ctx, st.btree_address).unwrap();

        for node_address in nodes {
            // A symbol table node is small; read a generous window.
            let raw = ctx.read_upto(node_address, 8 + 64 * 40).unwrap();
            let node = SymbolTableNode::parse(&raw, sb.sizes).unwrap();
            for entry in node.entries {
                let name = heap.name_at(entry.link_name_offset).unwrap();
                if name == want {
                    return ObjectHeader::read(ctx, entry.object_header_address.unwrap()).unwrap();
                }
            }
        }
        panic!("dataset {want} not found");
    }
}
