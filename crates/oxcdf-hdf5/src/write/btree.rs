//! The version 1 B-tree that indexes the chunks of a dataset.
//!
//! The format is the inverse of [`crate::btree1::read_chunk_index`]. The HDF5
//! library can add chunks to this tree later.

use super::alloc::Allocator;
use super::encode::Encoder;
use super::sink::ByteSink;
use super::to_u64;
use crate::btree1::{NODE_TYPE_CHUNK, TREE_SIGNATURE};
use crate::error::{Error, Result};

/// Largest number of children in one node: two times the default K of 32.
///
/// A version 2 superblock has no field for K, so each reader uses the default.
const MAX_CHILDREN: usize = 64;

/// Length of the node header: signature, type, level, count and two siblings.
const NODE_HEADER_BYTES: usize = 24;

/// A key of the tree: the stored size of a chunk and the position of its first element.
#[derive(Debug, Clone)]
struct Key {
    size: u32,
    origin: Vec<u64>,
}

/// One node of a level that has its file address.
#[derive(Debug)]
struct Node {
    address: u64,
    /// The key and the address of each child.
    children: Vec<(Key, u64)>,
    /// The key after the last child. Each child key is below it.
    end: Key,
}

/// Write the chunk index for `chunks` and return the address of the root node.
///
/// `chunks` holds the position of the first element, the address and the
/// stored size of each chunk, in ascending order. No chunks gives `None`.
pub fn write_chunk_index<S: ByteSink>(
    sink: &mut S,
    alloc: &mut Allocator,
    chunk_dims: &[u64],
    chunks: &[(Vec<u64>, u64, u32)],
) -> Result<Option<u64>> {
    let Some((last, _, _)) = chunks.last() else {
        return Ok(None);
    };
    let rank = chunk_dims.len();
    // A key holds the size, the filter mask, and one offset more than the rank.
    let key_bytes = 8 + 8 * (rank + 1);
    // The library reads a full node, so each node has the space for all children.
    let node_bytes = NODE_HEADER_BYTES + MAX_CHILDREN * (key_bytes + 8) + key_bytes;

    // The key after the last chunk lies one chunk past it on each axis.
    let mut end = Key {
        size: 0,
        origin: Vec::with_capacity(rank),
    };
    for (&at, &dim) in last.iter().zip(chunk_dims) {
        end.origin.push(at.checked_add(dim).ok_or_else(overflow)?);
    }

    let mut children: Vec<(Key, u64)> = chunks
        .iter()
        .map(|(origin, address, size)| {
            let key = Key {
                size: *size,
                origin: origin.clone(),
            };
            (key, *address)
        })
        .collect();
    let mut level = 0u8;
    loop {
        let mut nodes = Vec::with_capacity(children.len().div_ceil(MAX_CHILDREN));
        let mut rest = children.into_iter();
        loop {
            let group: Vec<(Key, u64)> = rest.by_ref().take(MAX_CHILDREN).collect();
            if group.is_empty() {
                break;
            }
            nodes.push(Node {
                address: alloc.alloc_metadata(to_u64(node_bytes)?)?,
                children: group,
                end: end.clone(),
            });
        }
        // The end key of a node is the first key of the next node.
        for i in 1..nodes.len() {
            if let Some((first, _)) = nodes[i].children.first() {
                nodes[i - 1].end = first.clone();
            }
        }
        for (i, node) in nodes.iter().enumerate() {
            let left = i.checked_sub(1).map(|left| nodes[left].address);
            let right = nodes.get(i + 1).map(|right| right.address);
            let used = NODE_HEADER_BYTES + node.children.len() * (key_bytes + 8) + key_bytes;
            let mut e = Encoder::with_capacity(used);
            e.bytes(TREE_SIGNATURE);
            e.u8(NODE_TYPE_CHUNK);
            e.u8(level);
            // A node holds at most MAX_CHILDREN children, so the count fits.
            e.u16(u16::try_from(node.children.len()).unwrap_or(u16::MAX));
            e.address(left);
            e.address(right);
            for (key, child) in &node.children {
                encode_key(&mut e, key);
                e.u64(*child);
            }
            encode_key(&mut e, &node.end);
            sink.write_all_at(node.address, e.as_slice())?;
        }
        if let [root] = nodes.as_slice() {
            return Ok(Some(root.address));
        }
        children = nodes
            .iter()
            .filter_map(|node| Some((node.children.first()?.0.clone(), node.address)))
            .collect();
        level = level
            .checked_add(1)
            .ok_or_else(|| Error::unsupported("a chunk index with more than 255 levels"))?;
    }
}

/// Append one key: the size, an empty filter mask, each offset, and the offset of the element axis.
fn encode_key(e: &mut Encoder, key: &Key) {
    e.u32(key.size);
    e.u32(0);
    for &at in &key.origin {
        e.u64(at);
    }
    e.u64(0);
}

/// The error for a chunk position past `u64::MAX`.
fn overflow() -> Error {
    Error::bad_request("a chunk position passes u64::MAX")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::btree1::read_chunk_index;
    use crate::context::Ctx;
    use crate::source::MemorySource;
    use crate::superblock::Superblock;
    use crate::write::header::encode_superblock;
    use crate::write::MemorySink;

    /// The position of the first element, the address and the size of one chunk.
    type Chunk = (Vec<u64>, u64, u32);

    fn round_trip(count: u64) -> (Vec<Chunk>, Vec<u8>, Option<u64>) {
        let chunks: Vec<Chunk> = (0..count)
            .map(|i| {
                (
                    vec![i / 100 * 4, i % 100 * 5],
                    1_000_000 + i * 160,
                    100 + i as u32,
                )
            })
            .collect();
        let mut sink = MemorySink::new();
        let mut alloc = Allocator::new(0, 0);
        let root = write_chunk_index(&mut sink, &mut alloc, &[4, 5], &chunks).unwrap();
        sink.set_len(alloc.end_of_file()).unwrap();
        (chunks, sink.into_inner(), root)
    }

    #[test]
    fn no_chunks_give_no_index() {
        assert_eq!(round_trip(0).2, None);
    }

    #[test]
    fn each_chunk_reads_back_from_one_two_and_three_levels() {
        for count in [1, 64, 65, 4096, 4097, 9000] {
            let (chunks, file, root) = round_trip(count);
            let superblock =
                Superblock::read(&MemorySource::new(encode_superblock(48, 4096))).unwrap();
            let source = MemorySource::new(file);
            let ctx = Ctx::new(&source, &superblock);
            let records = read_chunk_index(ctx, root.unwrap(), 2).unwrap();
            assert_eq!(records.len(), chunks.len(), "{count}");
            for (record, (origin, address, size)) in records.iter().zip(&chunks) {
                assert_eq!(&record.offset, origin);
                assert_eq!(record.address, *address);
                assert_eq!((record.size, record.filter_mask), (*size, 0));
            }
        }
    }

    #[test]
    fn the_root_of_many_chunks_is_an_internal_node() {
        let (_, file, root) = round_trip(65);
        let root = root.unwrap() as usize;
        assert_eq!(&file[root..root + 4], b"TREE");
        // Type 1, level 1, two children.
        assert_eq!(file[root + 4..root + 8], [1, 1, 2, 0]);
    }
}
