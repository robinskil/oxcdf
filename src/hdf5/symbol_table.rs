//! Symbol table entries and nodes.
//!
//! This is the "old style" group storage, used whenever the superblock is
//! version 0 or 1. A group object header carries a symbol table message that
//! points at a version 1 B-tree and a local heap. The B-tree's leaves are
//! symbol table nodes, and each node holds entries that name a child object and
//! give its object header address.
//!
//! netcdf-c produces this layout by default, so it is not a legacy path.

use crate::cursor::{Cursor, Sizes};
use crate::error::{Error, Result};

/// Signature at the front of a symbol table node.
pub const SNOD_SIGNATURE: &[u8; 4] = b"SNOD";

/// What the scratch-pad of a symbol table entry caches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CachedInfo {
    /// Nothing cached.
    None,
    /// The child is a group. The B-tree and local heap addresses are cached so
    /// a reader can descend without opening the child's object header first.
    Group {
        /// Address of the version 1 B-tree holding the group's entries.
        btree_address: u64,
        /// Address of the local heap holding the group's link names.
        heap_address: u64,
    },
    /// The child is a soft link. The value is at this offset in the local heap.
    SymbolicLink {
        /// Offset of the link target string within the local heap.
        link_value_offset: u32,
    },
}

/// One entry of a symbol table: a named child of a group.
#[derive(Debug, Clone)]
pub struct SymbolTableEntry {
    /// Offset of this entry's name within the group's local heap.
    pub link_name_offset: u64,
    /// Address of the child's object header. `None` for a soft link, which has
    /// no object of its own.
    pub object_header_address: Option<u64>,
    /// Contents of the scratch-pad.
    pub cached: CachedInfo,
}

impl SymbolTableEntry {
    /// Size of one entry on disk, given the file's address width.
    pub fn encoded_len(sizes: Sizes) -> usize {
        // name offset + header address + cache type + reserved + scratch pad
        (sizes.offset as usize) * 2 + 4 + 4 + 16
    }

    /// Parse one entry at the cursor.
    pub fn parse(cur: &mut Cursor<'_>, sizes: Sizes) -> Result<Self> {
        let link_name_offset = cur.uint(sizes.offset)?;
        let object_header_address = cur.address(sizes)?;
        let cache_type = cur.u32()?;
        cur.skip(4)?; // reserved

        // The scratch-pad is always 16 bytes, whatever the cache type uses.
        let scratch = cur.take(16)?;
        let mut sp = Cursor::new(scratch);

        let cached = match cache_type {
            0 => CachedInfo::None,
            1 => {
                let btree_address = sp.address_required(sizes, "cached group B-tree")?;
                let heap_address = sp.address_required(sizes, "cached group local heap")?;
                CachedInfo::Group {
                    btree_address,
                    heap_address,
                }
            }
            2 => CachedInfo::SymbolicLink {
                link_value_offset: sp.u32()?,
            },
            other => {
                // Unknown cache types are safe to ignore: the scratch-pad is
                // only a shortcut, and the object header address is definitive.
                let _ = other;
                CachedInfo::None
            }
        };

        Ok(Self {
            link_name_offset,
            object_header_address,
            cached,
        })
    }
}

/// A leaf of a group's version 1 B-tree: a run of symbol table entries.
#[derive(Debug, Clone)]
pub struct SymbolTableNode {
    /// The entries this node holds, in the B-tree's sorted-by-name order.
    pub entries: Vec<SymbolTableEntry>,
}

impl SymbolTableNode {
    /// Parse a symbol table node from a buffer that starts at its signature.
    pub fn parse(buf: &[u8], sizes: Sizes) -> Result<Self> {
        let mut cur = Cursor::new(buf);
        let sig = cur.take(4)?;
        if sig != SNOD_SIGNATURE {
            return Err(Error::malformed(format!(
                "expected SNOD signature, found {sig:?}"
            )));
        }
        let version = cur.u8()?;
        if version != 1 {
            return Err(Error::unsupported(format!(
                "symbol table node version {version}"
            )));
        }
        cur.skip(1)?; // reserved
        let count = cur.u16()? as usize;

        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            entries.push(SymbolTableEntry::parse(&mut cur, sizes)?);
        }
        Ok(Self { entries })
    }

    /// Size on disk of a node holding `capacity` entries, used to size reads.
    pub fn encoded_len(capacity: usize, sizes: Sizes) -> usize {
        8 + capacity * SymbolTableEntry::encoded_len(sizes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry_bytes(name_offset: u64, header: u64, cache_type: u32, scratch: [u8; 16]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&name_offset.to_le_bytes());
        v.extend_from_slice(&header.to_le_bytes());
        v.extend_from_slice(&cache_type.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&scratch);
        v
    }

    #[test]
    fn entry_encoded_len_is_forty_bytes_for_eight_byte_addresses() {
        assert_eq!(SymbolTableEntry::encoded_len(Sizes::EIGHT), 40);
    }

    #[test]
    fn parses_an_entry_with_no_cache() {
        let bytes = entry_bytes(8, 0x100, 0, [0u8; 16]);
        let mut cur = Cursor::new(&bytes);
        let e = SymbolTableEntry::parse(&mut cur, Sizes::EIGHT).unwrap();
        assert_eq!(e.link_name_offset, 8);
        assert_eq!(e.object_header_address, Some(0x100));
        assert_eq!(e.cached, CachedInfo::None);
        assert_eq!(cur.pos(), 40, "must consume the whole entry");
    }

    #[test]
    fn parses_an_entry_caching_group_addresses() {
        let mut scratch = [0u8; 16];
        scratch[..8].copy_from_slice(&0xAAu64.to_le_bytes());
        scratch[8..].copy_from_slice(&0xBBu64.to_le_bytes());
        let bytes = entry_bytes(0, 0x200, 1, scratch);
        let mut cur = Cursor::new(&bytes);
        let e = SymbolTableEntry::parse(&mut cur, Sizes::EIGHT).unwrap();
        assert_eq!(
            e.cached,
            CachedInfo::Group {
                btree_address: 0xAA,
                heap_address: 0xBB
            }
        );
    }

    #[test]
    fn parses_a_symbolic_link_entry() {
        let mut scratch = [0u8; 16];
        scratch[..4].copy_from_slice(&42u32.to_le_bytes());
        let bytes = entry_bytes(0, 0x300, 2, scratch);
        let mut cur = Cursor::new(&bytes);
        let e = SymbolTableEntry::parse(&mut cur, Sizes::EIGHT).unwrap();
        assert_eq!(
            e.cached,
            CachedInfo::SymbolicLink {
                link_value_offset: 42
            }
        );
    }

    #[test]
    fn parses_a_node_with_two_entries() {
        let mut buf = Vec::new();
        buf.extend_from_slice(SNOD_SIGNATURE);
        buf.push(1); // version
        buf.push(0); // reserved
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend(entry_bytes(0, 0x100, 0, [0u8; 16]));
        buf.extend(entry_bytes(8, 0x200, 0, [0u8; 16]));

        let node = SymbolTableNode::parse(&buf, Sizes::EIGHT).unwrap();
        assert_eq!(node.entries.len(), 2);
        assert_eq!(node.entries[1].object_header_address, Some(0x200));
    }

    #[test]
    fn rejects_a_node_with_a_wrong_signature() {
        let mut buf = b"XXXX".to_vec();
        buf.extend_from_slice(&[1, 0, 0, 0]);
        assert!(SymbolTableNode::parse(&buf, Sizes::EIGHT).is_err());
    }

    #[test]
    fn rejects_an_unknown_node_version() {
        let mut buf = SNOD_SIGNATURE.to_vec();
        buf.push(9); // version
        buf.extend_from_slice(&[0, 0, 0]);
        let err = SymbolTableNode::parse(&buf, Sizes::EIGHT).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }
}
