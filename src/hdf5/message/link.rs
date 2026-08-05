//! Messages that describe how a group holds its children.
//!
//! A new-style group stores each child either as an inline [`Link`] message
//! ("compact" storage) or, once there are enough of them, in a fractal heap
//! indexed by a version 2 B-tree ("dense" storage). [`LinkInfo`] says which.
//!
//! An old-style group instead carries a [`SymbolTable`] message pointing at a
//! version 1 B-tree and a local heap.

use crate::cursor::{Cursor, Sizes};
use crate::error::{Error, Result};

/// Where a link points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkTarget {
    /// A hard link: the address of the target's object header.
    Hard {
        /// Object header address.
        address: u64,
    },
    /// A soft link: a path to resolve later.
    Soft {
        /// The stored path.
        path: String,
    },
    /// A link into another file.
    External {
        /// Raw payload; the first byte is a version and flags pair.
        raw: Vec<u8>,
    },
    /// A link type this reader does not understand.
    Unknown {
        /// The link type code.
        kind: u8,
        /// Raw payload.
        raw: Vec<u8>,
    },
}

/// One named child of a group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    /// The child's name within this group.
    pub name: String,
    /// Where the link points.
    pub target: LinkTarget,
    /// Creation order, when the group tracks it.
    pub creation_order: Option<u64>,
}

impl Link {
    /// Parse a link message body.
    pub fn parse(body: &[u8], sizes: Sizes) -> Result<Self> {
        let mut cur = Cursor::new(body);
        Self::parse_at(&mut cur, sizes)
    }

    /// Parse a link at the cursor. Fractal heap records hold links back to back,
    /// so parsing has to be resumable.
    pub fn parse_at(cur: &mut Cursor<'_>, sizes: Sizes) -> Result<Self> {
        let version = cur.u8()?;
        if version != 1 {
            return Err(Error::unsupported(format!("link message version {version}")));
        }
        let flags = cur.u8()?;

        // Bits 0 and 1 select the width of the name-length field.
        let name_len_width = 1u8 << (flags & 0x03);
        let has_link_type = flags & 0x08 != 0;
        let has_creation_order = flags & 0x04 != 0;
        let has_charset = flags & 0x10 != 0;

        let link_type = if has_link_type { cur.u8()? } else { 0 };
        let creation_order = if has_creation_order {
            Some(cur.u64()?)
        } else {
            None
        };
        if has_charset {
            cur.skip(1)?;
        }

        let name_len = cur.uint(name_len_width)? as usize;
        let raw_name = cur.take(name_len)?;
        let name = String::from_utf8_lossy(raw_name).into_owned();

        let target = match link_type {
            0 => LinkTarget::Hard {
                address: cur.address_required(sizes, "hard link target")?,
            },
            1 => {
                let len = cur.u16()? as usize;
                let raw = cur.take(len)?;
                LinkTarget::Soft {
                    path: String::from_utf8_lossy(raw).into_owned(),
                }
            }
            64 => {
                let len = cur.u16()? as usize;
                LinkTarget::External {
                    raw: cur.take(len)?.to_vec(),
                }
            }
            kind => {
                let len = cur.u16()? as usize;
                LinkTarget::Unknown {
                    kind,
                    raw: cur.take(len)?.to_vec(),
                }
            }
        };

        Ok(Self {
            name,
            target,
            creation_order,
        })
    }

    /// The object header address, when this is a hard link.
    pub fn hard_address(&self) -> Option<u64> {
        match &self.target {
            LinkTarget::Hard { address } => Some(*address),
            _ => None,
        }
    }
}

/// How a new-style group stores its links.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkInfo {
    /// Highest creation index used so far, when creation order is tracked.
    pub max_creation_index: Option<u64>,
    /// Fractal heap holding the links, when storage is dense.
    pub fractal_heap_address: Option<u64>,
    /// Version 2 B-tree indexing the heap by name, when storage is dense.
    pub name_btree_address: Option<u64>,
    /// Version 2 B-tree indexing the heap by creation order.
    pub creation_order_btree_address: Option<u64>,
}

impl LinkInfo {
    /// Parse a link info message body.
    pub fn parse(body: &[u8], sizes: Sizes) -> Result<Self> {
        let mut cur = Cursor::new(body);
        let version = cur.u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "link info message version {version}"
            )));
        }
        let flags = cur.u8()?;

        let max_creation_index = if flags & 0x01 != 0 {
            Some(cur.u64()?)
        } else {
            None
        };
        let fractal_heap_address = cur.address(sizes)?;
        let name_btree_address = cur.address(sizes)?;
        let creation_order_btree_address = if flags & 0x02 != 0 {
            cur.address(sizes)?
        } else {
            None
        };

        Ok(Self {
            max_creation_index,
            fractal_heap_address,
            name_btree_address,
            creation_order_btree_address,
        })
    }

    /// Whether the group's links live in a fractal heap rather than inline.
    pub fn is_dense(&self) -> bool {
        self.fractal_heap_address.is_some()
    }
}

/// An old-style group: a version 1 B-tree plus a local heap of names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolTable {
    /// Address of the version 1 B-tree over symbol table nodes.
    pub btree_address: u64,
    /// Address of the local heap holding link names.
    pub local_heap_address: u64,
}

impl SymbolTable {
    /// Parse a symbol table message body.
    pub fn parse(body: &[u8], sizes: Sizes) -> Result<Self> {
        let mut cur = Cursor::new(body);
        Ok(Self {
            btree_address: cur.address_required(sizes, "group B-tree")?,
            local_heap_address: cur.address_required(sizes, "group local heap")?,
        })
    }
}

/// How an object stores its attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeInfo {
    /// Fractal heap holding the attributes, when storage is dense.
    pub fractal_heap_address: Option<u64>,
    /// Version 2 B-tree indexing the heap by name.
    pub name_btree_address: Option<u64>,
    /// Version 2 B-tree indexing the heap by creation order.
    pub creation_order_btree_address: Option<u64>,
}

impl AttributeInfo {
    /// Parse an attribute info message body.
    pub fn parse(body: &[u8], sizes: Sizes) -> Result<Self> {
        let mut cur = Cursor::new(body);
        let version = cur.u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "attribute info message version {version}"
            )));
        }
        let flags = cur.u8()?;
        if flags & 0x01 != 0 {
            cur.skip(2)?; // maximum creation index
        }
        let fractal_heap_address = cur.address(sizes)?;
        let name_btree_address = cur.address(sizes)?;
        let creation_order_btree_address = if flags & 0x02 != 0 {
            cur.address(sizes)?
        } else {
            None
        };

        Ok(Self {
            fractal_heap_address,
            name_btree_address,
            creation_order_btree_address,
        })
    }

    /// Whether the object's attributes live in a fractal heap.
    pub fn is_dense(&self) -> bool {
        self.fractal_heap_address.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_minimal_hard_link() {
        // Flags 0: one-byte name length, no type, no creation order.
        let mut d = vec![1u8, 0x00];
        d.push(3); // name length
        d.extend_from_slice(b"lat");
        d.extend_from_slice(&0x500u64.to_le_bytes());

        let l = Link::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(l.name, "lat");
        assert_eq!(l.hard_address(), Some(0x500));
        assert_eq!(l.creation_order, None);
    }

    #[test]
    fn parses_a_link_with_creation_order_and_charset() {
        // Flags: 2-byte name length (1), creation order (4), type (8), charset (16).
        let flags = 0x01 | 0x04 | 0x08 | 0x10;
        let mut d = vec![1u8, flags];
        d.push(0); // link type: hard
        d.extend_from_slice(&7u64.to_le_bytes()); // creation order
        d.push(0); // charset
        d.extend_from_slice(&4u16.to_le_bytes()); // name length
        d.extend_from_slice(b"time");
        d.extend_from_slice(&0x900u64.to_le_bytes());

        let l = Link::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(l.name, "time");
        assert_eq!(l.creation_order, Some(7));
        assert_eq!(l.hard_address(), Some(0x900));
    }

    #[test]
    fn parses_a_soft_link() {
        let mut d = vec![1u8, 0x08];
        d.push(1); // soft
        d.push(2); // name length
        d.extend_from_slice(b"ln");
        d.extend_from_slice(&5u16.to_le_bytes());
        d.extend_from_slice(b"/data");

        let l = Link::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(
            l.target,
            LinkTarget::Soft {
                path: "/data".into()
            }
        );
        assert_eq!(l.hard_address(), None);
    }

    #[test]
    fn parses_dense_link_info() {
        let mut d = vec![0u8, 0x00];
        d.extend_from_slice(&0x100u64.to_le_bytes()); // fractal heap
        d.extend_from_slice(&0x200u64.to_le_bytes()); // name btree
        let li = LinkInfo::parse(&d, Sizes::EIGHT).unwrap();
        assert!(li.is_dense());
        assert_eq!(li.fractal_heap_address, Some(0x100));
        assert_eq!(li.name_btree_address, Some(0x200));
    }

    #[test]
    fn compact_link_info_has_no_heap() {
        let mut d = vec![0u8, 0x00];
        d.extend_from_slice(&u64::MAX.to_le_bytes());
        d.extend_from_slice(&u64::MAX.to_le_bytes());
        let li = LinkInfo::parse(&d, Sizes::EIGHT).unwrap();
        assert!(!li.is_dense(), "compact groups store links inline");
    }

    #[test]
    fn link_info_reads_the_creation_order_btree_when_flagged() {
        let mut d = vec![0u8, 0x03];
        d.extend_from_slice(&11u64.to_le_bytes()); // max creation index
        d.extend_from_slice(&0x100u64.to_le_bytes());
        d.extend_from_slice(&0x200u64.to_le_bytes());
        d.extend_from_slice(&0x300u64.to_le_bytes());
        let li = LinkInfo::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(li.max_creation_index, Some(11));
        assert_eq!(li.creation_order_btree_address, Some(0x300));
    }

    #[test]
    fn parses_a_symbol_table_message() {
        let mut d = Vec::new();
        d.extend_from_slice(&0x40u64.to_le_bytes());
        d.extend_from_slice(&0x80u64.to_le_bytes());
        let st = SymbolTable::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(st.btree_address, 0x40);
        assert_eq!(st.local_heap_address, 0x80);
    }

    #[test]
    fn parses_dense_attribute_info() {
        let mut d = vec![0u8, 0x00];
        d.extend_from_slice(&0xAAu64.to_le_bytes());
        d.extend_from_slice(&0xBBu64.to_le_bytes());
        let ai = AttributeInfo::parse(&d, Sizes::EIGHT).unwrap();
        assert!(ai.is_dense());
        assert_eq!(ai.fractal_heap_address, Some(0xAA));
    }

    #[test]
    fn attribute_info_skips_the_max_creation_index_when_flagged() {
        let mut d = vec![0u8, 0x01];
        d.extend_from_slice(&9u16.to_le_bytes()); // max creation index
        d.extend_from_slice(&0xAAu64.to_le_bytes());
        d.extend_from_slice(&0xBBu64.to_le_bytes());
        let ai = AttributeInfo::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(ai.fractal_heap_address, Some(0xAA));
        assert_eq!(ai.name_btree_address, Some(0xBB));
    }

    #[test]
    fn rejects_an_unknown_link_version() {
        let err = Link::parse(&[4u8, 0, 0], Sizes::EIGHT).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }
}
