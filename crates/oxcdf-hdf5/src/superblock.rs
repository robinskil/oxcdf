//! The HDF5 superblock.
//!
//! Versions 0 and 1 describe the root group with a symbol table entry and carry
//! the B-tree K values. Versions 2 and 3 drop all of that and simply name the
//! root group's object header address, then checksum themselves.
//!
//! Both shapes appear in ordinary netcdf-c output, so both are required.

use crate::checksum;
use crate::cursor::{Cursor, Sizes};
use crate::error::{Error, Result};
use crate::source::ByteSource;
use crate::symbol_table::SymbolTableEntry;
use crate::HDF5_SIGNATURE;

/// Default number of entries in an indexed-storage B-tree node. Versions 0 and
/// 2+ do not store the value; the library's own default is 32.
pub const DEFAULT_ISTORE_K: u16 = 32;

/// How the superblock names the root group.
#[derive(Debug, Clone)]
pub enum RootGroup {
    /// Versions 0 and 1: a full symbol table entry.
    SymbolTable(Box<SymbolTableEntry>),
    /// Versions 2 and 3: the object header address, directly.
    ObjectHeader(u64),
}

/// A parsed superblock.
#[derive(Debug, Clone)]
pub struct Superblock {
    /// Superblock version, 0 through 3.
    pub version: u8,
    /// Width of addresses and lengths in this file.
    pub sizes: Sizes,
    /// Absolute file offset where the superblock signature was found.
    ///
    /// A user block may precede it, in which case this is not zero.
    pub superblock_offset: u64,
    /// Base address that every stored address is relative to.
    pub base_address: u64,
    /// Address recorded as the end of the HDF5 data.
    pub end_of_file_address: u64,
    /// Address of the superblock extension object header, when present.
    pub extension_address: Option<u64>,
    /// How the root group is named.
    pub root: RootGroup,
    /// Rank of group B-tree internal nodes. Versions 0 and 1 only.
    pub group_internal_node_k: u16,
    /// Rank of group B-tree leaf nodes. Versions 0 and 1 only.
    pub group_leaf_node_k: u16,
    /// Rank of chunk B-tree internal nodes.
    ///
    /// Only version 1 records this. Every other version uses
    /// [`DEFAULT_ISTORE_K`], which is what the library writes.
    pub indexed_storage_internal_node_k: u16,
}

impl Superblock {
    /// Find and parse the superblock.
    ///
    /// Probes offset 0 then each power-of-two user-block size, which is how the
    /// format allows an application to prepend its own bytes.
    pub fn read(source: &dyn ByteSource) -> Result<Self> {
        let mut offset = 0u64;
        loop {
            if offset + 8 > source.size() {
                return Err(Error::malformed(
                    "no HDF5 superblock signature found at any user-block offset",
                ));
            }
            let mut magic = [0u8; 8];
            source.read_exact_at(offset, &mut magic)?;
            if magic == HDF5_SIGNATURE {
                return Self::parse_at(source, offset);
            }
            offset = if offset == 0 { 512 } else { offset * 2 };
        }
    }

    /// Parse a superblock whose signature starts at `offset`.
    pub fn parse_at(source: &dyn ByteSource, offset: u64) -> Result<Self> {
        // The version byte decides the layout and therefore the length. Read a
        // generous fixed block first; every superblock variant fits in 96 bytes
        // once addresses are 8 bytes wide.
        let avail = (source.size() - offset).min(96) as usize;
        let buf = source.read_vec(offset, avail)?;
        let mut cur = Cursor::new(&buf);

        cur.skip(8)?; // signature, already matched
        let version = cur.u8()?;

        match version {
            0 | 1 => Self::parse_v0_v1(&buf, &mut cur, version, offset),
            2 | 3 => Self::parse_v2_v3(&buf, &mut cur, version, offset),
            other => Err(Error::unsupported(format!("superblock version {other}"))),
        }
    }

    fn parse_v0_v1(
        buf: &[u8],
        cur: &mut Cursor<'_>,
        version: u8,
        superblock_offset: u64,
    ) -> Result<Self> {
        let _free_space_version = cur.u8()?;
        let _root_entry_version = cur.u8()?;
        cur.skip(1)?; // reserved
        let _shared_header_version = cur.u8()?;
        let offset_size = cur.u8()?;
        let length_size = cur.u8()?;
        cur.skip(1)?; // reserved
        let sizes = validate_sizes(offset_size, length_size)?;

        let group_leaf_node_k = cur.u16()?;
        let group_internal_node_k = cur.u16()?;
        let _consistency_flags = cur.u32()?;

        let indexed_storage_internal_node_k = if version == 1 {
            let k = cur.u16()?;
            cur.skip(2)?; // reserved
            k
        } else {
            DEFAULT_ISTORE_K
        };

        let base_address = cur.uint(sizes.offset)?;
        let _free_space_info = cur.address(sizes)?;
        let end_of_file_address = cur.uint(sizes.offset)?;
        let _driver_info = cur.address(sizes)?;

        let need = SymbolTableEntry::encoded_len(sizes);
        if cur.remaining() < need {
            return Err(Error::malformed(
                "superblock is truncated before the root group symbol table entry",
            ));
        }
        let _ = buf;
        let entry = SymbolTableEntry::parse(cur, sizes)?;

        Ok(Self {
            version,
            sizes,
            superblock_offset,
            base_address,
            end_of_file_address,
            extension_address: None,
            root: RootGroup::SymbolTable(Box::new(entry)),
            group_internal_node_k,
            group_leaf_node_k,
            indexed_storage_internal_node_k,
        })
    }

    fn parse_v2_v3(
        buf: &[u8],
        cur: &mut Cursor<'_>,
        version: u8,
        superblock_offset: u64,
    ) -> Result<Self> {
        let offset_size = cur.u8()?;
        let length_size = cur.u8()?;
        let sizes = validate_sizes(offset_size, length_size)?;
        let _consistency_flags = cur.u8()?;

        let base_address = cur.uint(sizes.offset)?;
        let extension_address = cur.address(sizes)?;
        let end_of_file_address = cur.uint(sizes.offset)?;
        let root_object_header = cur.address_required(sizes, "root group object header")?;

        let checksum_pos = cur.pos();
        let stored = cur.u32()?;
        let computed = checksum::metadata(&buf[..checksum_pos]);
        if stored != computed {
            return Err(Error::ChecksumMismatch {
                what: "superblock",
                stored,
                computed,
            });
        }

        Ok(Self {
            version,
            sizes,
            superblock_offset,
            base_address,
            end_of_file_address,
            extension_address,
            root: RootGroup::ObjectHeader(root_object_header),
            // Versions 2 and 3 do not store group B-tree ranks; the old-style
            // group layout they describe is never used with these versions.
            group_internal_node_k: 0,
            group_leaf_node_k: 0,
            indexed_storage_internal_node_k: DEFAULT_ISTORE_K,
        })
    }

    /// Address of the root group's object header.
    pub fn root_object_header_address(&self) -> Result<u64> {
        match &self.root {
            RootGroup::ObjectHeader(a) => Ok(*a),
            RootGroup::SymbolTable(e) => e
                .object_header_address
                .ok_or_else(|| Error::malformed("root group has no object header address")),
        }
    }

    /// Translate a stored address into an absolute file offset.
    ///
    /// Every address inside the file is relative to the base address, which is
    /// zero in practice but is not guaranteed to be.
    pub fn resolve(&self, address: u64) -> u64 {
        self.base_address.wrapping_add(address)
    }
}

fn validate_sizes(offset: u8, length: u8) -> Result<Sizes> {
    for (what, v) in [("offset", offset), ("length", length)] {
        if v == 0 || v > 8 || !v.is_power_of_two() {
            return Err(Error::unsupported(format!(
                "size of {what}s is {v} bytes; only 1, 2, 4 and 8 are supported"
            )));
        }
    }
    Ok(Sizes { offset, length })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{FileSource, MemorySource};

    #[test]
    fn parses_superblock_version_zero_from_the_corpus() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test_files/gridded-example.nc"
        );
        let src = FileSource::open(path).unwrap();
        let sb = Superblock::read(&src).unwrap();

        assert_eq!(sb.version, 0);
        assert_eq!(sb.sizes, Sizes::EIGHT);
        assert_eq!(sb.base_address, 0);
        assert_eq!(sb.group_leaf_node_k, 4, "h5dump reports BTREE_LEAF 4");
        assert_eq!(sb.group_internal_node_k, 16, "h5dump reports BTREE_RANK 16");
        assert_eq!(
            sb.end_of_file_address,
            src.size(),
            "the recorded EOF should match the file length"
        );
        assert_eq!(
            sb.root_object_header_address().unwrap(),
            0x60,
            "root object header sits at 96"
        );
    }

    #[test]
    fn parses_superblock_version_two_from_the_corpus() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files/test_file.nc");
        let src = FileSource::open(path).unwrap();
        let sb = Superblock::read(&src).unwrap();

        assert_eq!(sb.version, 2);
        assert_eq!(sb.sizes, Sizes::EIGHT);
        assert_eq!(sb.extension_address, None);
        assert_eq!(sb.end_of_file_address, src.size());
        assert_eq!(
            sb.root_object_header_address().unwrap(),
            0x30,
            "root object header sits at 48, where the OHDR signature is"
        );
    }

    /// The version 2 superblock stores a lookup3 checksum of its own bytes.
    /// Parsing succeeds only if this crate's checksum agrees with the file, so
    /// a successful parse of every v2 file is an end-to-end check of `checksum`.
    #[test]
    fn superblock_checksum_is_verified_on_every_v2_corpus_file() {
        let mut checked = 0;
        for path in crate::test_corpus::paths() {
            let src = FileSource::open(&path).unwrap();
            let sb = Superblock::read(&src).unwrap();
            if sb.version >= 2 {
                checked += 1;
            }
        }
        assert!(checked >= 2, "expected at least two v2 files in the corpus");
    }

    #[test]
    fn rejects_a_corrupted_v2_checksum() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files/test_file.nc");
        let mut data = std::fs::read(path).unwrap();
        // Flip a bit inside the checksummed region (the EOF address).
        data[28] ^= 0xFF;
        let src = MemorySource::new(data);
        let err = Superblock::read(&src).unwrap_err();
        assert!(
            matches!(err, Error::ChecksumMismatch { .. }),
            "expected a checksum mismatch, got {err:?}"
        );
    }

    #[test]
    fn rejects_an_unsupported_superblock_version() {
        let mut data = HDF5_SIGNATURE.to_vec();
        data.push(9); // version
        data.extend_from_slice(&[0u8; 87]);
        let src = MemorySource::new(data);
        let err = Superblock::read(&src).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }

    #[test]
    fn rejects_an_implausible_address_width() {
        let mut data = HDF5_SIGNATURE.to_vec();
        data.push(2); // version 2
        data.push(7); // offset size, not a power of two
        data.push(8);
        data.extend_from_slice(&[0u8; 85]);
        let src = MemorySource::new(data);
        assert!(matches!(
            Superblock::read(&src).unwrap_err(),
            Error::Unsupported(_)
        ));
    }

    #[test]
    fn resolve_adds_the_base_address() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files/test_file.nc");
        let src = FileSource::open(path).unwrap();
        let sb = Superblock::read(&src).unwrap();
        assert_eq!(sb.resolve(0x30), 0x30, "base address is zero in this file");
    }
}
