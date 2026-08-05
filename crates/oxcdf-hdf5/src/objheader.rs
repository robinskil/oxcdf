//! Object headers, versions 1 and 2.
//!
//! An object header is the record for one HDF5 object: a group, a dataset or a
//! committed datatype. It is a list of typed messages. Everything this reader
//! needs about an object (its shape, its type, where its bytes live, its
//! attributes) arrives as a message.
//!
//! Two on-disk shapes exist and both appear in netcdf-c output:
//!
//! * Version 1 has no signature, pads every message to an 8-byte boundary, and
//!   continues into unsignposted continuation blocks.
//! * Version 2 starts with `OHDR`, packs messages without padding, optionally
//!   records a creation order per message, and checksums every block. Its
//!   continuation blocks start with `OCHK`.
//!
//! This module only frames the messages. It stores their bytes and leaves
//! interpretation to `message`, which keeps the framing rules in one place.

use std::collections::HashSet;

use crate::checksum;
use crate::context::Ctx;
use crate::cursor::Cursor;
use crate::error::{Error, Result};

/// Signature of a version 2 object header.
pub const OHDR_SIGNATURE: &[u8; 4] = b"OHDR";
/// Signature of a version 2 object header continuation block.
pub const OCHK_SIGNATURE: &[u8; 4] = b"OCHK";

/// Upper bound on continuation blocks followed for one object.
///
/// A corrupt file could otherwise send the parser round a cycle. Real objects
/// use a handful of blocks at most.
const MAX_CONTINUATION_BLOCKS: usize = 4096;

/// The type of a header message.
///
/// Values come from the format specification. Unrecognised types are preserved
/// as [`MessageType::Unknown`] rather than rejected, because the specification
/// requires readers to skip messages they do not understand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageType {
    /// Padding. Ignore.
    Nil,
    /// Rank and dimensions of a dataset or attribute.
    Dataspace,
    /// How links are stored in a group.
    LinkInfo,
    /// Element type.
    Datatype,
    /// Fill value, original encoding.
    FillValueOld,
    /// Fill value.
    FillValue,
    /// One link (a named child of a group).
    Link,
    /// Data stored outside the file.
    ExternalDataFiles,
    /// Where and how the dataset's raw bytes are stored.
    DataLayout,
    /// Test message. Never appears in real files.
    Bogus,
    /// Group creation properties.
    GroupInfo,
    /// The chain of filters applied to chunks.
    FilterPipeline,
    /// One attribute.
    Attribute,
    /// A comment.
    ObjectComment,
    /// Modification time, original encoding.
    ObjectModificationTimeOld,
    /// Shared message table.
    SharedMessageTable,
    /// Points at a continuation block.
    ObjectHeaderContinuation,
    /// Old-style group storage: B-tree plus local heap.
    SymbolTable,
    /// Modification time.
    ObjectModificationTime,
    /// B-tree rank overrides.
    BtreeKValues,
    /// File driver settings.
    DriverInfo,
    /// How attributes are stored in this object.
    AttributeInfo,
    /// Reference count.
    ObjectReferenceCount,
    /// A type this reader does not recognise.
    Unknown(u16),
}

impl MessageType {
    /// Map the on-disk code to a type.
    pub fn from_code(code: u16) -> Self {
        match code {
            0x00 => Self::Nil,
            0x01 => Self::Dataspace,
            0x02 => Self::LinkInfo,
            0x03 => Self::Datatype,
            0x04 => Self::FillValueOld,
            0x05 => Self::FillValue,
            0x06 => Self::Link,
            0x07 => Self::ExternalDataFiles,
            0x08 => Self::DataLayout,
            0x09 => Self::Bogus,
            0x0A => Self::GroupInfo,
            0x0B => Self::FilterPipeline,
            0x0C => Self::Attribute,
            0x0D => Self::ObjectComment,
            0x0E => Self::ObjectModificationTimeOld,
            0x0F => Self::SharedMessageTable,
            0x10 => Self::ObjectHeaderContinuation,
            0x11 => Self::SymbolTable,
            0x12 => Self::ObjectModificationTime,
            0x13 => Self::BtreeKValues,
            0x14 => Self::DriverInfo,
            0x15 => Self::AttributeInfo,
            0x16 => Self::ObjectReferenceCount,
            other => Self::Unknown(other),
        }
    }
}

/// Message flag: the message body is a reference to a shared message.
pub const MSG_FLAG_SHARED: u8 = 0x02;

/// One framed message, with its body still unparsed.
#[derive(Debug, Clone)]
pub struct HeaderMessage {
    /// What kind of message this is.
    pub kind: MessageType,
    /// Raw flag bits.
    pub flags: u8,
    /// Creation order, when the object header tracks it.
    pub creation_order: Option<u16>,
    /// The message body.
    pub data: Vec<u8>,
}

impl HeaderMessage {
    /// Whether the body is a pointer to a message shared with other objects.
    pub fn is_shared(&self) -> bool {
        self.flags & MSG_FLAG_SHARED != 0
    }
}

/// A fully read object header, with continuation blocks already followed.
#[derive(Debug, Clone)]
pub struct ObjectHeader {
    /// Address this header was read from.
    pub address: u64,
    /// On-disk version, 1 or 2.
    pub version: u8,
    /// Every message, in the order encountered.
    pub messages: Vec<HeaderMessage>,
}

impl ObjectHeader {
    /// Read and parse the object header at `address`.
    pub fn read(ctx: Ctx<'_>, address: u64) -> Result<Self> {
        let probe = ctx.read_upto(address, 4)?;
        if probe.len() == 4 && &probe[..] == OHDR_SIGNATURE {
            Self::read_v2(ctx, address)
        } else {
            Self::read_v1(ctx, address)
        }
    }

    /// Every message of a given type.
    pub fn messages_of(&self, kind: MessageType) -> impl Iterator<Item = &HeaderMessage> {
        self.messages.iter().filter(move |m| m.kind == kind)
    }

    /// The first message of a given type, if any.
    pub fn message_of(&self, kind: MessageType) -> Option<&HeaderMessage> {
        self.messages_of(kind).next()
    }

    // ── version 1 ──────────────────────────────────────────────────────────

    fn read_v1(ctx: Ctx<'_>, address: u64) -> Result<Self> {
        const PREFIX_LEN: usize = 16;
        let prefix = ctx.read(address, PREFIX_LEN)?;
        let mut cur = Cursor::new(&prefix);

        let version = cur.u8()?;
        if version != 1 {
            return Err(Error::unsupported(format!(
                "object header version {version} at address {address}"
            )));
        }
        cur.skip(1)?; // reserved
        let _total_messages = cur.u16()?;
        let _reference_count = cur.u32()?;
        let first_block_len = cur.u32()? as usize;
        // The remaining 4 bytes of the prefix pad the first message to an
        // 8-byte boundary.

        let mut messages = Vec::new();
        let mut queue = vec![(address + PREFIX_LEN as u64, first_block_len)];
        let mut visited = HashSet::new();
        let mut blocks = 0usize;

        while let Some((block_address, block_len)) = queue.pop() {
            blocks += 1;
            if blocks > MAX_CONTINUATION_BLOCKS {
                return Err(Error::malformed(
                    "object header has too many continuation blocks; the file may be cyclic",
                ));
            }
            if !visited.insert(block_address) {
                return Err(Error::malformed(format!(
                    "object header continuation cycle at address {block_address}"
                )));
            }

            let block = ctx.read(block_address, block_len)?;
            // Message offsets inside a v1 header are aligned relative to the
            // start of the header, and every block begins 8-byte aligned, so
            // aligning within the block gives the same answer.
            Self::parse_v1_messages(ctx, &block, &mut messages, &mut queue)?;
        }

        Ok(Self {
            address,
            version,
            messages,
        })
    }

    fn parse_v1_messages(
        ctx: Ctx<'_>,
        block: &[u8],
        out: &mut Vec<HeaderMessage>,
        queue: &mut Vec<(u64, usize)>,
    ) -> Result<()> {
        const MSG_HEADER_LEN: usize = 8;
        let mut cur = Cursor::new(block);

        while cur.remaining() >= MSG_HEADER_LEN {
            let code = cur.u16()?;
            let size = cur.u16()? as usize;
            let flags = cur.u8()?;
            cur.skip(3)?; // reserved

            if cur.remaining() < size {
                // A trailing gap too small to hold the body. Treat as the end
                // of the block rather than a failure: version 1 headers pad.
                break;
            }
            let data = cur.take(size)?.to_vec();

            let kind = MessageType::from_code(code);
            if kind == MessageType::ObjectHeaderContinuation {
                let (addr, len) = parse_continuation(&data, ctx)?;
                queue.push((addr, len));
            } else if kind != MessageType::Nil {
                out.push(HeaderMessage {
                    kind,
                    flags,
                    creation_order: None,
                    data,
                });
            }
        }
        Ok(())
    }

    // ── version 2 ──────────────────────────────────────────────────────────

    fn read_v2(ctx: Ctx<'_>, address: u64) -> Result<Self> {
        // Read a window big enough for the longest possible prefix: signature,
        // version, flags, four timestamps, the phase-change pair and an 8-byte
        // chunk size.
        let head = ctx.read_upto(address, 32)?;
        let mut cur = Cursor::new(&head);

        let sig = cur.take(4)?;
        if sig != OHDR_SIGNATURE {
            return Err(Error::malformed(
                "object header is missing its OHDR signature",
            ));
        }
        let version = cur.u8()?;
        if version != 2 {
            return Err(Error::unsupported(format!(
                "OHDR object header version {version}"
            )));
        }
        let flags = cur.u8()?;

        if flags & 0x20 != 0 {
            cur.skip(16)?; // access, modification, change and birth times
        }
        if flags & 0x10 != 0 {
            cur.skip(4)?; // max compact and min dense attribute counts
        }

        let size_width = 1usize << (flags & 0x03);
        let chunk0_len = cur.uint(size_width as u8)? as usize;
        let prefix_len = cur.pos();

        let tracks_creation_order = flags & 0x04 != 0;

        // The first chunk's checksum covers the prefix and the message area.
        let first_block = ctx.read(address, prefix_len + chunk0_len + 4)?;
        let body_start = prefix_len;
        let body_end = prefix_len + chunk0_len;
        verify_checksum(&first_block, body_end, "object header")?;

        let mut messages = Vec::new();
        let mut queue: Vec<(u64, usize)> = Vec::new();
        Self::parse_v2_messages(
            ctx,
            &first_block[body_start..body_end],
            tracks_creation_order,
            &mut messages,
            &mut queue,
        )?;

        let mut visited = HashSet::new();
        visited.insert(address);
        let mut blocks = 0usize;

        while let Some((block_address, block_len)) = queue.pop() {
            blocks += 1;
            if blocks > MAX_CONTINUATION_BLOCKS {
                return Err(Error::malformed(
                    "object header has too many continuation blocks; the file may be cyclic",
                ));
            }
            if !visited.insert(block_address) {
                return Err(Error::malformed(format!(
                    "object header continuation cycle at address {block_address}"
                )));
            }
            if block_len < 8 {
                return Err(Error::malformed(
                    "OCHK continuation block is too short to hold a signature and checksum",
                ));
            }

            let block = ctx.read(block_address, block_len)?;
            if &block[..4] != OCHK_SIGNATURE {
                return Err(Error::malformed(
                    "object header continuation block is missing its OCHK signature",
                ));
            }
            verify_checksum(&block, block_len - 4, "object header continuation")?;

            Self::parse_v2_messages(
                ctx,
                &block[4..block_len - 4],
                tracks_creation_order,
                &mut messages,
                &mut queue,
            )?;
        }

        Ok(Self {
            address,
            version,
            messages,
        })
    }

    fn parse_v2_messages(
        ctx: Ctx<'_>,
        body: &[u8],
        tracks_creation_order: bool,
        out: &mut Vec<HeaderMessage>,
        queue: &mut Vec<(u64, usize)>,
    ) -> Result<()> {
        let header_len = if tracks_creation_order { 6 } else { 4 };
        let mut cur = Cursor::new(body);

        while cur.remaining() >= header_len {
            let code = cur.u8()? as u16;
            let size = cur.u16()? as usize;
            let flags = cur.u8()?;
            let creation_order = if tracks_creation_order {
                Some(cur.u16()?)
            } else {
                None
            };

            if cur.remaining() < size {
                // The tail of a chunk may hold a gap too small for another
                // message. Stop rather than fail.
                break;
            }
            let data = cur.take(size)?.to_vec();

            let kind = MessageType::from_code(code);
            if kind == MessageType::ObjectHeaderContinuation {
                let (addr, len) = parse_continuation(&data, ctx)?;
                queue.push((addr, len));
            } else if kind != MessageType::Nil {
                out.push(HeaderMessage {
                    kind,
                    flags,
                    creation_order,
                    data,
                });
            }
        }
        Ok(())
    }
}

/// Read the address and length out of a continuation message body.
fn parse_continuation(data: &[u8], ctx: Ctx<'_>) -> Result<(u64, usize)> {
    let sizes = ctx.sizes();
    let mut cur = Cursor::new(data);
    let address = cur.address_required(sizes, "object header continuation")?;
    let length = cur.length(sizes)? as usize;
    Ok((address, length))
}

/// Check the 4-byte checksum that follows `len` bytes of `block`.
fn verify_checksum(block: &[u8], len: usize, what: &'static str) -> Result<()> {
    if block.len() < len + 4 {
        return Err(Error::malformed(format!(
            "{what} is truncated before its checksum"
        )));
    }
    let stored = u32::from_le_bytes([block[len], block[len + 1], block[len + 2], block[len + 3]]);
    let computed = checksum::metadata(&block[..len]);
    if stored != computed {
        return Err(Error::ChecksumMismatch {
            what,
            stored,
            computed,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::FileSource;
    use crate::superblock::Superblock;

    /// Superblock v0, but version 2 object headers: netcdf-c tracks attribute
    /// creation order, which forces OHDR even in an otherwise old file.
    const MIXED_FILE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_files/gridded-example.nc"
    );
    const V2_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files/test_file.nc");
    /// Version 1 object headers with symbol table groups. See
    /// `test_files/generate_legacy.c`.
    const LEGACY_FILE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_files/legacy_v1_objheader.h5"
    );

    fn open(path: &str) -> (FileSource, Superblock) {
        let src = FileSource::open(path).unwrap();
        let sb = Superblock::read(&src).unwrap();
        (src, sb)
    }

    #[test]
    fn message_codes_map_to_types() {
        assert_eq!(MessageType::from_code(0x01), MessageType::Dataspace);
        assert_eq!(MessageType::from_code(0x08), MessageType::DataLayout);
        assert_eq!(MessageType::from_code(0x11), MessageType::SymbolTable);
        assert_eq!(MessageType::from_code(0xFF), MessageType::Unknown(0xFF));
    }

    #[test]
    fn reads_a_version_one_root_group_header() {
        let (src, sb) = open(LEGACY_FILE);
        let ctx = Ctx::new(&src, &sb);
        let oh = ObjectHeader::read(ctx, sb.root_object_header_address().unwrap()).unwrap();

        assert_eq!(oh.version, 1, "the legacy fixture uses v1 object headers");
        assert!(
            oh.message_of(MessageType::SymbolTable).is_some(),
            "a v1 group is described by a symbol table message"
        );
        assert!(
            oh.message_of(MessageType::Attribute).is_some(),
            "the fixture puts a `title` attribute on the root group"
        );
    }

    #[test]
    fn reads_a_version_two_root_group_header() {
        let (src, sb) = open(V2_FILE);
        let ctx = Ctx::new(&src, &sb);
        let oh = ObjectHeader::read(ctx, sb.root_object_header_address().unwrap()).unwrap();

        assert_eq!(oh.version, 2);
        assert!(
            oh.message_of(MessageType::LinkInfo).is_some(),
            "a v2 group stores links through a link info message"
        );
    }

    /// A version 0 superblock does not imply version 1 object headers. This
    /// file pairs the two, and the reader must dispatch on the OHDR signature
    /// rather than on the superblock version.
    #[test]
    fn detects_version_two_headers_under_a_version_zero_superblock() {
        let (src, sb) = open(MIXED_FILE);
        assert_eq!(sb.version, 0);
        let ctx = Ctx::new(&src, &sb);
        let oh = ObjectHeader::read(ctx, sb.root_object_header_address().unwrap()).unwrap();
        assert_eq!(oh.version, 2);
    }

    /// Every version 2 block carries a checksum. Reading all of them without a
    /// mismatch proves the framing consumed exactly the right bytes, including
    /// the creation-order field and any continuation blocks.
    #[test]
    fn version_two_headers_pass_their_checksums() {
        let (src, sb) = open(V2_FILE);
        let ctx = Ctx::new(&src, &sb);
        ObjectHeader::read(ctx, sb.root_object_header_address().unwrap()).unwrap();
    }

    #[test]
    fn rejects_a_corrupted_object_header_checksum() {
        let mut data = std::fs::read(V2_FILE).unwrap();
        // The root object header body starts just past the OHDR prefix at 0x30.
        data[0x40] ^= 0xFF;
        let src = crate::source::MemorySource::new(data);
        let sb = Superblock::read(&src).unwrap();
        let ctx = Ctx::new(&src, &sb);
        let err = ObjectHeader::read(ctx, sb.root_object_header_address().unwrap()).unwrap_err();
        assert!(
            matches!(err, Error::ChecksumMismatch { .. }),
            "expected a checksum mismatch, got {err:?}"
        );
    }

    #[test]
    fn nil_and_continuation_messages_are_not_surfaced() {
        let (src, sb) = open(V2_FILE);
        let ctx = Ctx::new(&src, &sb);
        let oh = ObjectHeader::read(ctx, sb.root_object_header_address().unwrap()).unwrap();
        assert!(
            !oh.messages
                .iter()
                .any(|m| m.kind == MessageType::Nil
                    || m.kind == MessageType::ObjectHeaderContinuation),
            "padding and continuation pointers are framing details, not content"
        );
    }
}
