//! Object header and superblock encoding for the writer.

use super::encode::{Encoder, UNDEFINED_ADDRESS};
use crate::error::{Error, Result};
use crate::objheader::{HeaderMessage, MessageType, OHDR_SIGNATURE};
use crate::HDF5_SIGNATURE;

/// Length of a version 2 superblock with 8-byte offsets and lengths.
pub const SUPERBLOCK_BYTES: u64 = 48;

/// Length of the version 2 object header prefix with a 4-byte size field.
const PREFIX_BYTES: usize = 10;

/// Length of one message frame: type, size, flags and creation order.
const FRAME_BYTES: usize = 6;

/// Length of a checksum.
const CHECKSUM_BYTES: usize = 4;

/// Largest message body that the 2-byte size field holds.
const MAX_BODY_BYTES: usize = u16::MAX as usize;

/// Object header flags: 4-byte chunk size and creation order tracked.
const OHDR_FLAGS: u8 = 0x06;

/// A header message with no flags and no creation order.
pub fn message(kind: MessageType, data: Vec<u8>) -> HeaderMessage {
    HeaderMessage {
        kind,
        flags: 0,
        creation_order: None,
        data,
    }
}

/// Encode a version 2 superblock with 8-byte offsets and lengths.
pub fn encode_superblock(root_address: u64, end_of_file: u64) -> Vec<u8> {
    let mut e = Encoder::new();
    e.bytes(&HDF5_SIGNATURE);
    e.u8(2); // version
    e.u8(8); // offset size
    e.u8(8); // length size
    e.u8(0); // consistency flags
    e.u64(0); // base address
    e.u64(UNDEFINED_ADDRESS); // superblock extension address
    e.u64(end_of_file);
    e.u64(root_address);
    e.checksum();
    e.into_bytes()
}

/// Exact length of the bytes that [`encode_header`] returns for these arguments.
pub fn header_len(messages: &[HeaderMessage], slack: usize) -> usize {
    let mut len = PREFIX_BYTES.saturating_add(CHECKSUM_BYTES);
    for m in messages {
        len = len.saturating_add(FRAME_BYTES.saturating_add(m.data.len()));
    }
    if slack > 0 {
        len = len.saturating_add(FRAME_BYTES.saturating_add(slack));
    }
    len
}

/// Encode a version 2 object header with all messages in one chunk.
///
/// A `slack` above zero appends one `Nil` message with `slack` zero bytes.
pub fn encode_header(messages: &[HeaderMessage], slack: usize) -> Result<Vec<u8>> {
    // header_len is never below the prefix plus the checksum.
    let area = header_len(messages, slack) - PREFIX_BYTES - CHECKSUM_BYTES;
    let area = u32::try_from(area).map_err(|_| {
        Error::unsupported(format!(
            "object header message area of {area} bytes is above the 4-byte size limit"
        ))
    })?;

    let mut e = Encoder::new();
    e.bytes(OHDR_SIGNATURE);
    e.u8(2); // version
    e.u8(OHDR_FLAGS);
    e.u32(area);
    for m in messages {
        frame(&mut e, m.kind, m.flags, m.creation_order, m.data.len())?;
        e.bytes(&m.data);
    }
    if slack > 0 {
        frame(&mut e, MessageType::Nil, 0, None, slack)?;
        e.zeros(slack);
    }
    e.checksum();
    Ok(e.into_bytes())
}

/// Append one message frame, or reject a type or length the frame cannot hold.
fn frame(
    e: &mut Encoder,
    kind: MessageType,
    flags: u8,
    creation_order: Option<u16>,
    len: usize,
) -> Result<()> {
    let code = u8::try_from(kind.code()).map_err(|_| {
        Error::unsupported(format!(
            "message type code {} does not fit the 1-byte type field",
            kind.code()
        ))
    })?;
    let size = u16::try_from(len).map_err(|_| {
        Error::unsupported(format!(
            "{kind:?} message body of {len} bytes is above the {MAX_BODY_BYTES}-byte limit"
        ))
    })?;
    e.u8(code);
    e.u16(size);
    e.u8(flags);
    e.u16(creation_order.unwrap_or(0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Ctx;
    use crate::cursor::Sizes;
    use crate::error::Error;
    use crate::objheader::{HeaderMessage, MessageType, ObjectHeader};
    use crate::source::MemorySource;
    use crate::superblock::Superblock;

    fn parse(bytes: Vec<u8>, address: u64) -> ObjectHeader {
        let mut file = vec![0u8; address as usize];
        file.extend_from_slice(&bytes);
        let source = MemorySource::new(file);
        let superblock = Superblock::read(&MemorySource::new(encode_superblock(48, 4096))).unwrap();
        let ctx = Ctx::new(&source, &superblock);
        ObjectHeader::read(ctx, address).unwrap()
    }

    #[test]
    fn a_superblock_reads_back() {
        let bytes = encode_superblock(48, 70000);
        assert_eq!(bytes.len() as u64, SUPERBLOCK_BYTES);
        let sb = Superblock::read(&MemorySource::new(bytes)).unwrap();
        assert_eq!(sb.version, 2);
        assert_eq!(sb.sizes, Sizes::EIGHT);
        assert_eq!(sb.end_of_file_address, 70000);
        assert_eq!(sb.root_object_header_address().unwrap(), 48);
    }

    #[test]
    fn a_header_reads_back_without_its_slack() {
        let messages = vec![
            HeaderMessage {
                kind: MessageType::GroupInfo,
                flags: 0,
                creation_order: None,
                data: vec![0, 0],
            },
            HeaderMessage {
                kind: MessageType::Attribute,
                flags: 0,
                creation_order: Some(5),
                data: vec![9; 40],
            },
        ];
        let bytes = encode_header(&messages, 128).unwrap();
        assert_eq!(bytes.len(), header_len(&messages, 128));
        assert_eq!(bytes.len(), 10 + (6 + 2) + (6 + 40) + (6 + 128) + 4);
        let header = parse(bytes, 48);
        assert_eq!(header.version, 2);
        assert_eq!(header.messages.len(), 2);
        assert_eq!(header.messages[1].creation_order, Some(5));
        assert_eq!(header.messages[1].data, vec![9; 40]);
    }

    #[test]
    fn a_header_with_no_slack_has_no_nil_message() {
        let bytes = encode_header(&[], 0).unwrap();
        assert_eq!(bytes.len(), 14);
    }

    #[test]
    fn a_message_above_the_limit_is_unsupported() {
        let big = HeaderMessage {
            kind: MessageType::Attribute,
            flags: 0,
            creation_order: None,
            data: vec![0; 65536],
        };
        assert!(matches!(
            encode_header(&[big], 0),
            Err(Error::Unsupported(_))
        ));
    }
}
