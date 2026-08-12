//! A dataset whose datatype lives in another object header.
//!
//! HDF5 calls this a committed, or named, datatype. The dataset's datatype
//! message carries the shared flag, and its body is a pointer rather than a
//! type. netCDF-4 writes one for every user-defined type, so a file that
//! declares a compound, enum, opaque or vlen type depends on this path.
//!
//! The fixtures here are built byte by byte. A version 1 object header carries
//! no checksum, so a test can assemble one and vary a single field, which is
//! what pins the behaviour down. Real files are covered by `corpus_index`.

use oxcdf_hdf5::cursor::Sizes;
use oxcdf_hdf5::message::{Datatype, DatatypeClass};
use oxcdf_hdf5::{ByteSource, Ctx, Error, MemorySource, ObjectHeader, RootGroup, Superblock};

/// Message type code for a datatype.
const MSG_DATATYPE: u16 = 0x03;
/// Message type code for an attribute, used as filler.
const MSG_ATTRIBUTE: u16 = 0x0C;
/// Message flag: the body points at a message held elsewhere.
const FLAG_SHARED: u8 = 0x02;
/// Shared message type: the target is a committed datatype.
const SHARE_COMMITTED: u8 = 2;
/// Shared message type: the target is in the shared message heap.
const SHARE_HEAP: u8 = 1;

/// A signed 32-bit little-endian datatype message body.
///
/// Byte 0 packs version 1 and class 0 (fixed point). Bit 3 of the class bit
/// field marks it signed. These are the bytes netcdf-c writes for `int`.
fn int32_datatype() -> Vec<u8> {
    let mut body = vec![0x10, 0x08, 0x00, 0x00];
    body.extend_from_slice(&4u32.to_le_bytes()); // size in bytes
    body.extend_from_slice(&0u16.to_le_bytes()); // bit offset
    body.extend_from_slice(&32u16.to_le_bytes()); // bit precision
    body
}

/// A shared message body pointing at `address`.
fn shared_pointer(version: u8, kind: u8, address: u64) -> Vec<u8> {
    let mut body = vec![version, kind];
    if version == 1 {
        // Version 1 pads the header out to eight bytes before the address.
        body.resize(8, 0);
    }
    body.extend_from_slice(&address.to_le_bytes());
    body
}

/// Frame one version 1 object header message.
fn message(code: u16, flags: u8, body: &[u8]) -> Vec<u8> {
    // Version 1 pads every body out to a multiple of eight bytes.
    let padded = body.len().div_ceil(8) * 8;
    let mut out = Vec::with_capacity(8 + padded);
    out.extend_from_slice(&code.to_le_bytes());
    out.extend_from_slice(&(padded as u16).to_le_bytes());
    out.push(flags);
    out.extend_from_slice(&[0, 0, 0]); // reserved
    out.extend_from_slice(body);
    out.resize(8 + padded, 0);
    out
}

/// Wrap framed messages in a version 1 object header.
fn object_header(messages: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = messages.concat();
    let mut out = Vec::with_capacity(16 + body.len());
    out.push(1); // version
    out.push(0); // reserved
    out.extend_from_slice(&(messages.len() as u16).to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes()); // reference count
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&[0, 0, 0, 0]); // pad to the 16-byte prefix
    out.extend_from_slice(&body);
    out
}

/// Build a file: an object header that points, then the object it points at.
///
/// The pointing header always starts at address zero, so the committed object
/// begins at the pointing header's own length. That length does not depend on
/// the address it carries, so it can be measured once and reused.
fn file_pointing_at(version: u8, kind: u8, committed: Option<Vec<u8>>) -> Vec<u8> {
    let pointer = |address| {
        object_header(&[message(
            MSG_DATATYPE,
            FLAG_SHARED,
            &shared_pointer(version, kind, address),
        )])
    };

    let target_address = pointer(0).len() as u64;
    let mut bytes = pointer(target_address);
    assert_eq!(bytes.len() as u64, target_address, "framing changed size");

    if let Some(messages) = committed {
        bytes.extend_from_slice(&object_header(&[messages]));
    }
    bytes
}

/// Resolve the datatype of the object header at address zero.
fn datatype_of(bytes: Vec<u8>) -> oxcdf_hdf5::Result<Option<Datatype>> {
    let source = MemorySource::new(bytes);
    let superblock = Superblock {
        version: 2,
        sizes: Sizes::EIGHT,
        superblock_offset: 0,
        base_address: 0,
        end_of_file_address: source.size(),
        extension_address: None,
        root: RootGroup::ObjectHeader(0),
        group_internal_node_k: 16,
        group_leaf_node_k: 4,
        indexed_storage_internal_node_k: 32,
    };
    let ctx = Ctx::new(&source, &superblock);
    ObjectHeader::read(ctx, 0)?.datatype(ctx)
}

#[test]
fn follows_a_shared_datatype_to_the_committed_type() {
    let committed = message(MSG_DATATYPE, 0, &int32_datatype());
    let bytes = file_pointing_at(2, SHARE_COMMITTED, Some(committed));

    let datatype = datatype_of(bytes).unwrap().expect("a resolved datatype");

    assert_eq!(datatype.size, 4);
    assert!(
        matches!(
            datatype.class,
            DatatypeClass::FixedPoint { signed: true, .. }
        ),
        "{:?}",
        datatype.class
    );
}

#[test]
fn follows_every_shared_message_version() {
    // Version 1 pads before the address; versions 2 and 3 do not. All three
    // must land on the same committed type.
    for version in [1u8, 2, 3] {
        let committed = message(MSG_DATATYPE, 0, &int32_datatype());
        let bytes = file_pointing_at(version, SHARE_COMMITTED, Some(committed));

        let datatype = datatype_of(bytes)
            .unwrap_or_else(|e| panic!("version {version} pointer failed: {e}"))
            .expect("a resolved datatype");

        assert_eq!(datatype.size, 4, "version {version}");
    }
}

#[test]
fn reports_the_shared_message_heap_as_unsupported() {
    // Type 1 puts the message in the shared message heap, which this reader
    // does not read. That must stay a fallback-worthy error, not a wrong type.
    let err = datatype_of(file_pointing_at(2, SHARE_HEAP, None)).unwrap_err();

    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    assert!(err.is_fallback_worthy());
}

#[test]
fn rejects_a_pointer_to_a_header_without_a_datatype() {
    let filler = message(MSG_ATTRIBUTE, 0, &[0u8; 8]);
    let bytes = file_pointing_at(2, SHARE_COMMITTED, Some(filler));

    let err = datatype_of(bytes).unwrap_err();
    assert!(matches!(err, Error::Malformed(_)), "{err:?}");
}

#[test]
fn stops_on_a_committed_datatype_that_points_at_itself() {
    // A cycle must end in an error, not a hang or a stack overflow.
    let bytes = object_header(&[message(
        MSG_DATATYPE,
        FLAG_SHARED,
        &shared_pointer(2, SHARE_COMMITTED, 0),
    )]);

    let err = datatype_of(bytes).unwrap_err();
    assert!(matches!(err, Error::Malformed(_)), "{err:?}");
}

#[test]
fn rejects_a_truncated_shared_body() {
    let bytes = object_header(&[message(MSG_DATATYPE, FLAG_SHARED, &[0x02, 0x02, 0xff])]);
    assert!(datatype_of(bytes).is_err());
}

#[test]
fn an_unshared_datatype_still_parses_inline() {
    // The common case must not regress: no shared flag, body read as the type.
    let bytes = object_header(&[message(MSG_DATATYPE, 0, &int32_datatype())]);

    let datatype = datatype_of(bytes).unwrap().expect("a datatype");
    assert_eq!(datatype.size, 4);
}
