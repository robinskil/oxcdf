//! The shared message pointer: a message body that names another location.
//!
//! A message header sets [`MSG_FLAG_SHARED`] when its body is not the message
//! itself but a pointer to one message held elsewhere. HDF5 uses this for a
//! committed (named) datatype: the type lives in its own object header, and
//! every dataset of that type points at it.
//!
//! netCDF-4 writes a committed datatype for each user-defined type, so a file
//! that declares a compound, enum, opaque or vlen type reaches this code.
//!
//! [`MSG_FLAG_SHARED`]: crate::objheader::MSG_FLAG_SHARED

use crate::cursor::{Cursor, Sizes};
use crate::error::{Error, Result};

/// Where a shared message actually lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedLocation {
    /// The message is not shared after all.
    Unshared,
    /// The message sits in the shared object header message heap.
    ///
    /// Only a file that turns on shared message tables holds one. netcdf-c
    /// never does.
    MessageHeap,
    /// The message is the sole message of the object header at this address.
    ///
    /// This is a committed, or named, datatype.
    Committed(u64),
    /// The message is shared but stored in the pointing object's own header.
    Here,
}

/// A parsed pointer to a message stored somewhere other than inline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedMessage {
    /// On-disk version, 1 through 3.
    pub version: u8,
    /// What the pointer refers to.
    pub location: SharedLocation,
}

impl SharedMessage {
    /// Parse a shared message body.
    ///
    /// The three versions differ only in where the address starts. Version 1
    /// pads the header out to eight bytes; versions 2 and 3 put the address
    /// straight after the type byte.
    pub fn parse(body: &[u8], sizes: Sizes) -> Result<Self> {
        let mut cur = Cursor::new(body);
        let version = cur.u8()?;

        // Version 1 records no type byte in practice: the field is there, but
        // the only thing version 1 ever pointed at was a committed datatype.
        let kind = cur.u8()?;

        match version {
            1 => cur.seek(8)?,
            2 | 3 => {}
            other => {
                return Err(Error::unsupported(format!(
                    "shared message version {other}"
                )))
            }
        }

        let location = match (version, kind) {
            // Version 1 has no meaningful type code. It is always committed.
            (1, _) => SharedLocation::Committed(address_of(&mut cur, sizes)?),
            (_, 0) => SharedLocation::Unshared,
            (_, 1) => SharedLocation::MessageHeap,
            (_, 2) => SharedLocation::Committed(address_of(&mut cur, sizes)?),
            (_, 3) => SharedLocation::Here,
            (_, other) => return Err(Error::unsupported(format!("shared message type {other}"))),
        };

        Ok(Self { version, location })
    }

    /// The object header address of a committed message, or an error naming
    /// what was found instead.
    ///
    /// Every caller wants a committed datatype, so the error text is written
    /// for that case.
    pub fn committed_address(&self, what: &str) -> Result<u64> {
        match self.location {
            SharedLocation::Committed(address) => Ok(address),
            SharedLocation::MessageHeap => Err(Error::unsupported(format!(
                "{what} held in the shared message heap"
            ))),
            SharedLocation::Here => Err(Error::unsupported(format!(
                "{what} shared within its own object header"
            ))),
            SharedLocation::Unshared => Err(Error::malformed(format!(
                "{what} is flagged shared but points nowhere"
            ))),
        }
    }
}

fn address_of(cur: &mut Cursor<'_>, sizes: Sizes) -> Result<u64> {
    cur.address_required(sizes, "shared message target")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_version_2_committed_pointer() {
        // Taken from a netCDF-4 file that declares a compound type: version 2,
        // type 2, then the eight-byte address of the committed datatype.
        let body = [0x02, 0x02, 0xff, 0, 0, 0, 0, 0, 0, 0];
        let shared = SharedMessage::parse(&body, Sizes::EIGHT).unwrap();

        assert_eq!(shared.version, 2);
        assert_eq!(shared.location, SharedLocation::Committed(255));
        assert_eq!(shared.committed_address("datatype").unwrap(), 255);
    }

    #[test]
    fn reads_a_version_1_pointer_past_its_padding() {
        // Version 1 pads to eight bytes before the address.
        let mut body = vec![0x01, 0x00, 0, 0, 0, 0, 0, 0];
        body.extend_from_slice(&4096u64.to_le_bytes());

        let shared = SharedMessage::parse(&body, Sizes::EIGHT).unwrap();
        assert_eq!(shared.location, SharedLocation::Committed(4096));
    }

    #[test]
    fn reads_a_version_3_committed_pointer() {
        let mut body = vec![0x03, 0x02];
        body.extend_from_slice(&1234u64.to_le_bytes());

        let shared = SharedMessage::parse(&body, Sizes::EIGHT).unwrap();
        assert_eq!(shared.location, SharedLocation::Committed(1234));
    }

    #[test]
    fn reports_the_message_heap_as_unsupported() {
        let body = [0x03, 0x01, 0, 0, 0, 0, 0, 0, 0, 0];
        let shared = SharedMessage::parse(&body, Sizes::EIGHT).unwrap();

        assert_eq!(shared.location, SharedLocation::MessageHeap);
        let err = shared.committed_address("datatype").unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    }

    #[test]
    fn rejects_a_truncated_body() {
        assert!(SharedMessage::parse(&[0x02, 0x02, 0xff], Sizes::EIGHT).is_err());
    }

    #[test]
    fn rejects_an_unknown_version() {
        assert!(SharedMessage::parse(&[0x09, 0x02, 0, 0, 0, 0, 0, 0, 0, 0], Sizes::EIGHT).is_err());
    }
}
