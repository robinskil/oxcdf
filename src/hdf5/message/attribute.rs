//! The attribute message: one named value attached to an object.
//!
//! An attribute packs its own datatype and dataspace inline, followed by the
//! raw value bytes. netCDF exposes these directly as variable and global
//! attributes, and also uses them for its own bookkeeping (`DIMENSION_LIST`,
//! `CLASS`, `NAME`, `_Netcdf4Dimid`).

use crate::cursor::{Cursor, Sizes};
use crate::error::{Error, Result};
use crate::hdf5::message::dataspace::Dataspace;
use crate::hdf5::message::datatype::Datatype;

/// Flag: the datatype is stored as a shared message rather than inline.
pub const FLAG_SHARED_DATATYPE: u8 = 0x01;
/// Flag: the dataspace is stored as a shared message rather than inline.
pub const FLAG_SHARED_DATASPACE: u8 = 0x02;

/// A parsed attribute.
#[derive(Debug, Clone)]
pub struct Attribute {
    /// Attribute name.
    pub name: String,
    /// Element type.
    pub datatype: Datatype,
    /// Shape.
    pub dataspace: Dataspace,
    /// Raw value bytes, still in the file's byte order.
    pub data: Vec<u8>,
}

impl Attribute {
    /// Parse a whole attribute message body.
    ///
    /// The value is taken as everything after the dataspace, which is right for
    /// an object header message because the framing already bounded the body.
    pub fn parse(body: &[u8], sizes: Sizes) -> Result<Self> {
        let mut cur = Cursor::new(body);
        Self::read(&mut cur, sizes, ValueExtent::Rest)
    }

    /// Parse one attribute at the cursor, leaving it on the next record.
    ///
    /// Fractal heap records sit back to back with no framing, so here the value
    /// length is computed from the dataspace and datatype instead of running to
    /// the end of the buffer.
    pub fn parse_at(cur: &mut Cursor<'_>, sizes: Sizes) -> Result<Self> {
        Self::read(cur, sizes, ValueExtent::Computed)
    }

    fn read(cur: &mut Cursor<'_>, sizes: Sizes, extent: ValueExtent) -> Result<Self> {
        let version = cur.u8()?;
        match version {
            1 => Self::parse_v1(cur, sizes, extent),
            2 | 3 => Self::parse_v2_v3(cur, version, sizes, extent),
            other => Err(Error::unsupported(format!(
                "attribute message version {other}"
            ))),
        }
    }

    fn parse_v1(cur: &mut Cursor<'_>, sizes: Sizes, extent: ValueExtent) -> Result<Self> {
        cur.skip(1)?; // reserved
        let name_size = cur.u16()? as usize;
        let datatype_size = cur.u16()? as usize;
        let dataspace_size = cur.u16()? as usize;

        // Version 1 pads each of the three blocks to a multiple of 8 bytes.
        let name = read_padded_name(cur, name_size, 8)?;
        let datatype = Datatype::parse(cur.take(pad8(datatype_size))?)?;
        let dataspace = Dataspace::parse(cur.take(pad8(dataspace_size))?, sizes)?;
        let data = take_value(cur, &datatype, &dataspace, extent)?;

        Ok(Self {
            name,
            datatype,
            dataspace,
            data,
        })
    }

    fn parse_v2_v3(
        cur: &mut Cursor<'_>,
        version: u8,
        sizes: Sizes,
        extent: ValueExtent,
    ) -> Result<Self> {
        let flags = cur.u8()?;
        let name_size = cur.u16()? as usize;
        let datatype_size = cur.u16()? as usize;
        let dataspace_size = cur.u16()? as usize;

        if flags & (FLAG_SHARED_DATATYPE | FLAG_SHARED_DATASPACE) != 0 {
            // Resolving a shared message means following it into the shared
            // message table, which netcdf-c never writes. Say so rather than
            // decode the pointer as if it were a datatype.
            return Err(Error::unsupported(
                "attribute with a shared datatype or dataspace",
            ));
        }

        if version == 3 {
            cur.skip(1)?; // name character set encoding
        }

        // Versions 2 and 3 store the three blocks unpadded.
        let name = read_padded_name(cur, name_size, 1)?;
        let datatype = Datatype::parse(cur.take(datatype_size)?)?;
        let dataspace = Dataspace::parse(cur.take(dataspace_size)?, sizes)?;
        let data = take_value(cur, &datatype, &dataspace, extent)?;

        Ok(Self {
            name,
            datatype,
            dataspace,
            data,
        })
    }

    /// Number of elements the attribute holds.
    pub fn element_count(&self) -> u64 {
        self.dataspace.element_count()
    }
}

/// How much of the buffer after the dataspace belongs to the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueExtent {
    /// Everything that is left. Correct when the caller already bounded the body.
    Rest,
    /// Exactly what the dataspace and datatype imply. Required when records sit
    /// back to back, as they do in a fractal heap.
    Computed,
}

/// Take the attribute's value bytes according to `extent`.
fn take_value(
    cur: &mut Cursor<'_>,
    datatype: &Datatype,
    dataspace: &Dataspace,
    extent: ValueExtent,
) -> Result<Vec<u8>> {
    match extent {
        ValueExtent::Rest => Ok(cur.take(cur.remaining())?.to_vec()),
        ValueExtent::Computed => {
            let count = dataspace.element_count();
            let len = count
                .checked_mul(datatype.size as u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    Error::malformed("attribute value length overflows a machine word")
                })?;
            Ok(cur.take(len)?.to_vec())
        }
    }
}

/// Read a name of `size` bytes, then advance to the next `align` boundary.
fn read_padded_name(cur: &mut Cursor<'_>, size: usize, align: usize) -> Result<String> {
    let padded = if align > 1 { size.div_ceil(align) * align } else { size };
    let raw = cur.take(padded)?;
    let end = raw
        .iter()
        .take(size)
        .position(|&b| b == 0)
        .unwrap_or(size.min(raw.len()));
    Ok(String::from_utf8_lossy(&raw[..end]).into_owned())
}

fn pad8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdf5::message::datatype::{ByteOrder, DatatypeClass};

    fn i32_datatype() -> Vec<u8> {
        let mut v = vec![1u8 << 4, 0x08, 0, 0];
        v.extend_from_slice(&4u32.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(&32u16.to_le_bytes());
        v
    }

    fn scalar_dataspace() -> Vec<u8> {
        vec![2u8, 0, 0, 0]
    }

    #[test]
    fn parses_a_version_three_attribute() {
        let dt = i32_datatype();
        let ds = scalar_dataspace();
        let name = b"units\0";

        let mut d = vec![3u8, 0];
        d.extend_from_slice(&(name.len() as u16).to_le_bytes());
        d.extend_from_slice(&(dt.len() as u16).to_le_bytes());
        d.extend_from_slice(&(ds.len() as u16).to_le_bytes());
        d.push(0); // character set
        d.extend_from_slice(name);
        d.extend_from_slice(&dt);
        d.extend_from_slice(&ds);
        d.extend_from_slice(&42i32.to_le_bytes());

        let a = Attribute::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(a.name, "units");
        assert_eq!(a.datatype.size, 4);
        assert_eq!(a.datatype.byte_order(), Some(ByteOrder::Little));
        assert_eq!(a.element_count(), 1);
        assert_eq!(a.data, 42i32.to_le_bytes());
    }

    #[test]
    fn parses_a_version_two_attribute_without_a_charset_byte() {
        let dt = i32_datatype();
        let ds = scalar_dataspace();
        let name = b"scale\0";

        let mut d = vec![2u8, 0];
        d.extend_from_slice(&(name.len() as u16).to_le_bytes());
        d.extend_from_slice(&(dt.len() as u16).to_le_bytes());
        d.extend_from_slice(&(ds.len() as u16).to_le_bytes());
        d.extend_from_slice(name);
        d.extend_from_slice(&dt);
        d.extend_from_slice(&ds);
        d.extend_from_slice(&7i32.to_le_bytes());

        let a = Attribute::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(a.name, "scale");
        assert_eq!(a.data, 7i32.to_le_bytes());
    }

    #[test]
    fn parses_a_version_one_attribute_with_eight_byte_padding() {
        let dt = i32_datatype(); // 12 bytes, pads to 16
        let ds = scalar_dataspace(); // 4 bytes, pads to 8
        let name = b"a\0"; // 2 bytes, pads to 8

        let mut d = vec![1u8, 0];
        d.extend_from_slice(&(name.len() as u16).to_le_bytes());
        d.extend_from_slice(&(dt.len() as u16).to_le_bytes());
        d.extend_from_slice(&(ds.len() as u16).to_le_bytes());
        d.extend_from_slice(name);
        d.extend_from_slice(&[0u8; 6]); // name padding
        d.extend_from_slice(&dt);
        d.extend_from_slice(&[0u8; 4]); // datatype padding
        d.extend_from_slice(&ds);
        d.extend_from_slice(&[0u8; 4]); // dataspace padding
        d.extend_from_slice(&5i32.to_le_bytes());

        let a = Attribute::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(a.name, "a");
        assert_eq!(a.data, 5i32.to_le_bytes());
        match a.datatype.class {
            DatatypeClass::FixedPoint { signed, .. } => assert!(signed),
            other => panic!("expected fixed point, got {other:?}"),
        }
    }

    #[test]
    fn reports_a_shared_datatype_as_unsupported() {
        let mut d = vec![3u8, FLAG_SHARED_DATATYPE];
        d.extend_from_slice(&2u16.to_le_bytes());
        d.extend_from_slice(&8u16.to_le_bytes());
        d.extend_from_slice(&4u16.to_le_bytes());
        d.push(0);
        d.extend_from_slice(&[0u8; 32]);
        let err = Attribute::parse(&d, Sizes::EIGHT).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }

    #[test]
    fn rejects_an_unknown_attribute_version() {
        let err = Attribute::parse(&[8u8, 0, 0, 0, 0, 0, 0, 0], Sizes::EIGHT).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }
}
