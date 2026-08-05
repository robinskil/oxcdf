//! The datatype message: how to interpret the bytes of one element.
//!
//! HDF5 datatypes are a general system. netCDF-4 uses a narrow slice of it:
//! fixed-point and floating-point numbers, fixed-length strings, and
//! variable-length sequences of object references for the `DIMENSION_LIST`
//! attribute that ties a variable to its dimensions.
//!
//! Everything is parsed into [`Datatype`], including classes this reader cannot
//! decode values for. Knowing that a variable is a compound type is useful even
//! when reading it is not yet possible, because the caller can then fall back to
//! netcdf-c for that one variable instead of failing the whole file.

use crate::cursor::Cursor;
use crate::error::{Error, Result};

/// Byte order of a numeric datatype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// Least significant byte first.
    Little,
    /// Most significant byte first.
    Big,
}

/// How a fixed-length string is padded out to its declared width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringPad {
    /// Terminated by a NUL, remainder unspecified.
    NullTerminate,
    /// Padded with NULs.
    NullPad,
    /// Padded with spaces, with no terminator. This is the Fortran convention,
    /// and netCDF character variables use it.
    SpacePad,
}

/// Character set of a string datatype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CharSet {
    /// 7-bit ASCII.
    Ascii,
    /// UTF-8.
    Utf8,
}

/// What a variable-length datatype holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VlenKind {
    /// A sequence of the base type.
    Sequence,
    /// A string.
    String,
}

/// One member of a compound datatype.
#[derive(Debug, Clone, PartialEq)]
pub struct CompoundMember {
    /// Member name.
    pub name: String,
    /// Byte offset of the member within the compound element.
    pub offset: u64,
    /// Member type.
    pub datatype: Datatype,
}

/// The class-specific part of a datatype.
#[derive(Debug, Clone, PartialEq)]
pub enum DatatypeClass {
    /// An integer.
    FixedPoint {
        /// Byte order.
        order: ByteOrder,
        /// Whether the value is two's-complement signed.
        signed: bool,
        /// First significant bit within the element.
        bit_offset: u16,
        /// Number of significant bits.
        bit_precision: u16,
    },
    /// An IEEE floating-point number.
    FloatingPoint {
        /// Byte order.
        order: ByteOrder,
        /// First significant bit within the element.
        bit_offset: u16,
        /// Number of significant bits.
        bit_precision: u16,
        /// Bit position of the exponent.
        exponent_location: u8,
        /// Width of the exponent in bits.
        exponent_size: u8,
        /// Bit position of the mantissa.
        mantissa_location: u8,
        /// Width of the mantissa in bits.
        mantissa_size: u8,
        /// Exponent bias.
        exponent_bias: u32,
        /// Bit position of the sign.
        sign_location: u8,
    },
    /// A fixed-length string. The width is the datatype's `size`.
    String {
        /// How the value is padded to the full width.
        pad: StringPad,
        /// Character set.
        charset: CharSet,
    },
    /// An opaque blob with a descriptive tag.
    Opaque {
        /// Application-supplied tag.
        tag: String,
    },
    /// A record of named members.
    Compound {
        /// The members, in declaration order.
        members: Vec<CompoundMember>,
    },
    /// A reference to another object or to a region of a dataset.
    Reference {
        /// Reference flavour, straight from the bit field.
        kind: u8,
    },
    /// A named-constant type over an integer base.
    Enum {
        /// The underlying integer type.
        base: Box<Datatype>,
        /// Constant names.
        names: Vec<String>,
        /// Raw constant values, each `base.size` bytes wide.
        values: Vec<Vec<u8>>,
    },
    /// A variable-length sequence or string, stored through the global heap.
    VariableLength {
        /// Whether this is a sequence or a string.
        kind: VlenKind,
        /// Element type of the sequence, or the character type of the string.
        base: Box<Datatype>,
        /// Padding rule, for the string flavour.
        pad: StringPad,
        /// Character set, for the string flavour.
        charset: CharSet,
    },
    /// A fixed-shape array of the base type.
    Array {
        /// Shape of the array.
        dims: Vec<u32>,
        /// Element type.
        base: Box<Datatype>,
    },
    /// A class this reader does not decode.
    Unsupported {
        /// The class code from the file.
        class: u8,
    },
}

/// A parsed datatype message.
#[derive(Debug, Clone, PartialEq)]
pub struct Datatype {
    /// On-disk version of the message.
    pub version: u8,
    /// Width of one element in bytes.
    pub size: u32,
    /// The class-specific detail.
    pub class: DatatypeClass,
}

impl Datatype {
    /// Parse a datatype message body.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut cur = Cursor::new(data);
        Self::parse_at(&mut cur)
    }

    /// Parse a datatype at the cursor, leaving it just past the datatype.
    ///
    /// Nested types (compound members, array and vlen bases) are encoded inline,
    /// so parsing has to be resumable rather than whole-buffer.
    pub fn parse_at(cur: &mut Cursor<'_>) -> Result<Self> {
        let class_and_version = cur.u8()?;
        let version = class_and_version >> 4;
        let class_code = class_and_version & 0x0F;

        if version == 0 || version > 4 {
            return Err(Error::malformed(format!(
                "datatype message version {version}"
            )));
        }

        let bits0 = cur.u8()? as u32;
        let bits1 = cur.u8()? as u32;
        let bits2 = cur.u8()? as u32;
        let bitfield = bits0 | (bits1 << 8) | (bits2 << 16);
        let size = cur.u32()?;

        let class = match class_code {
            0 => Self::parse_fixed_point(cur, bitfield)?,
            1 => Self::parse_floating_point(cur, bitfield)?,
            3 => Self::parse_string(bitfield)?,
            5 => Self::parse_opaque(cur, bitfield)?,
            6 => Self::parse_compound(cur, version, bitfield, size)?,
            7 => DatatypeClass::Reference {
                kind: (bitfield & 0x0F) as u8,
            },
            8 => Self::parse_enum(cur, version, bitfield)?,
            9 => Self::parse_vlen(cur, bitfield)?,
            10 => Self::parse_array(cur, version)?,
            // Class 2 is Time and class 4 is Bit field. Neither appears in
            // netCDF-4 output, and guessing at their semantics would be worse
            // than saying so.
            other => DatatypeClass::Unsupported { class: other },
        };

        Ok(Self {
            version,
            size,
            class,
        })
    }

    fn parse_fixed_point(cur: &mut Cursor<'_>, bitfield: u32) -> Result<DatatypeClass> {
        let order = if bitfield & 0x01 == 0 {
            ByteOrder::Little
        } else {
            ByteOrder::Big
        };
        let signed = bitfield & 0x08 != 0;
        let bit_offset = cur.u16()?;
        let bit_precision = cur.u16()?;
        Ok(DatatypeClass::FixedPoint {
            order,
            signed,
            bit_offset,
            bit_precision,
        })
    }

    fn parse_floating_point(cur: &mut Cursor<'_>, bitfield: u32) -> Result<DatatypeClass> {
        // Bit 0 is the low bit of the byte order and bit 6 is the high bit.
        // The pair (1,1) means VAX-order floats, which this reader rejects
        // rather than silently mis-decoding.
        let low = bitfield & 0x01 != 0;
        let high = bitfield & 0x40 != 0;
        let order = match (high, low) {
            (false, false) => ByteOrder::Little,
            (false, true) => ByteOrder::Big,
            _ => {
                return Err(Error::unsupported(
                    "VAX-order floating point (mixed-endian)",
                ))
            }
        };
        let sign_location = ((bitfield >> 8) & 0xFF) as u8;

        let bit_offset = cur.u16()?;
        let bit_precision = cur.u16()?;
        let exponent_location = cur.u8()?;
        let exponent_size = cur.u8()?;
        let mantissa_location = cur.u8()?;
        let mantissa_size = cur.u8()?;
        let exponent_bias = cur.u32()?;

        Ok(DatatypeClass::FloatingPoint {
            order,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
            sign_location,
        })
    }

    fn parse_string(bitfield: u32) -> Result<DatatypeClass> {
        Ok(DatatypeClass::String {
            pad: string_pad(bitfield & 0x0F)?,
            charset: char_set((bitfield >> 4) & 0x0F)?,
        })
    }

    fn parse_opaque(cur: &mut Cursor<'_>, bitfield: u32) -> Result<DatatypeClass> {
        // The low byte of the bit field is the tag length, padded to 8 bytes.
        let tag_len = (bitfield & 0xFF) as usize;
        let raw = cur.take(tag_len)?;
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        Ok(DatatypeClass::Opaque {
            tag: String::from_utf8_lossy(&raw[..end]).into_owned(),
        })
    }

    fn parse_compound(
        cur: &mut Cursor<'_>,
        version: u8,
        bitfield: u32,
        size: u32,
    ) -> Result<DatatypeClass> {
        let count = (bitfield & 0xFFFF) as usize;
        let mut members = Vec::with_capacity(count);

        for _ in 0..count {
            match version {
                1 => {
                    let name = cur.cstring_padded(8)?;
                    let offset = cur.u32()? as u64;
                    let rank = cur.u8()?;
                    cur.skip(3)?; // reserved
                    cur.skip(4)?; // dimension permutation
                    cur.skip(4)?; // reserved
                    let mut dims = [0u32; 4];
                    for d in dims.iter_mut() {
                        *d = cur.u32()?;
                    }
                    let base = Self::parse_at(cur)?;
                    // Version 1 members carry an inline array shape. A rank of
                    // zero means the member is a plain scalar of `base`.
                    let datatype = if rank == 0 {
                        base
                    } else {
                        let dims = dims[..rank as usize].to_vec();
                        let size = base.size * dims.iter().product::<u32>();
                        Datatype {
                            version,
                            size,
                            class: DatatypeClass::Array {
                                dims,
                                base: Box::new(base),
                            },
                        }
                    };
                    members.push(CompoundMember {
                        name,
                        offset,
                        datatype,
                    });
                }
                2 => {
                    let name = cur.cstring_padded(8)?;
                    let offset = cur.u32()? as u64;
                    let datatype = Self::parse_at(cur)?;
                    members.push(CompoundMember {
                        name,
                        offset,
                        datatype,
                    });
                }
                3 | 4 => {
                    let name = cur.cstring()?;
                    // The offset field is only as wide as the compound's own
                    // size needs, which the writer minimises.
                    let offset = cur.uint(compound_offset_width(size))?;
                    let datatype = Self::parse_at(cur)?;
                    members.push(CompoundMember {
                        name,
                        offset,
                        datatype,
                    });
                }
                other => {
                    return Err(Error::unsupported(format!(
                        "compound datatype version {other}"
                    )))
                }
            }
        }

        Ok(DatatypeClass::Compound { members })
    }

    fn parse_enum(cur: &mut Cursor<'_>, version: u8, bitfield: u32) -> Result<DatatypeClass> {
        let count = (bitfield & 0xFFFF) as usize;
        let base = Self::parse_at(cur)?;

        let mut names = Vec::with_capacity(count);
        for _ in 0..count {
            if version >= 3 {
                names.push(cur.cstring()?);
            } else {
                names.push(cur.cstring_padded(8)?);
            }
        }

        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(cur.take(base.size as usize)?.to_vec());
        }

        Ok(DatatypeClass::Enum {
            base: Box::new(base),
            names,
            values,
        })
    }

    fn parse_vlen(cur: &mut Cursor<'_>, bitfield: u32) -> Result<DatatypeClass> {
        let kind = match bitfield & 0x0F {
            0 => VlenKind::Sequence,
            1 => VlenKind::String,
            other => {
                return Err(Error::malformed(format!(
                    "variable-length datatype flavour {other}"
                )))
            }
        };
        let pad = string_pad((bitfield >> 4) & 0x0F)?;
        let charset = char_set((bitfield >> 8) & 0x0F)?;
        let base = Self::parse_at(cur)?;
        Ok(DatatypeClass::VariableLength {
            kind,
            base: Box::new(base),
            pad,
            charset,
        })
    }

    fn parse_array(cur: &mut Cursor<'_>, version: u8) -> Result<DatatypeClass> {
        let rank = cur.u8()? as usize;
        if version == 2 {
            cur.skip(3)?; // reserved
        }
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            dims.push(cur.u32()?);
        }
        if version == 2 {
            // Permutation indices, never meaningfully used.
            cur.skip(rank * 4)?;
        }
        let base = Self::parse_at(cur)?;
        Ok(DatatypeClass::Array {
            dims,
            base: Box::new(base),
        })
    }

    /// Byte order, for the classes that have one.
    pub fn byte_order(&self) -> Option<ByteOrder> {
        match &self.class {
            DatatypeClass::FixedPoint { order, .. } => Some(*order),
            DatatypeClass::FloatingPoint { order, .. } => Some(*order),
            _ => None,
        }
    }

    /// Whether values of this type can be decoded by this reader.
    pub fn is_decodable(&self) -> bool {
        match &self.class {
            DatatypeClass::FixedPoint { .. }
            | DatatypeClass::FloatingPoint { .. }
            | DatatypeClass::String { .. }
            | DatatypeClass::Reference { .. }
            | DatatypeClass::Enum { .. }
            | DatatypeClass::VariableLength { .. }
            | DatatypeClass::Opaque { .. } => true,
            DatatypeClass::Array { base, .. } => base.is_decodable(),
            DatatypeClass::Compound { members } => {
                members.iter().all(|m| m.datatype.is_decodable())
            }
            DatatypeClass::Unsupported { .. } => false,
        }
    }
}

/// Width of the member-offset field in a version 3 or 4 compound datatype.
///
/// The writer minimises it against the compound's own size, using
/// `(floor(log2(size)) + 7) / 8` bytes. Deriving it from anything else (the
/// enclosing buffer, say) desynchronises the parse of every later member.
fn compound_offset_width(size: u32) -> u8 {
    if size == 0 {
        return 1;
    }
    let log2 = u32::BITS - 1 - size.leading_zeros();
    (log2.div_ceil(8) as u8).max(1)
}

fn string_pad(code: u32) -> Result<StringPad> {
    Ok(match code {
        0 => StringPad::NullTerminate,
        1 => StringPad::NullPad,
        2 => StringPad::SpacePad,
        other => return Err(Error::malformed(format!("string padding code {other}"))),
    })
}

fn char_set(code: u32) -> Result<CharSet> {
    Ok(match code {
        0 => CharSet::Ascii,
        1 => CharSet::Utf8,
        other => return Err(Error::unsupported(format!("character set code {other}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a datatype header: class/version byte, 3 bit-field bytes, size.
    fn header(class: u8, version: u8, bitfield: u32, size: u32) -> Vec<u8> {
        let mut v = vec![(version << 4) | class];
        v.push((bitfield & 0xFF) as u8);
        v.push(((bitfield >> 8) & 0xFF) as u8);
        v.push(((bitfield >> 16) & 0xFF) as u8);
        v.extend_from_slice(&size.to_le_bytes());
        v
    }

    #[test]
    fn parses_a_signed_little_endian_integer() {
        let mut d = header(0, 1, 0x08, 4);
        d.extend_from_slice(&0u16.to_le_bytes()); // bit offset
        d.extend_from_slice(&32u16.to_le_bytes()); // precision
        let t = Datatype::parse(&d).unwrap();
        assert_eq!(t.size, 4);
        assert_eq!(
            t.class,
            DatatypeClass::FixedPoint {
                order: ByteOrder::Little,
                signed: true,
                bit_offset: 0,
                bit_precision: 32
            }
        );
    }

    #[test]
    fn parses_an_unsigned_big_endian_integer() {
        let mut d = header(0, 1, 0x01, 2);
        d.extend_from_slice(&0u16.to_le_bytes());
        d.extend_from_slice(&16u16.to_le_bytes());
        let t = Datatype::parse(&d).unwrap();
        assert_eq!(t.byte_order(), Some(ByteOrder::Big));
        match t.class {
            DatatypeClass::FixedPoint { signed, .. } => assert!(!signed),
            other => panic!("expected fixed point, got {other:?}"),
        }
    }

    #[test]
    fn parses_a_big_endian_float() {
        // Bit 0 set, bit 6 clear: big endian. Sign at bit 31.
        let mut d = header(1, 1, 0x01 | (31 << 8), 4);
        d.extend_from_slice(&0u16.to_le_bytes());
        d.extend_from_slice(&32u16.to_le_bytes());
        d.push(23); // exponent location
        d.push(8); // exponent size
        d.push(0); // mantissa location
        d.push(23); // mantissa size
        d.extend_from_slice(&127u32.to_le_bytes());
        let t = Datatype::parse(&d).unwrap();
        assert_eq!(t.byte_order(), Some(ByteOrder::Big));
        match t.class {
            DatatypeClass::FloatingPoint {
                exponent_bias,
                sign_location,
                ..
            } => {
                assert_eq!(exponent_bias, 127);
                assert_eq!(sign_location, 31);
            }
            other => panic!("expected floating point, got {other:?}"),
        }
    }

    #[test]
    fn rejects_vax_order_floats() {
        // Both order bits set is the VAX encoding.
        let mut d = header(1, 1, 0x01 | 0x40, 4);
        d.extend_from_slice(&[0u8; 12]);
        let err = Datatype::parse(&d).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }

    #[test]
    fn parses_a_space_padded_ascii_string() {
        // Padding code 2 is space pad, which netCDF character variables use.
        let d = header(3, 1, 0x02, 8);
        let t = Datatype::parse(&d).unwrap();
        assert_eq!(t.size, 8);
        assert_eq!(
            t.class,
            DatatypeClass::String {
                pad: StringPad::SpacePad,
                charset: CharSet::Ascii
            }
        );
    }

    #[test]
    fn parses_a_utf8_null_terminated_string() {
        let d = header(3, 1, 1 << 4, 16);
        let t = Datatype::parse(&d).unwrap();
        assert_eq!(
            t.class,
            DatatypeClass::String {
                pad: StringPad::NullTerminate,
                charset: CharSet::Utf8
            }
        );
    }

    #[test]
    fn parses_an_object_reference() {
        let d = header(7, 1, 0x00, 8);
        let t = Datatype::parse(&d).unwrap();
        assert_eq!(t.class, DatatypeClass::Reference { kind: 0 });
    }

    /// This is the shape netCDF uses for `DIMENSION_LIST`: a variable-length
    /// sequence whose base type is an object reference.
    #[test]
    fn parses_a_vlen_sequence_of_references() {
        let mut d = header(9, 1, 0x00, 16);
        d.extend(header(7, 1, 0x00, 8));
        let t = Datatype::parse(&d).unwrap();
        match t.class {
            DatatypeClass::VariableLength { kind, base, .. } => {
                assert_eq!(kind, VlenKind::Sequence);
                assert_eq!(base.class, DatatypeClass::Reference { kind: 0 });
            }
            other => panic!("expected a variable-length type, got {other:?}"),
        }
    }

    #[test]
    fn parses_a_variable_length_string() {
        let mut d = header(9, 1, 0x01, 16);
        d.extend(header(3, 1, 0x00, 1));
        let t = Datatype::parse(&d).unwrap();
        match t.class {
            DatatypeClass::VariableLength { kind, .. } => assert_eq!(kind, VlenKind::String),
            other => panic!("expected a variable-length type, got {other:?}"),
        }
    }

    #[test]
    fn parses_an_array_type() {
        // Version 3 array: rank, dims, then the base type.
        let mut d = header(10, 3, 0x00, 24);
        d.push(2); // rank
        d.extend_from_slice(&2u32.to_le_bytes());
        d.extend_from_slice(&3u32.to_le_bytes());
        d.extend(header(0, 1, 0x08, 4));
        d.extend_from_slice(&0u16.to_le_bytes());
        d.extend_from_slice(&32u16.to_le_bytes());
        let t = Datatype::parse(&d).unwrap();
        match t.class {
            DatatypeClass::Array { dims, base } => {
                assert_eq!(dims, vec![2, 3]);
                assert_eq!(base.size, 4);
            }
            other => panic!("expected an array type, got {other:?}"),
        }
    }

    #[test]
    fn unknown_classes_are_reported_not_guessed() {
        // Class 2 is Time, which this reader does not decode.
        let d = header(2, 1, 0x00, 8);
        let t = Datatype::parse(&d).unwrap();
        assert_eq!(t.class, DatatypeClass::Unsupported { class: 2 });
        assert!(!t.is_decodable(), "callers must be able to see this");
    }

    #[test]
    fn rejects_an_implausible_version() {
        let d = header(0, 0, 0x00, 4);
        assert!(Datatype::parse(&d).is_err());
    }

    #[test]
    fn decodability_recurses_through_arrays() {
        let mut d = header(10, 3, 0x00, 16);
        d.push(1);
        d.extend_from_slice(&2u32.to_le_bytes());
        d.extend(header(2, 1, 0x00, 8)); // an undecodable Time base
        let t = Datatype::parse(&d).unwrap();
        assert!(!t.is_decodable());
    }
}
