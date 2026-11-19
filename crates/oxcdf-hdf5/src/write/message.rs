//! Header message encoding for the writer.
//!
//! Each encoder is the inverse of the parser in [`crate::message`].

use super::encode::Encoder;
use super::gheap::DESCRIPTOR_BYTES;
use crate::error::{Error, Result};
use crate::message::datatype::CompoundMember;
use crate::message::{
    Attribute, ByteOrder, CharSet, Dataspace, DataspaceKind, Datatype, DatatypeClass, FillValue,
    StringPad, VlenKind,
};
use crate::DType;

/// The HDF5 datatype for variable-length UTF-8 strings.
///
/// It is the datatype that netcdf-c writes for the netCDF string type.
pub fn vlen_string_datatype() -> Datatype {
    let character = Datatype {
        version: 1,
        size: 1,
        class: DatatypeClass::FixedPoint {
            order: ByteOrder::Little,
            signed: false,
            bit_offset: 0,
            bit_precision: 8,
        },
    };
    Datatype {
        version: 1,
        size: DESCRIPTOR_BYTES as u32,
        class: DatatypeClass::VariableLength {
            kind: VlenKind::String,
            base: Box::new(character),
            pad: StringPad::NullTerminate,
            charset: CharSet::Utf8,
        },
    }
}

/// The HDF5 datatype for a numeric netCDF type in the given byte order.
///
/// `Int` and `Uint` take a size of 1, 2, 4 or 8. `Float` takes 4 or 8.
///
/// # Examples
///
/// ```
/// use oxcdf_hdf5::message::ByteOrder;
/// use oxcdf_hdf5::write::numeric_datatype;
/// use oxcdf_hdf5::DType;
///
/// let datatype = numeric_datatype(DType::Float(8), ByteOrder::Big)?;
/// assert_eq!(datatype.size, 8);
/// assert_eq!(DType::of(&datatype), DType::Float(8));
/// # Ok::<(), oxcdf_hdf5::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::Unsupported`] for each other [`DType`] or size.
pub fn numeric_datatype(dtype: DType, order: ByteOrder) -> Result<Datatype> {
    let (size, class) = match dtype {
        DType::Int(n @ (1 | 2 | 4 | 8)) | DType::Uint(n @ (1 | 2 | 4 | 8)) => (
            n,
            DatatypeClass::FixedPoint {
                order,
                signed: matches!(dtype, DType::Int(_)),
                bit_offset: 0,
                bit_precision: u16::from(n) * 8,
            },
        ),
        DType::Float(n @ (4 | 8)) => {
            // The field layouts of IEEE 754 binary32 and binary64.
            let single = n == 4;
            (
                n,
                DatatypeClass::FloatingPoint {
                    order,
                    bit_offset: 0,
                    bit_precision: u16::from(n) * 8,
                    exponent_location: if single { 23 } else { 52 },
                    exponent_size: if single { 8 } else { 11 },
                    mantissa_location: 0,
                    mantissa_size: if single { 23 } else { 52 },
                    exponent_bias: if single { 127 } else { 1023 },
                    sign_location: n * 8 - 1,
                },
            )
        }
        other => {
            return Err(Error::unsupported(format!(
                "no numeric HDF5 datatype for {other:?}"
            )))
        }
    };
    Ok(Datatype {
        version: 1,
        size: u32::from(size),
        class,
    })
}

/// The datatype of a `DIMENSION_LIST` attribute: a sequence of object references.
pub fn dimension_list_datatype() -> Datatype {
    Datatype {
        version: 1,
        size: DESCRIPTOR_BYTES as u32,
        class: DatatypeClass::VariableLength {
            kind: VlenKind::Sequence,
            base: Box::new(object_reference_datatype()),
            pad: StringPad::NullTerminate,
            charset: CharSet::Ascii,
        },
    }
}

/// The datatype of a `REFERENCE_LIST` attribute: a dataset and one of its axes.
pub fn reference_list_datatype() -> Datatype {
    let axis = Datatype {
        version: 1,
        size: 4,
        class: DatatypeClass::FixedPoint {
            order: ByteOrder::Little,
            signed: false,
            bit_offset: 0,
            bit_precision: 32,
        },
    };
    let member = |name: &str, offset, datatype| CompoundMember {
        name: name.to_owned(),
        offset,
        datatype,
    };
    Datatype {
        version: 3,
        // The HDF5 library pads this C struct of 12 bytes to 16 bytes.
        size: 16,
        class: DatatypeClass::Compound {
            members: vec![
                member("dataset", 0, object_reference_datatype()),
                member("dimension", 8, axis),
            ],
        },
    }
}

/// The datatype of a reference to an object header.
fn object_reference_datatype() -> Datatype {
    Datatype {
        version: 1,
        size: 8,
        class: DatatypeClass::Reference { kind: 0 },
    }
}

/// Class code of a fixed-point number.
const CLASS_FIXED_POINT: u8 = 0;
/// Class code of a floating-point number.
const CLASS_FLOATING_POINT: u8 = 1;
/// Class code of a fixed-length string.
const CLASS_STRING: u8 = 3;
/// Class code of a compound.
const CLASS_COMPOUND: u8 = 6;
/// Class code of a reference.
const CLASS_REFERENCE: u8 = 7;
/// Class code of a variable-length sequence or string.
const CLASS_VARIABLE_LENGTH: u8 = 9;

/// Encode a datatype message body.
///
/// A compound gets version 3. Each other class gets version 1.
/// The encoder ignores [`Datatype::version`].
///
/// The encoder supports integers, floats, fixed strings, variable-length
/// strings, object references, sequences and compounds. Each other class is
/// [`Error::Unsupported`]. A size of 0 bytes, a bit field outside the size or
/// a compound with no members is [`Error::BadRequest`]. The HDF5 library
/// rejects these datatypes.
pub fn encode_datatype(datatype: &Datatype) -> Result<Vec<u8>> {
    if datatype.size == 0 {
        return Err(Error::bad_request("a datatype has a size of 0 bytes"));
    }
    let (class, bits, properties) = match &datatype.class {
        DatatypeClass::FixedPoint {
            order,
            signed,
            bit_offset,
            bit_precision,
        } => {
            check_precision(datatype.size, *bit_offset, *bit_precision)?;
            let mut bits0 = order_bit(*order);
            if *signed {
                // Bit 3 marks a two's complement number.
                bits0 |= 0x08;
            }
            let mut p = Encoder::new();
            p.u16(*bit_offset);
            p.u16(*bit_precision);
            (CLASS_FIXED_POINT, [bits0, 0, 0], p.into_bytes())
        }
        DatatypeClass::FloatingPoint {
            order,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
            sign_location,
        } => {
            check_precision(datatype.size, *bit_offset, *bit_precision)?;
            let sign = (*sign_location, 1);
            let exponent = (*exponent_location, *exponent_size);
            let mantissa = (*mantissa_location, *mantissa_size);
            let end = |field: (u8, u8)| u16::from(field.0) + u16::from(field.1);
            let inside = |field: (u8, u8)| field.1 > 0 && end(field) <= *bit_precision;
            let apart =
                |a: (u8, u8), b: (u8, u8)| end(a) <= u16::from(b.0) || end(b) <= u16::from(a.0);
            let valid = inside(sign)
                && inside(exponent)
                && inside(mantissa)
                && apart(sign, exponent)
                && apart(sign, mantissa)
                && apart(exponent, mantissa);
            if !valid {
                return Err(Error::bad_request(format!(
                    "the sign, exponent and mantissa fields do not fit apart in {bit_precision} bits"
                )));
            }
            // 0x20 marks the implied leading mantissa bit of IEEE floats.
            let bits0 = order_bit(*order) | 0x20;
            let mut p = Encoder::new();
            p.u16(*bit_offset);
            p.u16(*bit_precision);
            p.u8(*exponent_location);
            p.u8(*exponent_size);
            p.u8(*mantissa_location);
            p.u8(*mantissa_size);
            p.u32(*exponent_bias);
            (
                CLASS_FLOATING_POINT,
                [bits0, *sign_location, 0],
                p.into_bytes(),
            )
        }
        DatatypeClass::String { pad, charset } => (
            CLASS_STRING,
            [pad_code(*pad) | (charset_code(*charset) << 4), 0, 0],
            Vec::new(),
        ),
        DatatypeClass::VariableLength {
            kind: VlenKind::String,
            base,
            pad,
            charset,
        } => {
            if datatype.size as usize != DESCRIPTOR_BYTES || base.size != 1 {
                return Err(Error::bad_request(format!(
                    "a variable-length string has {DESCRIPTOR_BYTES}-byte elements and a 1-byte base"
                )));
            }
            // Bits 0 to 3 hold the kind: 1 is a string.
            let bits = [0x01 | (pad_code(*pad) << 4), charset_code(*charset), 0];
            (CLASS_VARIABLE_LENGTH, bits, encode_datatype(base)?)
        }
        DatatypeClass::VariableLength {
            kind: VlenKind::Sequence,
            base,
            ..
        } => {
            if datatype.size as usize != DESCRIPTOR_BYTES {
                return Err(Error::bad_request(format!(
                    "a sequence has {DESCRIPTOR_BYTES}-byte elements"
                )));
            }
            (CLASS_VARIABLE_LENGTH, [0, 0, 0], encode_datatype(base)?)
        }
        DatatypeClass::Reference { kind: 0 } => {
            if datatype.size != 8 {
                return Err(Error::bad_request("an object reference has 8 bytes"));
            }
            (CLASS_REFERENCE, [0, 0, 0], Vec::new())
        }
        DatatypeClass::Compound { members } => {
            if members.is_empty() {
                return Err(Error::bad_request("a compound has no members"));
            }
            let count = u16::try_from(members.len())
                .map_err(|_| Error::unsupported("a compound with more than 65535 members"))?;
            // An offset takes the smallest number of bytes that holds the size.
            let width = (1..=4usize)
                .find(|&n| u64::from(datatype.size) < 1u64 << (8 * n))
                .unwrap_or(4);
            let mut p = Encoder::new();
            for member in members {
                let end = member.offset.checked_add(u64::from(member.datatype.size));
                let inside = end.is_some_and(|end| end <= u64::from(datatype.size));
                if !inside || member.name.is_empty() || member.name.contains('\0') {
                    return Err(Error::bad_request(format!(
                        "the compound member {:?} has a wrong name or lies outside the element",
                        member.name
                    )));
                }
                p.bytes(member.name.as_bytes());
                p.u8(0);
                p.bytes(&member.offset.to_le_bytes()[..width]);
                p.bytes(&encode_datatype(&member.datatype)?);
            }
            let [low, high] = count.to_le_bytes();
            let mut e = Encoder::new();
            // The high nibble holds the version.
            e.u8((3 << 4) | CLASS_COMPOUND);
            e.bytes(&[low, high, 0]);
            e.u32(datatype.size);
            e.bytes(&p.into_bytes());
            return Ok(e.into_bytes());
        }
        other => {
            return Err(Error::unsupported(format!(
                "encoding the datatype class {other:?}"
            )))
        }
    };
    let mut e = Encoder::new();
    // The high nibble holds the version.
    e.u8((1 << 4) | class);
    e.bytes(&bits);
    e.u32(datatype.size);
    e.bytes(&properties);
    Ok(e.into_bytes())
}

/// The code of a string pad rule.
fn pad_code(pad: StringPad) -> u8 {
    match pad {
        StringPad::NullTerminate => 0,
        StringPad::NullPad => 1,
        StringPad::SpacePad => 2,
    }
}

/// The code of a character set.
fn charset_code(charset: CharSet) -> u8 {
    match charset {
        CharSet::Ascii => 0,
        CharSet::Utf8 => 1,
    }
}

/// Reject a precision of zero bits, or bits that pass the end of the datatype.
fn check_precision(size: u32, bit_offset: u16, bit_precision: u16) -> Result<()> {
    let bits = u64::from(size) * 8;
    if bit_precision == 0 || u64::from(bit_offset) + u64::from(bit_precision) > bits {
        return Err(Error::bad_request(format!(
            "a precision of {bit_precision} bits at offset {bit_offset} does not fit in {size} bytes"
        )));
    }
    Ok(())
}

/// Bit 0 of the first bit-field byte: set for big-endian.
fn order_bit(order: ByteOrder) -> u8 {
    match order {
        ByteOrder::Little => 0,
        ByteOrder::Big => 1,
    }
}

/// The highest dataspace rank the HDF5 library accepts.
pub const MAX_RANK: usize = 32;

/// Encode a version 2 dataspace message body.
///
/// A rank above [`MAX_RANK`], a `max_dims` of another length, or a scalar or
/// null space with dimensions is [`Error::BadRequest`].
pub fn encode_dataspace(space: &Dataspace) -> Result<Vec<u8>> {
    let len = space.dims.len();
    if len > MAX_RANK {
        return Err(Error::bad_request(format!(
            "dataspace rank {len} is above the HDF5 limit of {MAX_RANK}"
        )));
    }
    if let Some(max) = &space.max_dims {
        if max.len() != len {
            return Err(Error::bad_request(format!(
                "dataspace has {len} dimensions but {} maximum dimensions",
                max.len()
            )));
        }
    }
    if space.kind != DataspaceKind::Simple && len != 0 {
        return Err(Error::bad_request(format!(
            "a {:?} dataspace has no dimensions, but this one has {len}",
            space.kind
        )));
    }
    // The rank is at most MAX_RANK, so it fits in one byte.
    let rank = u8::try_from(len).unwrap_or(u8::MAX);
    let mut e = Encoder::new();
    e.u8(2);
    e.u8(rank);
    e.u8(u8::from(space.max_dims.is_some()));
    e.u8(match space.kind {
        DataspaceKind::Scalar => 0,
        DataspaceKind::Simple => 1,
        DataspaceKind::Null => 2,
    });
    for &dim in &space.dims {
        e.u64(dim);
    }
    for &dim in space.max_dims.iter().flatten() {
        e.u64(dim);
    }
    Ok(e.into_bytes())
}

/// Encode a version 3 fill value message body.
///
/// An empty value encodes as no fill value. A body above
/// [`MAX_MESSAGE_BYTES`] is [`Error::Unsupported`].
pub fn encode_fill_value(fill: &FillValue) -> Result<Vec<u8>> {
    let mut e = Encoder::new();
    e.u8(3);
    match fill.bytes.as_deref() {
        Some(bytes) if !bytes.is_empty() => {
            let total = bytes.len().saturating_add(6);
            if total > MAX_MESSAGE_BYTES {
                return Err(Error::unsupported(format!(
                    "fill value needs {total} bytes, above the {MAX_MESSAGE_BYTES}-byte message limit"
                )));
            }
            // Late allocation, write if set, and a value follows.
            e.u8(0x2A);
            e.u32(u32::try_from(bytes.len()).unwrap_or(u32::MAX));
            e.bytes(bytes);
        }
        // Late allocation, write if set, and no value.
        _ => e.u8(0x0A),
    }
    Ok(e.into_bytes())
}

/// Encode a version 3 contiguous layout message body.
///
/// `None` writes the undefined address: no storage is allocated yet.
pub fn encode_contiguous_layout(address: Option<u64>, size: u64) -> Vec<u8> {
    let mut e = Encoder::new();
    e.u8(3); // version
    e.u8(1); // layout class: contiguous
    e.address(address);
    e.u64(size);
    e.into_bytes()
}

/// Encode a version 3 chunked layout message body with a version 1 B-tree index.
///
/// `None` writes the undefined address: the dataset has no chunk yet.
/// `chunk_dims` holds at most [`MAX_RANK`] sizes.
pub fn encode_chunked_layout(
    address: Option<u64>,
    chunk_dims: &[u32],
    element_size: u32,
) -> Vec<u8> {
    let mut e = Encoder::new();
    e.u8(3); // version
    e.u8(2); // layout class: chunked
             // The element size is one more dimension.
    e.u8(u8::try_from(chunk_dims.len() + 1).unwrap_or(u8::MAX));
    e.address(address);
    for &dim in chunk_dims {
        e.u32(dim);
    }
    e.u32(element_size);
    e.into_bytes()
}

/// Encode a version 3 fill value message body for a chunked dataset.
///
/// It is [`encode_fill_value`] with incremental space allocation, the HDF5
/// library default for chunks.
pub fn encode_chunked_fill_value(fill: &FillValue) -> Result<Vec<u8>> {
    let mut body = encode_fill_value(fill)?;
    if let Some(flags) = body.get_mut(1) {
        // Bits 0 and 1 hold the allocation time: 2 is late, 3 is incremental.
        *flags |= 0x01;
    }
    Ok(body)
}

/// The largest message body the writer emits, in bytes.
pub const MAX_MESSAGE_BYTES: usize = 65535;

/// Encode a version 0 link info message body for compact storage.
///
/// It tracks creation order and has no fractal heap or name B-tree.
pub fn encode_link_info(max_creation_index: u64) -> Vec<u8> {
    let mut e = Encoder::new();
    e.u8(0); // version
    e.u8(0x01); // flags: creation order tracked
    e.u64(max_creation_index);
    e.address(None); // fractal heap
    e.address(None); // name index B-tree
    e.into_bytes()
}

/// Encode a version 0 group info message body with default values.
pub fn encode_group_info() -> Vec<u8> {
    vec![0, 0]
}

/// Width code of the name length field: 0 is one byte, 1 is two bytes.
fn link_width_code(name: &str) -> u8 {
    u8::from(name.len() >= 256)
}

/// Exact length of the body that [`encode_hard_link`] returns for `name`.
pub fn hard_link_len(name: &str) -> usize {
    // Fixed fields are 19 bytes, plus the length field and the name.
    name.len()
        .saturating_add(19 + (1usize << link_width_code(name)))
}

/// Encode a version 1 hard link message body with a UTF-8 name.
///
/// An empty name is [`Error::BadRequest`]. A body above
/// [`MAX_MESSAGE_BYTES`] is [`Error::Unsupported`].
pub fn encode_hard_link(name: &str, address: u64, creation_order: u64) -> Result<Vec<u8>> {
    let len = name.len();
    if len == 0 {
        return Err(Error::bad_request("a link name is empty"));
    }
    let width_code = link_width_code(name);
    let total = hard_link_len(name);
    if total > MAX_MESSAGE_BYTES {
        return Err(Error::unsupported(format!(
            "link needs {total} bytes, above the {MAX_MESSAGE_BYTES}-byte message limit"
        )));
    }
    let mut e = Encoder::new();
    e.u8(1); // version
             // Flags: bit 2 marks a creation order, bit 4 marks a character set.
    e.u8(0x14 | width_code);
    e.u64(creation_order);
    e.u8(1); // character set: UTF-8
    e.uint(len as u64, 1 << width_code);
    e.bytes(name.as_bytes());
    e.u64(address);
    Ok(e.into_bytes())
}

/// Encode a version 0 attribute info message body for compact storage.
///
/// It records the maximum creation index and has no fractal heap.
pub fn encode_attribute_info(max_creation_index: u16) -> Vec<u8> {
    let mut e = Encoder::new();
    e.u8(0); // version
    e.u8(0x01); // flags: creation order tracked
    e.u16(max_creation_index);
    e.address(None); // fractal heap
    e.address(None); // name index B-tree
    e.into_bytes()
}

/// Encode a version 3 attribute message body with a UTF-8 name.
///
/// A wrong dataspace, a wrong data length or a NUL in the name is
/// [`Error::BadRequest`]. A body above [`MAX_MESSAGE_BYTES`] is
/// [`Error::Unsupported`].
pub fn encode_attribute(attribute: &Attribute) -> Result<Vec<u8>> {
    if attribute.name.as_bytes().contains(&0) {
        return Err(Error::bad_request(format!(
            "attribute name {:?} holds a NUL byte",
            attribute.name
        )));
    }
    let dataspace = encode_dataspace(&attribute.dataspace)?;
    let expected = attribute
        .dataspace
        .element_count()
        .checked_mul(u64::from(attribute.datatype.size));
    if expected != u64::try_from(attribute.data.len()).ok() {
        return Err(Error::bad_request(format!(
            "attribute {:?} has {} data bytes, but its shape and type need {}",
            attribute.name,
            attribute.data.len(),
            expected.map_or_else(|| "more than u64::MAX".to_string(), |n| n.to_string())
        )));
    }

    let datatype = encode_datatype(&attribute.datatype)?;
    let name_size = attribute.name.len().saturating_add(1);
    // The fixed fields are 9 bytes: version, flags, three sizes and the character set.
    let total = [
        name_size,
        datatype.len(),
        dataspace.len(),
        attribute.data.len(),
    ]
    .iter()
    .fold(9usize, |sum, &n| sum.saturating_add(n));
    if total > MAX_MESSAGE_BYTES {
        return Err(Error::unsupported(format!(
            "attribute {:?} needs {total} bytes, above the {MAX_MESSAGE_BYTES}-byte message limit",
            attribute.name
        )));
    }
    // Each part is below the total, so each size fits in a u16.
    let size16 = |n: usize| u16::try_from(n).unwrap_or(u16::MAX);

    let mut e = Encoder::with_capacity(total);
    e.u8(3); // version
    e.u8(0); // flags
    e.u16(size16(name_size));
    e.u16(size16(datatype.len()));
    e.u16(size16(dataspace.len()));
    e.u8(1); // character set: UTF-8
    e.bytes(attribute.name.as_bytes());
    e.u8(0);
    e.bytes(&datatype);
    e.bytes(&dataspace);
    e.bytes(&attribute.data);
    Ok(e.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::Sizes;
    use crate::message::{AttributeInfo, Layout, Link, LinkInfo};

    #[test]
    fn an_f32_little_endian_datatype_has_the_library_bytes() {
        let dt = numeric_datatype(DType::Float(4), ByteOrder::Little).unwrap();
        assert_eq!(
            encode_datatype(&dt).unwrap(),
            [
                0x11, 0x20, 0x1F, 0x00, 4, 0, 0, 0, 0, 0, 0x20, 0, 0x17, 0x08, 0x00, 0x17, 0x7F, 0,
                0, 0
            ]
        );
    }

    #[test]
    fn an_i32_little_endian_datatype_has_the_library_bytes() {
        let dt = numeric_datatype(DType::Int(4), ByteOrder::Little).unwrap();
        assert_eq!(
            encode_datatype(&dt).unwrap(),
            [0x10, 0x08, 0, 0, 4, 0, 0, 0, 0, 0, 0x20, 0]
        );
    }

    #[test]
    fn every_numeric_datatype_round_trips() {
        for order in [ByteOrder::Little, ByteOrder::Big] {
            for dtype in [
                DType::Int(1),
                DType::Int(2),
                DType::Int(4),
                DType::Int(8),
                DType::Uint(1),
                DType::Uint(2),
                DType::Uint(4),
                DType::Uint(8),
                DType::Float(4),
                DType::Float(8),
            ] {
                let dt = numeric_datatype(dtype.clone(), order).unwrap();
                assert_eq!(Datatype::parse(&encode_datatype(&dt).unwrap()).unwrap(), dt);
                assert_eq!(DType::of(&dt), dtype);
            }
        }
    }

    #[test]
    fn a_fixed_string_datatype_round_trips() {
        let dt = Datatype {
            version: 1,
            size: 12,
            class: DatatypeClass::String {
                pad: StringPad::NullPad,
                charset: CharSet::Utf8,
            },
        };
        assert_eq!(Datatype::parse(&encode_datatype(&dt).unwrap()).unwrap(), dt);
    }

    #[test]
    fn a_datatype_the_library_rejects_is_a_bad_request() {
        let string = |size| Datatype {
            version: 1,
            size,
            class: DatatypeClass::String {
                pad: StringPad::NullTerminate,
                charset: CharSet::Ascii,
            },
        };
        assert!(matches!(
            encode_datatype(&string(0)),
            Err(Error::BadRequest(_))
        ));
        assert!(encode_datatype(&string(1)).is_ok());

        let int = |size, bit_offset, bit_precision| Datatype {
            version: 1,
            size,
            class: DatatypeClass::FixedPoint {
                order: ByteOrder::Little,
                signed: true,
                bit_offset,
                bit_precision,
            },
        };
        for (size, offset, precision) in [(0, 0, 8), (4, 0, 0), (4, 0, 33), (4, 8, 32)] {
            assert!(
                matches!(
                    encode_datatype(&int(size, offset, precision)),
                    Err(Error::BadRequest(_))
                ),
                "{size} {offset} {precision}"
            );
        }
        assert!(encode_datatype(&int(4, 8, 24)).is_ok());
        assert!(encode_datatype(&int(260, 0, 2080)).is_ok());

        let float = |change: fn(&mut DatatypeClass)| {
            let mut dt = numeric_datatype(DType::Float(4), ByteOrder::Little).unwrap();
            change(&mut dt.class);
            encode_datatype(&dt)
        };
        let wrong: [fn(&mut DatatypeClass); 5] = [
            |c| {
                if let DatatypeClass::FloatingPoint { bit_precision, .. } = c {
                    *bit_precision = 64;
                }
            },
            |c| {
                if let DatatypeClass::FloatingPoint { sign_location, .. } = c {
                    *sign_location = 32;
                }
            },
            |c| {
                if let DatatypeClass::FloatingPoint { exponent_size, .. } = c {
                    *exponent_size = 0;
                }
            },
            |c| {
                if let DatatypeClass::FloatingPoint { mantissa_size, .. } = c {
                    *mantissa_size = 24;
                }
            },
            |c| {
                if let DatatypeClass::FloatingPoint { sign_location, .. } = c {
                    *sign_location = 30;
                }
            },
        ];
        for change in wrong {
            assert!(matches!(float(change), Err(Error::BadRequest(_))));
        }
    }

    #[test]
    fn the_link_length_is_the_encoded_length() {
        for name in ["a", &"n".repeat(255), &"n".repeat(256), &"n".repeat(300)] {
            assert_eq!(
                hard_link_len(name),
                encode_hard_link(name, 48, 0).unwrap().len()
            );
        }
    }

    #[test]
    fn a_vlen_string_datatype_has_the_library_bytes() {
        let dt = vlen_string_datatype();
        let body = encode_datatype(&dt).unwrap();
        assert_eq!(
            body,
            [0x19, 0x01, 0x01, 0, 16, 0, 0, 0, 0x10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 8, 0]
        );
        assert_eq!(Datatype::parse(&body).unwrap(), dt);

        let mut wide = dt;
        wide.size = 24;
        assert!(matches!(encode_datatype(&wide), Err(Error::BadRequest(_))));
    }

    #[test]
    fn a_region_reference_datatype_is_unsupported() {
        let dt = Datatype {
            version: 1,
            size: 12,
            class: DatatypeClass::Reference { kind: 1 },
        };
        assert!(matches!(encode_datatype(&dt), Err(Error::Unsupported(_))));
    }

    #[test]
    fn the_scale_list_datatypes_parse_back() {
        for dt in [dimension_list_datatype(), reference_list_datatype()] {
            let body = encode_datatype(&dt).unwrap();
            assert_eq!(Datatype::parse(&body).unwrap(), dt);
        }
    }

    #[test]
    fn a_compound_member_outside_the_element_is_a_bad_request() {
        let mut dt = reference_list_datatype();
        dt.size = 8;
        assert!(matches!(encode_datatype(&dt), Err(Error::BadRequest(_))));
    }

    #[test]
    fn a_compound_with_no_members_is_a_bad_request() {
        let dt = Datatype {
            version: 3,
            size: 8,
            class: DatatypeClass::Compound {
                members: Vec::new(),
            },
        };
        assert!(matches!(encode_datatype(&dt), Err(Error::BadRequest(_))));
    }

    #[test]
    fn dataspaces_round_trip() {
        for (kind, dims, max) in [
            (DataspaceKind::Scalar, vec![], None),
            (DataspaceKind::Simple, vec![3, 0, 5], None),
            (DataspaceKind::Simple, vec![2, 4], Some(vec![u64::MAX, 4])),
            (DataspaceKind::Null, vec![], None),
        ] {
            let space = Dataspace {
                kind,
                dims,
                max_dims: max,
            };
            let back = Dataspace::parse(&encode_dataspace(&space).unwrap(), Sizes::EIGHT).unwrap();
            assert_eq!(back.kind, space.kind);
            assert_eq!(back.dims, space.dims);
            assert_eq!(back.max_dims, space.max_dims);
        }
    }

    #[test]
    fn a_dataspace_with_a_wrong_shape_is_a_bad_request() {
        for (kind, dims, max_dims) in [
            (DataspaceKind::Simple, vec![1; 33], None),
            (DataspaceKind::Simple, vec![2, 3], Some(vec![2])),
            (DataspaceKind::Simple, vec![2], Some(vec![2, 3])),
            (DataspaceKind::Scalar, vec![2], None),
            (DataspaceKind::Null, vec![2], None),
        ] {
            let space = Dataspace {
                kind,
                dims,
                max_dims,
            };
            assert!(matches!(
                encode_dataspace(&space),
                Err(Error::BadRequest(_))
            ));
        }
        let rank32 = Dataspace {
            kind: DataspaceKind::Simple,
            dims: vec![1; 32],
            max_dims: None,
        };
        assert!(encode_dataspace(&rank32).is_ok());
    }

    #[test]
    fn fill_values_round_trip() {
        let some = FillValue {
            bytes: Some(vec![1, 2, 3, 4]),
        };
        assert_eq!(
            FillValue::parse(&encode_fill_value(&some).unwrap()).unwrap(),
            some
        );
        let none = FillValue { bytes: None };
        assert_eq!(
            FillValue::parse(&encode_fill_value(&none).unwrap()).unwrap(),
            none
        );
    }

    #[test]
    fn an_empty_fill_value_encodes_as_no_fill_value() {
        assert_eq!(
            encode_fill_value(&FillValue {
                bytes: Some(vec![])
            })
            .unwrap(),
            encode_fill_value(&FillValue { bytes: None }).unwrap()
        );
    }

    #[test]
    fn a_contiguous_layout_round_trips() {
        for address in [None, Some(65584)] {
            let back =
                Layout::parse(&encode_contiguous_layout(address, 960), Sizes::EIGHT).unwrap();
            assert_eq!(back, Layout::Contiguous { address, size: 960 });
        }
    }

    #[test]
    fn a_chunked_layout_round_trips() {
        for address in [None, Some(4096)] {
            let body = encode_chunked_layout(address, &[10, 20], 8);
            let back = Layout::parse(&body, Sizes::EIGHT).unwrap();
            assert_eq!(
                back,
                Layout::Chunked {
                    address,
                    chunk_dims: vec![10, 20],
                    element_size: 8,
                    index: crate::message::ChunkIndex::BtreeV1,
                }
            );
        }
        let fill = FillValue {
            bytes: Some(vec![1, 2]),
        };
        let body = encode_chunked_fill_value(&fill).unwrap();
        assert_eq!(body[..2], [3, 0x2B]);
        assert_eq!(FillValue::parse(&body).unwrap(), fill);
    }

    #[test]
    fn a_hard_link_round_trips() {
        for name in ["temp", "zee\u{00eb}n", &"n".repeat(300)] {
            let link =
                Link::parse(&encode_hard_link(name, 4096, 7).unwrap(), Sizes::EIGHT).unwrap();
            assert_eq!(link.name, name);
            assert_eq!(link.hard_address(), Some(4096));
            assert_eq!(link.creation_order, Some(7));
        }
    }

    #[test]
    fn wrong_link_names_are_rejected() {
        assert!(matches!(
            encode_hard_link("", 48, 0),
            Err(Error::BadRequest(_))
        ));
        assert!(matches!(
            encode_hard_link(&"n".repeat(70000), 48, 0),
            Err(Error::Unsupported(_))
        ));
        assert!(encode_hard_link(&"n".repeat(300), 48, 0).is_ok());
    }

    #[test]
    fn link_info_is_compact_and_tracks_creation_order() {
        let body = encode_link_info(12);
        assert_eq!(body.len(), 26);
        let info = LinkInfo::parse(&body, Sizes::EIGHT).unwrap();
        assert_eq!(info.max_creation_index, Some(12));
        assert!(!info.is_dense());
    }

    #[test]
    fn attribute_info_is_compact() {
        let body = encode_attribute_info(3);
        assert_eq!(body.len(), 20);
        assert!(!AttributeInfo::parse(&body, Sizes::EIGHT)
            .unwrap()
            .is_dense());
    }

    #[test]
    fn an_attribute_round_trips() {
        let attribute = Attribute {
            name: "scale_factor".into(),
            datatype: numeric_datatype(DType::Float(8), ByteOrder::Little).unwrap(),
            dataspace: Dataspace {
                kind: DataspaceKind::Simple,
                dims: vec![2],
                max_dims: None,
            },
            data: [0.5f64.to_le_bytes(), 2.0f64.to_le_bytes()].concat(),
        };
        let back = Attribute::parse(&encode_attribute(&attribute).unwrap(), Sizes::EIGHT).unwrap();
        assert_eq!(back.name, attribute.name);
        assert_eq!(back.datatype, attribute.datatype);
        assert_eq!(back.dataspace.dims, vec![2]);
        assert_eq!(back.data, attribute.data);
    }

    #[test]
    fn an_attribute_with_a_wrong_dataspace_is_a_bad_request() {
        for dataspace in [
            Dataspace {
                kind: DataspaceKind::Scalar,
                dims: vec![1],
                max_dims: None,
            },
            Dataspace {
                kind: DataspaceKind::Simple,
                dims: vec![1],
                max_dims: Some(vec![1, 1]),
            },
        ] {
            let attribute = Attribute {
                name: "n".into(),
                datatype: numeric_datatype(DType::Int(4), ByteOrder::Little).unwrap(),
                dataspace,
                data: vec![0; 4],
            };
            assert!(matches!(
                encode_attribute(&attribute),
                Err(Error::BadRequest(_))
            ));
        }
    }

    #[test]
    fn an_oversize_attribute_is_unsupported() {
        let attribute = Attribute {
            name: "big".into(),
            datatype: numeric_datatype(DType::Uint(1), ByteOrder::Little).unwrap(),
            dataspace: Dataspace {
                kind: DataspaceKind::Simple,
                dims: vec![70000],
                max_dims: None,
            },
            data: vec![0; 70000],
        };
        assert!(matches!(
            encode_attribute(&attribute),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn an_attribute_with_a_wrong_data_length_is_a_bad_request() {
        let attribute = Attribute {
            name: "n".into(),
            datatype: numeric_datatype(DType::Int(4), ByteOrder::Little).unwrap(),
            dataspace: Dataspace {
                kind: DataspaceKind::Scalar,
                dims: vec![],
                max_dims: None,
            },
            data: vec![0; 3],
        };
        assert!(matches!(
            encode_attribute(&attribute),
            Err(Error::BadRequest(_))
        ));
    }
}
