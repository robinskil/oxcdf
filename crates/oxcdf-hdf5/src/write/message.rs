//! Header message encoding for the writer.
//!
//! Each encoder is the inverse of the parser in [`crate::message`].

use super::encode::Encoder;
use crate::error::{Error, Result};
use crate::message::{
    Attribute, ByteOrder, CharSet, Dataspace, DataspaceKind, Datatype, DatatypeClass, FillValue,
    StringPad,
};
use crate::DType;

/// The HDF5 datatype for a numeric netCDF type in the given byte order.
///
/// `Int` and `Uint` take a size of 1, 2, 4 or 8. `Float` takes 4 or 8.
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

/// Encode a version 1 datatype message body.
///
/// It always writes version 1 and ignores [`Datatype::version`]. Integer,
/// float and fixed string classes only. Each other class is
/// [`Error::Unsupported`].
pub fn encode_datatype(datatype: &Datatype) -> Result<Vec<u8>> {
    let (class, bits, properties) = match &datatype.class {
        DatatypeClass::FixedPoint {
            order,
            signed,
            bit_offset,
            bit_precision,
        } => {
            let mut bits0 = order_bit(*order);
            if *signed {
                bits0 |= 0x08;
            }
            let mut p = Encoder::new();
            p.u16(*bit_offset);
            p.u16(*bit_precision);
            (0u8, [bits0, 0, 0], p.into_bytes())
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
            (1u8, [bits0, *sign_location, 0], p.into_bytes())
        }
        DatatypeClass::String { pad, charset } => {
            let pad = match pad {
                StringPad::NullTerminate => 0,
                StringPad::NullPad => 1,
                StringPad::SpacePad => 2,
            };
            let charset = match charset {
                CharSet::Ascii => 0,
                CharSet::Utf8 => 1,
            };
            (3u8, [pad | (charset << 4), 0, 0], Vec::new())
        }
        other => {
            return Err(Error::unsupported(format!(
                "encoding the datatype class {other:?}"
            )))
        }
    };
    let mut e = Encoder::new();
    e.u8(0x10 | class);
    e.bytes(&bits);
    e.u32(datatype.size);
    e.bytes(&properties);
    Ok(e.into_bytes())
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
        _ => e.u8(0x0A),
    }
    Ok(e.into_bytes())
}

/// Encode a version 3 contiguous layout message body.
///
/// `None` writes the undefined address: no storage is allocated yet.
pub fn encode_contiguous_layout(address: Option<u64>, size: u64) -> Vec<u8> {
    let mut e = Encoder::new();
    e.u8(3);
    e.u8(1);
    e.address(address);
    e.u64(size);
    e.into_bytes()
}

/// The largest message body the writer emits, in bytes.
pub const MAX_MESSAGE_BYTES: usize = 65535;

/// Encode a version 0 link info message body for compact storage.
///
/// It tracks creation order and has no fractal heap or name B-tree.
pub fn encode_link_info(max_creation_index: u64) -> Vec<u8> {
    let mut e = Encoder::new();
    e.u8(0);
    e.u8(0x01);
    e.u64(max_creation_index);
    e.address(None);
    e.address(None);
    e.into_bytes()
}

/// Encode a version 0 group info message body with default values.
pub fn encode_group_info() -> Vec<u8> {
    vec![0, 0]
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
    // Width code 0 is a one-byte length, 1 is a two-byte length.
    let width_code: u8 = if len < 256 { 0 } else { 1 };
    // Fixed fields are 19 bytes, plus the length field and the name.
    let total = len.saturating_add(19 + (1usize << width_code));
    if total > MAX_MESSAGE_BYTES {
        return Err(Error::unsupported(format!(
            "link needs {total} bytes, above the {MAX_MESSAGE_BYTES}-byte message limit"
        )));
    }
    let mut e = Encoder::new();
    e.u8(1);
    e.u8(0x14 | width_code);
    e.u64(creation_order);
    e.u8(1);
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
    e.u8(0);
    e.u8(0x01);
    e.u16(max_creation_index);
    e.address(None);
    e.address(None);
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

    let mut e = Encoder::new();
    e.u8(3);
    e.u8(0);
    e.u16(size16(name_size));
    e.u16(size16(datatype.len()));
    e.u16(size16(dataspace.len()));
    e.u8(1);
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
    fn a_compound_datatype_is_unsupported() {
        let dt = Datatype {
            version: 1,
            size: 8,
            class: DatatypeClass::Compound { members: vec![] },
        };
        assert!(matches!(encode_datatype(&dt), Err(Error::Unsupported(_))));
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
