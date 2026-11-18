//! Header message encoding for the writer.
//!
//! Each encoder is the inverse of the parser in [`crate::message`].

use super::encode::Encoder;
use crate::error::{Error, Result};
use crate::message::{
    ByteOrder, CharSet, Dataspace, DataspaceKind, Datatype, DatatypeClass, FillValue, StringPad,
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
/// Integer, float and fixed string classes only. Each other class is
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

/// Encode a version 2 dataspace message body.
///
/// HDF5 limits the rank to one byte. Dimensions past rank 255 are dropped.
pub fn encode_dataspace(space: &Dataspace) -> Vec<u8> {
    let rank = u8::try_from(space.dims.len()).unwrap_or(u8::MAX);
    let mut e = Encoder::new();
    e.u8(2);
    e.u8(rank);
    e.u8(u8::from(space.max_dims.is_some()));
    e.u8(match space.kind {
        DataspaceKind::Scalar => 0,
        DataspaceKind::Simple => 1,
        DataspaceKind::Null => 2,
    });
    for &dim in space.dims.iter().take(usize::from(rank)) {
        e.u64(dim);
    }
    if let Some(max) = &space.max_dims {
        for &dim in max.iter().take(usize::from(rank)) {
            e.u64(dim);
        }
    }
    e.into_bytes()
}

/// Encode a version 3 fill value message body.
pub fn encode_fill_value(fill: &FillValue) -> Vec<u8> {
    let mut e = Encoder::new();
    e.u8(3);
    match &fill.bytes {
        Some(bytes) => {
            // Late allocation, write if set, and a value follows.
            e.u8(0x2A);
            e.u32(u32::try_from(bytes.len()).unwrap_or(u32::MAX));
            e.bytes(bytes);
        }
        None => e.u8(0x0A),
    }
    e.into_bytes()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::Sizes;
    use crate::message::Layout;

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
            let back = Dataspace::parse(&encode_dataspace(&space), Sizes::EIGHT).unwrap();
            assert_eq!(back.kind, space.kind);
            assert_eq!(back.dims, space.dims);
            assert_eq!(back.max_dims, space.max_dims);
        }
    }

    #[test]
    fn fill_values_round_trip() {
        let some = FillValue {
            bytes: Some(vec![1, 2, 3, 4]),
        };
        assert_eq!(FillValue::parse(&encode_fill_value(&some)).unwrap(), some);
        let none = FillValue { bytes: None };
        assert_eq!(FillValue::parse(&encode_fill_value(&none)).unwrap(), none);
    }

    #[test]
    fn a_contiguous_layout_round_trips() {
        for address in [None, Some(65584)] {
            let back =
                Layout::parse(&encode_contiguous_layout(address, 960), Sizes::EIGHT).unwrap();
            assert_eq!(back, Layout::Contiguous { address, size: 960 });
        }
    }
}
