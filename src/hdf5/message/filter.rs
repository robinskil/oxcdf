//! The filter pipeline message: the transforms applied to each chunk.
//!
//! Filters run in listed order when writing, so a reader applies them in
//! reverse. netcdf-c uses shuffle followed by deflate, which means a reader
//! inflates first and unshuffles second.

use crate::cursor::Cursor;
use crate::error::{Error, Result};

/// Registered filter identifiers.
pub mod id {
    /// zlib deflate.
    pub const DEFLATE: u16 = 1;
    /// Byte shuffle, a reversible transform that groups like-significance bytes.
    pub const SHUFFLE: u16 = 2;
    /// Fletcher-32 checksum appended to the chunk.
    pub const FLETCHER32: u16 = 3;
    /// Szip compression.
    pub const SZIP: u16 = 4;
    /// N-bit packing.
    pub const NBIT: u16 = 5;
    /// Scale plus offset packing.
    pub const SCALE_OFFSET: u16 = 6;
    /// Zstandard, registered id.
    pub const ZSTD: u16 = 32015;
    /// Blosc, registered id.
    pub const BLOSC: u16 = 32001;
}

/// Filter flag: the filter may be skipped for a chunk without failing the read.
pub const FLAG_OPTIONAL: u16 = 0x0001;

/// One stage of the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filter {
    /// Registered filter identifier.
    pub id: u16,
    /// Human-readable name, when the file records one.
    pub name: String,
    /// Filter flags.
    pub flags: u16,
    /// Filter-specific parameters.
    pub client_data: Vec<u32>,
}

impl Filter {
    /// Whether a read may skip this filter rather than fail.
    pub fn is_optional(&self) -> bool {
        self.flags & FLAG_OPTIONAL != 0
    }
}

/// The ordered chain of filters applied to every chunk of a dataset.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FilterPipeline {
    /// Filters in write order. Decode runs them in reverse.
    pub filters: Vec<Filter>,
}

impl FilterPipeline {
    /// Parse a filter pipeline message body.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut cur = Cursor::new(data);
        let version = cur.u8()?;
        let count = cur.u8()? as usize;

        match version {
            1 => {
                cur.skip(6)?; // reserved
                let mut filters = Vec::with_capacity(count);
                for _ in 0..count {
                    filters.push(parse_filter_v1(&mut cur)?);
                }
                Ok(Self { filters })
            }
            2 => {
                let mut filters = Vec::with_capacity(count);
                for _ in 0..count {
                    filters.push(parse_filter_v2(&mut cur)?);
                }
                Ok(Self { filters })
            }
            other => Err(Error::unsupported(format!(
                "filter pipeline message version {other}"
            ))),
        }
    }

    /// Whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

fn parse_filter_v1(cur: &mut Cursor<'_>) -> Result<Filter> {
    let id = cur.u16()?;
    let name_len = cur.u16()? as usize;
    let flags = cur.u16()?;
    let n_client = cur.u16()? as usize;

    // Version 1 pads the name to a multiple of 8 bytes.
    let name = if name_len > 0 {
        let padded = name_len.div_ceil(8) * 8;
        let raw = cur.take(padded)?;
        let end = raw.iter().position(|&b| b == 0).unwrap_or(name_len);
        String::from_utf8_lossy(&raw[..end]).into_owned()
    } else {
        String::new()
    };

    let mut client_data = Vec::with_capacity(n_client);
    for _ in 0..n_client {
        client_data.push(cur.u32()?);
    }
    // Version 1 pads the client data block to a multiple of 8 bytes.
    if n_client % 2 == 1 {
        cur.skip(4)?;
    }

    Ok(Filter {
        id,
        name,
        flags,
        client_data,
    })
}

fn parse_filter_v2(cur: &mut Cursor<'_>) -> Result<Filter> {
    let id = cur.u16()?;
    // Version 2 omits the name length for the filters the library defines.
    let name_len = if id < 256 { 0 } else { cur.u16()? as usize };
    let flags = cur.u16()?;
    let n_client = cur.u16()? as usize;

    let name = if name_len > 0 {
        let raw = cur.take(name_len)?;
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        String::from_utf8_lossy(&raw[..end]).into_owned()
    } else {
        String::new()
    };

    let mut client_data = Vec::with_capacity(n_client);
    for _ in 0..n_client {
        client_data.push(cur.u32()?);
    }

    Ok(Filter {
        id,
        name,
        flags,
        client_data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_version_two_shuffle_and_deflate_pipeline() {
        let mut d = vec![2u8, 2];
        // shuffle: id, flags, one client value holding the element size
        d.extend_from_slice(&id::SHUFFLE.to_le_bytes());
        d.extend_from_slice(&0u16.to_le_bytes()); // flags
        d.extend_from_slice(&1u16.to_le_bytes()); // client value count
        d.extend_from_slice(&4u32.to_le_bytes()); // element size
        // deflate: id, flags, one client value holding the level
        d.extend_from_slice(&id::DEFLATE.to_le_bytes());
        d.extend_from_slice(&0u16.to_le_bytes()); // flags
        d.extend_from_slice(&1u16.to_le_bytes()); // client value count
        d.extend_from_slice(&6u32.to_le_bytes()); // level

        let p = FilterPipeline::parse(&d).unwrap();
        assert_eq!(p.filters.len(), 2);
        assert_eq!(p.filters[0].id, id::SHUFFLE);
        assert_eq!(p.filters[0].client_data, vec![4]);
        assert_eq!(p.filters[1].id, id::DEFLATE);
        assert_eq!(p.filters[1].client_data, vec![6]);
    }

    #[test]
    fn parses_a_version_one_pipeline_with_padding() {
        let mut d = vec![1u8, 1, 0, 0, 0, 0, 0, 0];
        d.extend_from_slice(&id::DEFLATE.to_le_bytes());
        d.extend_from_slice(&8u16.to_le_bytes()); // name length
        d.extend_from_slice(&0u16.to_le_bytes()); // flags
        d.extend_from_slice(&1u16.to_le_bytes()); // one client value
        d.extend_from_slice(b"deflate\0"); // 8 bytes, already aligned
        d.extend_from_slice(&9u32.to_le_bytes());
        d.extend_from_slice(&0u32.to_le_bytes()); // pad to 8

        let p = FilterPipeline::parse(&d).unwrap();
        assert_eq!(p.filters.len(), 1);
        assert_eq!(p.filters[0].name, "deflate");
        assert_eq!(p.filters[0].client_data, vec![9]);
    }

    #[test]
    fn a_registered_filter_id_carries_a_name_in_version_two() {
        let mut d = vec![2u8, 1];
        d.extend_from_slice(&id::ZSTD.to_le_bytes());
        d.extend_from_slice(&5u16.to_le_bytes()); // name length
        d.extend_from_slice(&0u16.to_le_bytes()); // flags
        d.extend_from_slice(&0u16.to_le_bytes()); // no client data
        d.extend_from_slice(b"zstd\0");
        let p = FilterPipeline::parse(&d).unwrap();
        assert_eq!(p.filters[0].name, "zstd");
        assert_eq!(p.filters[0].id, id::ZSTD);
    }

    #[test]
    fn reads_the_optional_flag() {
        let mut d = vec![2u8, 1];
        d.extend_from_slice(&id::FLETCHER32.to_le_bytes());
        d.extend_from_slice(&FLAG_OPTIONAL.to_le_bytes());
        d.extend_from_slice(&0u16.to_le_bytes());
        let p = FilterPipeline::parse(&d).unwrap();
        assert!(p.filters[0].is_optional());
    }

    #[test]
    fn an_empty_pipeline_parses() {
        let p = FilterPipeline::parse(&[2u8, 0]).unwrap();
        assert!(p.is_empty());
    }

    #[test]
    fn rejects_an_unknown_pipeline_version() {
        let err = FilterPipeline::parse(&[7u8, 0]).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }
}
