//! Chunk filters for the writer: the pipeline message and the encoders.
//!
//! Each encoder is the inverse of a decoder in [`crate::filters`].

use std::borrow::Cow;
use std::io::Write;

use super::encode::Encoder;
use crate::checksum;
use crate::error::{Error, Result};
use crate::filters::shuffle;
use crate::message::filter::{id, FLAG_OPTIONAL};

/// One filter of a chunked dataset.
///
/// The writer applies the filters of a dataset in their order to each chunk.
/// The usual order is shuffle, then one compression filter.
///
/// # Examples
///
/// ```
/// use oxcdf_hdf5::message::ByteOrder;
/// use oxcdf_hdf5::write::{DatasetSpec, Filter};
/// use oxcdf_hdf5::DType;
///
/// let spec = DatasetSpec::numeric(DType::Float(4), ByteOrder::Little, &[100, 100])?
///     .chunked(&[10, 100])
///     .with_filters(&[Filter::Shuffle, Filter::Deflate(4)]);
/// # Ok::<(), oxcdf_hdf5::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Filter {
    /// Put byte 0 of each element first, then byte 1, and so on.
    ///
    /// It makes numbers smaller for a compression filter that follows.
    Shuffle,
    /// Deflate with a level from 0 to 9.
    Deflate(u8),
    /// Zstandard with a level from 1 to 22. A reader needs the Zstandard plugin.
    Zstd(u8),
    /// Add a Fletcher-32 checksum of the chunk.
    Fletcher32,
}

impl Filter {
    /// The HDF5 filter identifier.
    fn id(self) -> u16 {
        match self {
            Self::Shuffle => id::SHUFFLE,
            Self::Deflate(_) => id::DEFLATE,
            Self::Zstd(_) => id::ZSTD,
            Self::Fletcher32 => id::FLETCHER32,
        }
    }
}

/// The name of the Zstandard filter with its NUL, as netcdf-c writes it.
const ZSTD_NAME: &[u8] = b"zstd\0";

/// Encode a version 2 filter pipeline message body, as netcdf-c writes it.
///
/// `element_size` is the parameter of the shuffle filter. A level outside its
/// range, or a filter that is there two times, is [`Error::BadRequest`].
pub fn encode_filter_pipeline(filters: &[Filter], element_size: u32) -> Result<Vec<u8>> {
    let mut e = Encoder::new();
    e.u8(2); // version
             // Four kinds with no kind two times, so the count fits in one byte.
    e.u8(u8::try_from(filters.len()).unwrap_or(u8::MAX));
    for (position, filter) in filters.iter().enumerate() {
        if filters[..position].iter().any(|f| f.id() == filter.id()) {
            return Err(Error::bad_request(format!(
                "the filter {filter:?} is in the list two times"
            )));
        }
        e.u16(filter.id());
        let (optional, parameter) = match *filter {
            Filter::Shuffle => (true, Some(element_size)),
            Filter::Deflate(level @ 0..=9) => (true, Some(u32::from(level))),
            Filter::Zstd(level @ 1..=22) => {
                // A filter that is not in the HDF5 library has a name.
                e.u16(ZSTD_NAME.len() as u16);
                (true, Some(u32::from(level)))
            }
            Filter::Fletcher32 => (false, None),
            Filter::Deflate(_) | Filter::Zstd(_) => {
                return Err(Error::bad_request(format!(
                    "the level of {filter:?} is outside its range"
                )))
            }
        };
        e.u16(if optional { FLAG_OPTIONAL } else { 0 });
        e.u16(u16::from(parameter.is_some()));
        if matches!(filter, Filter::Zstd(_)) {
            e.bytes(ZSTD_NAME);
        }
        if let Some(parameter) = parameter {
            e.u32(parameter);
        }
    }
    Ok(e.into_bytes())
}

/// Apply `filters` in their order to the bytes of one chunk.
pub fn apply(filters: &[Filter], element_size: usize, chunk: &[u8]) -> Result<Vec<u8>> {
    let mut data = Cow::Borrowed(chunk);
    for filter in filters {
        data = Cow::Owned(match *filter {
            Filter::Shuffle => shuffle(&data, element_size),
            Filter::Deflate(level) => {
                let level = flate2::Compression::new(u32::from(level));
                let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), level);
                encoder.write_all(&data)?;
                encoder.finish()?
            }
            Filter::Zstd(level) => zstd::bulk::compress(&data, i32::from(level))?,
            Filter::Fletcher32 => {
                let sum = checksum::fletcher32(&data).to_le_bytes();
                let mut out = Vec::with_capacity(data.len() + sum.len());
                out.extend_from_slice(&data);
                out.extend_from_slice(&sum);
                out
            }
        });
    }
    Ok(data.into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::decode_chunk;
    use crate::message::filter::FilterPipeline;

    #[test]
    fn the_pipeline_has_the_netcdf_c_bytes() {
        let filters = [Filter::Fletcher32, Filter::Shuffle, Filter::Deflate(4)];
        assert_eq!(
            encode_filter_pipeline(&filters, 4).unwrap(),
            [2, 3, 3, 0, 0, 0, 0, 0, 2, 0, 1, 0, 1, 0, 4, 0, 0, 0, 1, 0, 1, 0, 1, 0, 4, 0, 0, 0]
        );
        assert_eq!(
            encode_filter_pipeline(&[Filter::Zstd(3)], 4).unwrap(),
            [2, 1, 0x0f, 0x7d, 5, 0, 1, 0, 1, 0, b'z', b's', b't', b'd', 0, 3, 0, 0, 0]
        );
    }

    #[test]
    fn wrong_filter_lists_are_bad_requests() {
        for filters in [
            vec![Filter::Deflate(10)],
            vec![Filter::Zstd(0)],
            vec![Filter::Zstd(23)],
            vec![Filter::Shuffle, Filter::Deflate(1), Filter::Shuffle],
            vec![Filter::Deflate(1), Filter::Deflate(2)],
        ] {
            assert!(matches!(
                encode_filter_pipeline(&filters, 4),
                Err(Error::BadRequest(_))
            ));
        }
    }

    #[test]
    fn each_filter_list_decodes_to_the_chunk() {
        let chunk: Vec<u8> = (0..4000u32).flat_map(|i| (i / 7).to_le_bytes()).collect();
        for filters in [
            vec![],
            vec![Filter::Shuffle],
            vec![Filter::Deflate(0)],
            vec![Filter::Shuffle, Filter::Deflate(6)],
            vec![Filter::Shuffle, Filter::Zstd(3)],
            vec![Filter::Fletcher32, Filter::Shuffle, Filter::Deflate(9)],
            vec![Filter::Shuffle, Filter::Zstd(22), Filter::Fletcher32],
        ] {
            let stored = apply(&filters, 4, &chunk).unwrap();
            // Level 0 of deflate stores the bytes as they are.
            let plain =
                |f: &Filter| matches!(f, Filter::Shuffle | Filter::Fletcher32 | Filter::Deflate(0));
            if !filters.iter().all(plain) {
                assert!(stored.len() < chunk.len() / 2, "{filters:?}");
            }
            let body = encode_filter_pipeline(&filters, 4).unwrap();
            let pipeline = FilterPipeline::parse(&body).unwrap();
            let back = decode_chunk(&pipeline, 0, stored, chunk.len()).unwrap();
            assert_eq!(back, chunk, "{filters:?}");
        }
    }
}
