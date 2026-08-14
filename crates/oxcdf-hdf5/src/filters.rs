//! Chunk filters.
//!
//! A writer applies a chain of filters to each chunk. A reader undoes them in
//! reverse order. netcdf-c writes shuffle, then deflate. A reader inflates
//! first, then unshuffles.
//!
//! Every function here is pure. Bytes go in. Bytes come out. Nothing is shared.
//! Many threads can decode different chunks at the same time.

use std::io::Read;

use crate::checksum;
use crate::error::{Error, Result};
use crate::message::filter::{id, Filter, FilterPipeline};

/// Undo a whole pipeline for one chunk.
///
/// `filter_mask` comes from the chunk's index record. A set bit means the
/// filter at that position was skipped when the chunk was written, so it must
/// be skipped when reading too. HDF5 uses this for chunks that compression made
/// bigger.
///
/// `expected_len` is the chunk's size once decoded, which the caller knows from
/// the chunk shape and element size. Some filters need it to size their output.
pub fn decode_chunk(
    pipeline: &FilterPipeline,
    filter_mask: u32,
    data: Vec<u8>,
    expected_len: usize,
) -> Result<Vec<u8>> {
    let mut buf = data;

    // Filters are listed in write order, so undo them from the back.
    for (index, filter) in pipeline.filters.iter().enumerate().rev() {
        if filter_mask & (1u32 << index) != 0 {
            continue; // skipped when written
        }
        buf = decode_one(filter, buf, expected_len)?;
    }

    Ok(buf)
}

/// Undo a single filter.
fn decode_one(filter: &Filter, data: Vec<u8>, expected_len: usize) -> Result<Vec<u8>> {
    match filter.id {
        id::DEFLATE => inflate(&data, expected_len),
        id::SHUFFLE => {
            // The element size is the filter's one client parameter. A zero or
            // missing value means there is nothing to undo.
            let element_size = filter.client_data.first().copied().unwrap_or(0) as usize;
            Ok(unshuffle(&data, element_size))
        }
        id::FLETCHER32 => verify_and_strip_fletcher32(data),
        id::ZSTD => unzstd(&data, expected_len),
        id::BLOSC => unblosc(&data),
        other => {
            // The "optional" flag says the *writer* could skip this filter if it
            // was unavailable. It says nothing about whether the filter was
            // actually applied to this chunk; the filter mask does, and it has
            // already been consulted above. Passing the bytes through here would
            // hand back still-compressed data as if they were values.
            Err(Error::unsupported(format!(
                "chunk filter {other}{}",
                if filter.name.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", filter.name)
                }
            )))
        }
    }
}

/// Whether every filter in a pipeline can be undone by this reader.
///
/// Callers use this at index time to decide, per variable, whether to fall back
/// to netcdf-c before any data is read.
pub fn pipeline_is_supported(pipeline: &FilterPipeline) -> bool {
    pipeline.filters.iter().all(|f| {
        matches!(
            f.id,
            id::DEFLATE | id::SHUFFLE | id::FLETCHER32 | id::ZSTD | id::BLOSC
        )
    })
}

/// Decompress a Zstandard-compressed chunk.
pub fn unzstd(data: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expected_len);
    let mut decoder = zstd::stream::read::Decoder::new(data)
        .map_err(|e| Error::malformed(format!("failed to start a zstd decoder: {e}")))?;
    decoder
        .read_to_end(&mut out)
        .map_err(|e| Error::malformed(format!("failed to decompress a zstd chunk: {e}")))?;
    Ok(out)
}

/// Sub-codecs that a Blosc container can wrap.
mod blosc_codec {
    /// The Blosc project's own LZ77 variant, and the default choice.
    pub const BLOSCLZ: u8 = 0;
    /// LZ4.
    pub const LZ4: u8 = 1;
    /// LZ4 high compression. Same decoder as LZ4.
    pub const LZ4HC: u8 = 2;
    /// Snappy.
    pub const SNAPPY: u8 = 3;
    /// zlib, which is deflate with a zlib wrapper.
    pub const ZLIB: u8 = 4;
    /// Zstandard.
    pub const ZSTD: u8 = 5;
}

/// Blosc header flag: the byte shuffle was applied inside the container.
const BLOSC_DOSHUFFLE: u8 = 0x01;
/// Blosc header flag: the bit shuffle was applied inside the container.
const BLOSC_DOBITSHUFFLE: u8 = 0x04;

/// Decompress a Blosc-compressed chunk.
///
/// Blosc is a container, not a codec. Its 16-byte header names a sub-codec and
/// splits the payload into independently compressed blocks. Blosc also applies
/// its own shuffle *inside* the container, separately from the HDF5 shuffle
/// filter, so that has to be undone here.
pub fn unblosc(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 16 {
        return Err(Error::malformed(
            "blosc chunk is shorter than its 16-byte header",
        ));
    }

    let version = data[0];
    if version != 2 {
        return Err(Error::unsupported(format!(
            "blosc container format version {version}"
        )));
    }
    let flags = data[2];
    let typesize = data[3] as usize;
    let nbytes = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let blocksize = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let cbytes = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;

    if cbytes > data.len() {
        return Err(Error::malformed(format!(
            "blosc header claims {cbytes} compressed bytes but the chunk holds {}",
            data.len()
        )));
    }
    if blocksize == 0 {
        return Err(Error::malformed("blosc header declares a zero block size"));
    }

    // The codec lives in the high nibble of the flags byte.
    let codec = flags >> 5;

    let block_count = nbytes.div_ceil(blocksize);
    let offsets_start = 16;
    let offsets_len = block_count * 4;
    if offsets_start + offsets_len > data.len() {
        return Err(Error::malformed(
            "blosc chunk is truncated before its block offset table",
        ));
    }

    let mut out = Vec::with_capacity(nbytes);
    for block in 0..block_count {
        let o = offsets_start + block * 4;
        let start = u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]) as usize;
        if start + 4 > data.len() {
            return Err(Error::malformed("blosc block offset points past the chunk"));
        }
        // Each block is prefixed by its own compressed length.
        let clen = u32::from_le_bytes([
            data[start],
            data[start + 1],
            data[start + 2],
            data[start + 3],
        ]) as usize;
        let body_start = start + 4;
        if body_start + clen > data.len() {
            return Err(Error::malformed("blosc block runs past the chunk"));
        }
        let body = &data[body_start..body_start + clen];
        let want = blocksize.min(nbytes - out.len());

        let decoded = match codec {
            blosc_codec::LZ4 | blosc_codec::LZ4HC => lz4_flex::block::decompress(body, want)
                .map_err(|e| Error::malformed(format!("failed to decompress an lz4 block: {e}")))?,
            blosc_codec::ZLIB => inflate(body, want)?,
            blosc_codec::ZSTD => unzstd(body, want)?,
            blosc_codec::BLOSCLZ => unblosclz(body, want)?,
            blosc_codec::SNAPPY => {
                return Err(Error::unsupported("blosc chunk using the snappy sub-codec"))
            }
            other => return Err(Error::unsupported(format!("blosc sub-codec {other}"))),
        };
        out.extend_from_slice(&decoded);
    }

    out.truncate(nbytes);

    // Blosc's own shuffle is applied inside the container and is not the same
    // thing as the HDF5 shuffle filter.
    if flags & BLOSC_DOBITSHUFFLE != 0 {
        out = unbitshuffle(&out, typesize);
    } else if flags & BLOSC_DOSHUFFLE != 0 {
        out = unshuffle(&out, typesize);
    }

    Ok(out)
}

/// Decompress a blosclz block.
///
/// blosclz is Blosc's own LZ77 variant, derived from FastLZ, and it is the
/// default codec, so skipping it would leave most real Blosc chunks unreadable.
/// The format is a stream of two kinds of token:
///
/// * A literal run: the token's top three bits are zero, and the low five bits
///   hold `count - 1` literal bytes that follow.
/// * A back-reference: the top three bits hold `length - 2`, and the remaining
///   bits, plus one or two following bytes, give the distance back into the
///   output. A length field of 7 means the real length follows in extra bytes.
///
/// Matches may overlap the current output position, so bytes are copied one at
/// a time rather than with a block move.
pub fn unblosclz(data: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::with_capacity(expected_len);
    let mut pos = 0usize;

    // The first token of a stream is always a literal run.
    let mut first = true;

    while pos < data.len() {
        let token = data[pos] as usize;
        pos += 1;

        if first || (token >> 5) == 0 {
            // Literal run of (token & 0x1f) + 1 bytes.
            let count = (token & 0x1f) + 1;
            if pos + count > data.len() {
                return Err(Error::malformed("blosclz literal run runs past the block"));
            }
            out.extend_from_slice(&data[pos..pos + count]);
            pos += count;
            first = false;
            continue;
        }

        // Back-reference.
        let mut length = token >> 5;
        if length == 7 {
            // A length of 7 means the remainder follows, 255 at a time.
            loop {
                if pos >= data.len() {
                    return Err(Error::malformed("blosclz match length runs past the block"));
                }
                let extra = data[pos] as usize;
                pos += 1;
                length += extra;
                if extra != 255 {
                    break;
                }
            }
        }
        length += 2;

        if pos >= data.len() {
            return Err(Error::malformed("blosclz match offset runs past the block"));
        }
        let mut distance = ((token & 0x1f) << 8) | data[pos] as usize;
        pos += 1;

        if distance == 0x1fff {
            // A maximal distance means two more bytes extend it.
            if pos + 1 >= data.len() {
                return Err(Error::malformed(
                    "blosclz extended match offset runs past the block",
                ));
            }
            distance += ((data[pos] as usize) << 8) | data[pos + 1] as usize;
            pos += 2;
        }

        let start = out.len().checked_sub(distance + 1).ok_or_else(|| {
            Error::malformed("blosclz match points before the start of the block")
        })?;

        // Copy byte by byte: a match may overlap what it is producing.
        for i in 0..length {
            let byte = out[start + i];
            out.push(byte);
        }
    }

    Ok(out)
}

/// Undo Blosc's bit shuffle.
///
/// Where the byte shuffle groups bytes of like significance, the bit shuffle
/// groups *bits*. Within each block of `typesize * 8` elements it transposes an
/// `8 * typesize` by `n` bit matrix. Undoing it is the transpose back.
pub fn unbitshuffle(data: &[u8], typesize: usize) -> Vec<u8> {
    if typesize == 0 || !data.len().is_multiple_of(typesize) {
        return data.to_vec();
    }
    let elements = data.len() / typesize;
    let bit_rows = typesize * 8;
    if elements == 0 {
        return data.to_vec();
    }

    let mut out = vec![0u8; data.len()];
    for row in 0..bit_rows {
        for element in 0..elements {
            // Bit `element` of row `row` in the shuffled layout.
            let src_bit = row * elements + element;
            let src_byte = src_bit / 8;
            if src_byte >= data.len() {
                continue;
            }
            let bit = (data[src_byte] >> (src_bit % 8)) & 1;
            if bit == 0 {
                continue;
            }
            // It belongs at bit `row` of element `element`.
            let dst_bit = element * bit_rows + row;
            out[dst_bit / 8] |= 1 << (dst_bit % 8);
        }
    }
    out
}

/// Apply Blosc's bit shuffle. Present so tests can round-trip.
pub fn bitshuffle(data: &[u8], typesize: usize) -> Vec<u8> {
    if typesize == 0 || !data.len().is_multiple_of(typesize) {
        return data.to_vec();
    }
    let elements = data.len() / typesize;
    let bit_rows = typesize * 8;
    let mut out = vec![0u8; data.len()];
    for element in 0..elements {
        for row in 0..bit_rows {
            let src_bit = element * bit_rows + row;
            let bit = (data[src_bit / 8] >> (src_bit % 8)) & 1;
            if bit == 0 {
                continue;
            }
            let dst_bit = row * elements + element;
            out[dst_bit / 8] |= 1 << (dst_bit % 8);
        }
    }
    out
}

/// Inflate zlib-wrapped deflate data.
pub fn inflate(data: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expected_len);
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    decoder
        .read_to_end(&mut out)
        .map_err(|e| Error::malformed(format!("failed to inflate a chunk: {e}")))?;
    Ok(out)
}

/// Undo the byte shuffle.
///
/// The shuffle filter groups bytes by significance: every first byte of an
/// element, then every second byte, and so on. Trailing bytes that do not fill
/// a whole element are left in place, so they are copied straight across.
///
/// Each element is gathered from one stream for each of its byte positions, so
/// the output is written straight through and every read runs forwards. Walking
/// the output once for each byte position instead, which is the shape this
/// filter is usually written in, strides across the whole buffer `element_size`
/// times over. On a 4 MB chunk of `float` this gather is six times faster.
pub fn unshuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.len() <= element_size {
        return data.to_vec();
    }

    let element_count = data.len() / element_size;
    let shuffled_len = element_count * element_size;
    let mut out = vec![0u8; data.len()];

    // The width has to be a constant for the gather to compile to one: a
    // runtime width leaves a bounds check on every byte and gives back the
    // whole gain. These are the widths netCDF stores.
    let target = &mut out[..shuffled_len];
    match element_size {
        2 => gather::<2>(data, target, element_count),
        4 => gather::<4>(data, target, element_count),
        8 => gather::<8>(data, target, element_count),
        16 => gather::<16>(data, target, element_count),
        _ => {
            for (element, slot) in target.chunks_exact_mut(element_size).enumerate() {
                for (byte_index, byte) in slot.iter_mut().enumerate() {
                    *byte = data[byte_index * element_count + element];
                }
            }
        }
    }

    // Any remainder was never shuffled.
    if shuffled_len < data.len() {
        out[shuffled_len..].copy_from_slice(&data[shuffled_len..]);
    }

    out
}

/// Gather the elements of a shuffled block whose width is known here.
///
/// `out` holds whole elements only, and `count` is how many. `data` holds one
/// stream for each byte position of an element, each `count` bytes long.
fn gather<const N: usize>(data: &[u8], out: &mut [u8], count: usize) {
    for (element, slot) in out.chunks_exact_mut(N).enumerate() {
        for (byte_index, byte) in slot.iter_mut().enumerate() {
            *byte = data[byte_index * count + element];
        }
    }
}

/// Apply the byte shuffle. Present so tests can round-trip.
pub fn shuffle(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.len() <= element_size {
        return data.to_vec();
    }

    let element_count = data.len() / element_size;
    let shuffled_len = element_count * element_size;
    let mut out = vec![0u8; data.len()];

    for byte_index in 0..element_size {
        let dst_base = byte_index * element_count;
        for element in 0..element_count {
            out[dst_base + element] = data[element * element_size + byte_index];
        }
    }

    if shuffled_len < data.len() {
        out[shuffled_len..].copy_from_slice(&data[shuffled_len..]);
    }

    out
}

/// Check the trailing Fletcher-32 checksum and remove it.
///
/// HDF5 writes this checksum in the writing machine's byte order, which is a
/// known wart of the format. A reader therefore accepts either orientation
/// before declaring the chunk corrupt.
pub fn verify_and_strip_fletcher32(mut data: Vec<u8>) -> Result<Vec<u8>> {
    if data.len() < 4 {
        return Err(Error::malformed(
            "chunk is too short to carry a fletcher32 checksum",
        ));
    }
    let split = data.len() - 4;
    let stored_le = u32::from_le_bytes([
        data[split],
        data[split + 1],
        data[split + 2],
        data[split + 3],
    ]);
    let stored_be = stored_le.swap_bytes();

    let computed = checksum::fletcher32(&data[..split]);
    if computed != stored_le && computed != stored_be {
        return Err(Error::ChecksumMismatch {
            what: "chunk fletcher32",
            stored: stored_le,
            computed,
        });
    }

    data.truncate(split);
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::filter::FilterPipeline;
    use std::io::Write;

    fn filter(id: u16, client: Vec<u32>) -> Filter {
        Filter {
            id,
            name: String::new(),
            flags: 0,
            client_data: client,
        }
    }

    fn deflate_bytes(data: &[u8]) -> Vec<u8> {
        let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    #[test]
    fn shuffle_round_trips() {
        // 2, 4, 8 and 16 are gathered at a width known to the compiler; 3, 5
        // and 6 take the general path. Both must agree with `shuffle`.
        let data: Vec<u8> = (0..240).map(|i| (i % 251) as u8).collect();
        for element_size in [1usize, 2, 3, 4, 5, 6, 8, 16] {
            let s = shuffle(&data, element_size);
            assert_eq!(
                unshuffle(&s, element_size),
                data,
                "element size {element_size} must round trip"
            );
        }
    }

    /// The width that divides the buffer unevenly is the one the two paths are
    /// most likely to disagree about.
    #[test]
    fn shuffle_round_trips_with_a_partial_trailing_element() {
        let data: Vec<u8> = (0..101).map(|i| (i % 251) as u8).collect();
        for element_size in [2usize, 3, 4, 8, 16] {
            let s = shuffle(&data, element_size);
            assert_eq!(
                unshuffle(&s, element_size),
                data,
                "element size {element_size} must round trip"
            );
        }
    }

    #[test]
    fn shuffle_groups_bytes_by_significance() {
        // Three 4-byte elements. Shuffling collects byte 0 of each, then byte 1.
        let data = vec![
            0x00, 0x01, 0x02, 0x03, //
            0x10, 0x11, 0x12, 0x13, //
            0x20, 0x21, 0x22, 0x23,
        ];
        let s = shuffle(&data, 4);
        assert_eq!(&s[..3], &[0x00, 0x10, 0x20]);
        assert_eq!(&s[3..6], &[0x01, 0x11, 0x21]);
        assert_eq!(unshuffle(&s, 4), data);
    }

    #[test]
    fn shuffle_leaves_a_trailing_partial_element_alone() {
        // 10 bytes with a 4-byte element leaves 2 bytes untouched at the end.
        let data: Vec<u8> = (0..10).collect();
        let s = shuffle(&data, 4);
        assert_eq!(&s[8..], &[8, 9], "the remainder is not shuffled");
        assert_eq!(unshuffle(&s, 4), data);
    }

    #[test]
    fn unshuffle_is_a_no_op_for_single_byte_elements() {
        let data: Vec<u8> = (0..16).collect();
        assert_eq!(unshuffle(&data, 1), data);
        assert_eq!(unshuffle(&data, 0), data);
    }

    #[test]
    fn inflate_recovers_deflated_bytes() {
        let original: Vec<u8> = (0..500).map(|i| (i % 7) as u8).collect();
        let compressed = deflate_bytes(&original);
        assert_eq!(inflate(&compressed, original.len()).unwrap(), original);
    }

    #[test]
    fn inflate_reports_corrupt_input() {
        let err = inflate(&[1, 2, 3, 4, 5], 16).unwrap_err();
        assert!(matches!(err, Error::Malformed(_)), "got {err:?}");
    }

    #[test]
    fn decode_chunk_undoes_shuffle_then_deflate_in_the_right_order() {
        let original: Vec<u8> = (0..128u32).flat_map(|v| v.to_le_bytes()).collect();
        // Write order: shuffle, then deflate.
        let shuffled = shuffle(&original, 4);
        let stored = deflate_bytes(&shuffled);

        let pipeline = FilterPipeline {
            filters: vec![filter(id::SHUFFLE, vec![4]), filter(id::DEFLATE, vec![6])],
        };
        let got = decode_chunk(&pipeline, 0, stored, original.len()).unwrap();
        assert_eq!(got, original);
    }

    #[test]
    fn decode_chunk_honours_the_filter_mask() {
        let original: Vec<u8> = (0..64).collect();
        // Deflate was skipped for this chunk, so only shuffle was applied.
        let stored = shuffle(&original, 4);

        let pipeline = FilterPipeline {
            filters: vec![filter(id::SHUFFLE, vec![4]), filter(id::DEFLATE, vec![6])],
        };
        // Bit 1 marks the deflate stage as skipped.
        let got = decode_chunk(&pipeline, 0b10, stored, original.len()).unwrap();
        assert_eq!(got, original);
    }

    #[test]
    fn fletcher32_round_trips_and_strips() {
        let payload: Vec<u8> = (0..40).collect();
        let sum = checksum::fletcher32(&payload);
        let mut stored = payload.clone();
        stored.extend_from_slice(&sum.to_le_bytes());
        assert_eq!(verify_and_strip_fletcher32(stored).unwrap(), payload);
    }

    #[test]
    fn fletcher32_accepts_either_byte_order() {
        let payload: Vec<u8> = (0..40).collect();
        let sum = checksum::fletcher32(&payload);
        let mut stored = payload.clone();
        stored.extend_from_slice(&sum.to_be_bytes());
        assert_eq!(
            verify_and_strip_fletcher32(stored).unwrap(),
            payload,
            "HDF5 writes this checksum in native order, so both must be accepted"
        );
    }

    #[test]
    fn fletcher32_rejects_a_corrupted_chunk() {
        let payload: Vec<u8> = (0..40).collect();
        let mut stored = payload.clone();
        stored.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let err = verify_and_strip_fletcher32(stored).unwrap_err();
        assert!(matches!(err, Error::ChecksumMismatch { .. }), "got {err:?}");
    }

    #[test]
    fn an_unknown_mandatory_filter_is_reported() {
        let pipeline = FilterPipeline {
            filters: vec![filter(id::SZIP, vec![])],
        };
        assert!(!pipeline_is_supported(&pipeline));
        let err = decode_chunk(&pipeline, 0, vec![1, 2, 3], 3).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }

    /// The optional flag must not turn an undecodable filter into a silent
    /// pass-through. It describes what a writer may skip, not what a chunk
    /// actually went through.
    #[test]
    fn an_unknown_filter_is_refused_even_when_marked_optional() {
        let mut f = filter(id::SZIP, vec![]);
        f.flags = crate::message::filter::FLAG_OPTIONAL;
        let pipeline = FilterPipeline { filters: vec![f] };
        assert!(!pipeline_is_supported(&pipeline));
        let err = decode_chunk(&pipeline, 0, vec![1, 2, 3], 3).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }

    /// A filter the chunk's mask marks as skipped is genuinely absent from the
    /// stored bytes, so it is right to do nothing for it.
    #[test]
    fn a_masked_out_unknown_filter_is_not_applied() {
        let pipeline = FilterPipeline {
            filters: vec![filter(id::SZIP, vec![])],
        };
        assert_eq!(
            decode_chunk(&pipeline, 0b1, vec![1, 2, 3], 3).unwrap(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn an_empty_pipeline_returns_the_bytes_unchanged() {
        let pipeline = FilterPipeline::default();
        assert!(pipeline_is_supported(&pipeline));
        assert_eq!(
            decode_chunk(&pipeline, 0, vec![9, 8, 7], 3).unwrap(),
            vec![9, 8, 7]
        );
    }
}
