//! The data layout message: where a dataset's raw bytes live.
//!
//! Three storage classes matter.
//!
//! * Compact keeps the values inside the object header itself.
//! * Contiguous stores them as one run of bytes.
//! * Chunked cuts the dataset into fixed-size blocks, each stored and filtered
//!   independently, and indexes them.
//!
//! Chunking is what makes parallel reads worthwhile: each chunk is an
//! independent byte range with its own filter pipeline, so many threads can
//! fetch and decompress different chunks at once.

use crate::cursor::{Cursor, Sizes};
use crate::error::{Error, Result};

/// Version 4 layout flag: a single-chunk index carries the chunk's filtered
/// size and filter mask.
pub const FLAG_SINGLE_INDEX_WITH_FILTER: u8 = 0x02;

/// How a chunked dataset's chunks are indexed. Version 4 layouts choose one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkIndex {
    /// One chunk covers the whole dataset. The address points straight at it.
    SingleChunk {
        /// Stored size of the one chunk in bytes.
        filtered_size: u64,
        /// Filter mask for the one chunk.
        filter_mask: u32,
    },
    /// No index needed: the dataset is unfiltered and fixed-size, so a chunk's
    /// address is computed arithmetically.
    Implicit,
    /// A fixed array, for a dataset whose maximum dimensions are all fixed.
    FixedArray {
        /// Bits per page of the fixed array.
        page_bits: u8,
    },
    /// An extensible array, for a dataset with exactly one unlimited dimension.
    ExtensibleArray {
        /// Maximum bits of the index.
        max_bits: u8,
        /// Number of elements in the index block.
        index_elements: u8,
        /// Minimum number of pointers per data block.
        min_pointers: u8,
        /// Minimum number of elements per data block.
        min_elements: u8,
        /// Bits per page.
        page_bits: u8,
    },
    /// A version 2 B-tree, for a dataset with several unlimited dimensions.
    BtreeV2 {
        /// Maximum number of records in a node.
        node_size: u32,
        /// Split percentage.
        split_percent: u8,
        /// Merge percentage.
        merge_percent: u8,
    },
    /// The version 1 B-tree used by layout versions 1 through 3.
    BtreeV1,
}

/// Where and how a dataset's bytes are stored.
#[derive(Debug, Clone, PartialEq)]
pub enum Layout {
    /// Values live in the object header message itself.
    Compact {
        /// The raw bytes.
        data: Vec<u8>,
    },
    /// Values live in one contiguous run.
    Contiguous {
        /// Start address, or `None` when nothing has been written yet.
        address: Option<u64>,
        /// Length of the run in bytes.
        size: u64,
    },
    /// Values are cut into chunks.
    Chunked {
        /// Address of the chunk index, or `None` when nothing is written.
        address: Option<u64>,
        /// Shape of one chunk, in elements, matching the dataset rank.
        chunk_dims: Vec<u32>,
        /// Size of one element in bytes.
        element_size: u32,
        /// How chunks are indexed.
        index: ChunkIndex,
    },
}

impl Layout {
    /// Parse a data layout message body.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Self> {
        let mut cur = Cursor::new(data);
        let version = cur.u8()?;
        match version {
            1 | 2 => Self::parse_v1_v2(&mut cur, sizes),
            3 => Self::parse_v3(&mut cur, sizes),
            4 => Self::parse_v4(&mut cur, sizes),
            other => Err(Error::unsupported(format!(
                "data layout message version {other}"
            ))),
        }
    }

    fn parse_v1_v2(cur: &mut Cursor<'_>, sizes: Sizes) -> Result<Self> {
        let dimensionality = cur.u8()? as usize;
        let class = cur.u8()?;
        cur.skip(5)?; // reserved

        match class {
            0 => {
                // Compact: the dimension list comes first, then the size and
                // the values.
                for _ in 0..dimensionality {
                    cur.u32()?;
                }
                let size = cur.u32()? as usize;
                Ok(Layout::Compact {
                    data: cur.take(size)?.to_vec(),
                })
            }
            1 => {
                let address = cur.address(sizes)?;
                let mut total: u64 = 1;
                for _ in 0..dimensionality {
                    total = total.saturating_mul(cur.u32()? as u64);
                }
                Ok(Layout::Contiguous {
                    address,
                    // Versions 1 and 2 record the shape rather than a byte
                    // count, so the size is the product. The element size is
                    // not present, so the caller multiplies it in.
                    size: total,
                })
            }
            2 => {
                let address = cur.address(sizes)?;
                // The stored dimensionality counts the trailing element-size
                // entry, so the chunk rank is one less.
                if dimensionality == 0 {
                    return Err(Error::malformed(
                        "chunked layout declares a dimensionality of zero",
                    ));
                }
                let rank = dimensionality - 1;
                let mut chunk_dims = Vec::with_capacity(rank);
                for _ in 0..rank {
                    chunk_dims.push(cur.u32()?);
                }
                let element_size = cur.u32()?;
                Ok(Layout::Chunked {
                    address,
                    chunk_dims,
                    element_size,
                    index: ChunkIndex::BtreeV1,
                })
            }
            other => Err(Error::malformed(format!("data layout class {other}"))),
        }
    }

    fn parse_v3(cur: &mut Cursor<'_>, sizes: Sizes) -> Result<Self> {
        let class = cur.u8()?;
        match class {
            0 => {
                let size = cur.u16()? as usize;
                Ok(Layout::Compact {
                    data: cur.take(size)?.to_vec(),
                })
            }
            1 => {
                let address = cur.address(sizes)?;
                let size = cur.length(sizes)?;
                Ok(Layout::Contiguous { address, size })
            }
            2 => {
                let dimensionality = cur.u8()? as usize;
                let address = cur.address(sizes)?;
                if dimensionality == 0 {
                    return Err(Error::malformed(
                        "chunked layout declares a dimensionality of zero",
                    ));
                }
                // The last entry is the element size, not a chunk dimension.
                let rank = dimensionality - 1;
                let mut chunk_dims = Vec::with_capacity(rank);
                for _ in 0..rank {
                    chunk_dims.push(cur.u32()?);
                }
                let element_size = cur.u32()?;
                Ok(Layout::Chunked {
                    address,
                    chunk_dims,
                    element_size,
                    index: ChunkIndex::BtreeV1,
                })
            }
            other => Err(Error::malformed(format!("data layout class {other}"))),
        }
    }

    fn parse_v4(cur: &mut Cursor<'_>, sizes: Sizes) -> Result<Self> {
        let class = cur.u8()?;
        match class {
            0 => {
                let size = cur.u16()? as usize;
                Ok(Layout::Compact {
                    data: cur.take(size)?.to_vec(),
                })
            }
            1 => {
                let address = cur.address(sizes)?;
                let size = cur.length(sizes)?;
                Ok(Layout::Contiguous { address, size })
            }
            2 => {
                let flags = cur.u8()?;
                let dimensionality = cur.u8()? as usize;
                if dimensionality == 0 {
                    return Err(Error::malformed(
                        "chunked layout declares a dimensionality of zero",
                    ));
                }
                let dim_width = cur.u8()?;
                // Like version 3, the trailing entry is the element size rather
                // than a chunk dimension.
                let rank = dimensionality - 1;
                let mut chunk_dims = Vec::with_capacity(rank);
                for _ in 0..rank {
                    chunk_dims.push(cur.uint(dim_width)? as u32);
                }
                let element_size = cur.uint(dim_width)? as u32;
                let index_type = cur.u8()?;

                let index = match index_type {
                    1 => {
                        // The size and mask are only stored when the dataset is
                        // filtered, which this flag records.
                        if flags & FLAG_SINGLE_INDEX_WITH_FILTER != 0 {
                            ChunkIndex::SingleChunk {
                                filtered_size: cur.length(sizes)?,
                                filter_mask: cur.u32()?,
                            }
                        } else {
                            ChunkIndex::SingleChunk {
                                filtered_size: 0,
                                filter_mask: 0,
                            }
                        }
                    }
                    2 => ChunkIndex::Implicit,
                    3 => ChunkIndex::FixedArray {
                        page_bits: cur.u8()?,
                    },
                    4 => ChunkIndex::ExtensibleArray {
                        max_bits: cur.u8()?,
                        index_elements: cur.u8()?,
                        min_pointers: cur.u8()?,
                        min_elements: cur.u8()?,
                        page_bits: cur.u8()?,
                    },
                    5 => ChunkIndex::BtreeV2 {
                        node_size: cur.u32()?,
                        split_percent: cur.u8()?,
                        merge_percent: cur.u8()?,
                    },
                    other => {
                        return Err(Error::unsupported(format!(
                            "version 4 chunk index type {other}"
                        )))
                    }
                };

                let address = cur.address(sizes)?;

                Ok(Layout::Chunked {
                    address,
                    chunk_dims,
                    element_size,
                    index,
                })
            }
            other => Err(Error::malformed(format!("data layout class {other}"))),
        }
    }

    /// Shape of one chunk, when the dataset is chunked.
    pub fn chunk_dims(&self) -> Option<&[u32]> {
        match self {
            Layout::Chunked { chunk_dims, .. } => Some(chunk_dims),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_version_three_contiguous_layout() {
        let mut d = vec![3u8, 1];
        d.extend_from_slice(&0x1234u64.to_le_bytes());
        d.extend_from_slice(&960u64.to_le_bytes());
        let l = Layout::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(
            l,
            Layout::Contiguous {
                address: Some(0x1234),
                size: 960
            }
        );
    }

    #[test]
    fn an_unwritten_contiguous_dataset_has_no_address() {
        let mut d = vec![3u8, 1];
        d.extend_from_slice(&u64::MAX.to_le_bytes());
        d.extend_from_slice(&0u64.to_le_bytes());
        let l = Layout::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(
            l,
            Layout::Contiguous {
                address: None,
                size: 0
            }
        );
    }

    #[test]
    fn parses_a_version_three_chunked_layout() {
        // Dimensionality 3 means a rank-2 chunk plus the element size.
        let mut d = vec![3u8, 2, 3];
        d.extend_from_slice(&0xABCDu64.to_le_bytes());
        d.extend_from_slice(&7u32.to_le_bytes());
        d.extend_from_slice(&4u32.to_le_bytes());
        d.extend_from_slice(&4u32.to_le_bytes()); // element size
        let l = Layout::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(
            l,
            Layout::Chunked {
                address: Some(0xABCD),
                chunk_dims: vec![7, 4],
                element_size: 4,
                index: ChunkIndex::BtreeV1
            }
        );
        assert_eq!(l.chunk_dims(), Some(&[7u32, 4][..]));
    }

    #[test]
    fn parses_a_version_three_compact_layout() {
        let mut d = vec![3u8, 0];
        d.extend_from_slice(&4u16.to_le_bytes());
        d.extend_from_slice(&[1, 2, 3, 4]);
        let l = Layout::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(
            l,
            Layout::Compact {
                data: vec![1, 2, 3, 4]
            }
        );
    }

    #[test]
    fn parses_a_version_one_chunked_layout() {
        let mut d = vec![1u8, 3, 2, 0, 0, 0, 0, 0];
        d.extend_from_slice(&0x99u64.to_le_bytes());
        d.extend_from_slice(&10u32.to_le_bytes());
        d.extend_from_slice(&20u32.to_le_bytes());
        d.extend_from_slice(&8u32.to_le_bytes()); // element size
        let l = Layout::parse(&d, Sizes::EIGHT).unwrap();
        match l {
            Layout::Chunked {
                chunk_dims,
                element_size,
                ..
            } => {
                assert_eq!(chunk_dims, vec![10, 20]);
                assert_eq!(element_size, 8);
            }
            other => panic!("expected a chunked layout, got {other:?}"),
        }
    }

    #[test]
    fn parses_a_version_four_single_chunk_index() {
        let mut d = vec![4u8, 2, FLAG_SINGLE_INDEX_WITH_FILTER, 3, 8];
        d.extend_from_slice(&4u64.to_le_bytes());
        d.extend_from_slice(&5u64.to_le_bytes());
        d.extend_from_slice(&4u64.to_le_bytes()); // element size
        d.push(1); // index type: single chunk
        d.extend_from_slice(&123u64.to_le_bytes()); // filtered size
        d.extend_from_slice(&0u32.to_le_bytes()); // filter mask
        d.extend_from_slice(&0x777u64.to_le_bytes()); // address
        let l = Layout::parse(&d, Sizes::EIGHT).unwrap();
        match l {
            Layout::Chunked {
                address,
                chunk_dims,
                index,
                ..
            } => {
                assert_eq!(address, Some(0x777));
                assert_eq!(chunk_dims, vec![4, 5]);
                assert_eq!(
                    index,
                    ChunkIndex::SingleChunk {
                        filtered_size: 123,
                        filter_mask: 0
                    }
                );
            }
            other => panic!("expected a chunked layout, got {other:?}"),
        }
    }

    #[test]
    fn rejects_a_chunked_layout_with_zero_dimensionality() {
        let d = vec![3u8, 2, 0];
        assert!(Layout::parse(&d, Sizes::EIGHT).is_err());
    }

    #[test]
    fn rejects_an_unknown_layout_version() {
        let err = Layout::parse(&[9u8, 1], Sizes::EIGHT).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }
}
