//! Version 4 chunk indexes.
//!
//! A version 3 layout always indexes chunks with a version 1 B-tree. Version 4
//! picks an index to suit the dataset's shape, which is cheaper but means a
//! reader has to implement five of them:
//!
//! | Index | Chosen when |
//! |---|---|
//! | Single chunk | one chunk covers the whole dataset |
//! | Implicit | unfiltered with fixed maximum dimensions, so addresses are arithmetic |
//! | Fixed array | filtered with fixed maximum dimensions |
//! | Extensible array | exactly one unlimited dimension |
//! | Version 2 B-tree | more than one unlimited dimension |
//!
//! All of them produce the same thing this reader wants: a list of chunks, each
//! with a position, an address and a stored size.

use crate::btree1::ChunkRecord;
use crate::btree2::BtreeV2;
use crate::checksum;
use crate::context::Ctx;
use crate::cursor::Cursor;
use crate::error::{Error, Result};
use crate::message::ChunkIndex;

/// Signature of a fixed array header.
pub const FAHD_SIGNATURE: &[u8; 4] = b"FAHD";
/// Signature of a fixed array data block.
pub const FADB_SIGNATURE: &[u8; 4] = b"FADB";
/// Signature of an extensible array header.
pub const EAHD_SIGNATURE: &[u8; 4] = b"EAHD";
/// Signature of an extensible array index block.
pub const EAIB_SIGNATURE: &[u8; 4] = b"EAIB";
/// Signature of an extensible array data block.
pub const EADB_SIGNATURE: &[u8; 4] = b"EADB";

/// Super blocks whose data blocks the index block points at directly.
///
/// Super blocks 0 and 1 hold one data block each, so the index block carries
/// two direct data block addresses before the secondary block addresses begin.
const SBLK_FIRST_IDX: usize = 2;

/// Client ID marking an index over filtered chunks.
const CLIENT_FILTERED_CHUNKS: u8 = 1;

/// Resolve a version 4 chunk index into chunk records.
///
/// `shape` and `chunk_dims` describe the dataset, and `element_size` the width
/// of one value; together they give each chunk's position and unfiltered size.
pub fn read(
    ctx: Ctx<'_>,
    index: &ChunkIndex,
    address: Option<u64>,
    shape: &[u64],
    chunk_dims: &[u64],
    element_size: usize,
    filtered: bool,
) -> Result<Vec<ChunkRecord>> {
    let grid = chunk_grid(shape, chunk_dims);
    let total: u64 = grid.iter().product();
    let unfiltered_size = chunk_dims.iter().product::<u64>() * element_size as u64;

    let Some(address) = address else {
        return Ok(Vec::new());
    };

    let mut records = match index {
        ChunkIndex::SingleChunk {
            filtered_size,
            filter_mask,
        } => vec![ChunkRecord {
            size: if filtered {
                *filtered_size as u32
            } else {
                unfiltered_size as u32
            },
            filter_mask: *filter_mask,
            offset: vec![0; shape.len()],
            address,
        }],

        // Chunks sit back to back in row-major chunk order, so each address is
        // arithmetic. An implicit index only exists for unfiltered data.
        ChunkIndex::Implicit => (0..total)
            .map(|i| ChunkRecord {
                size: unfiltered_size as u32,
                filter_mask: 0,
                offset: chunk_offset(i, &grid, chunk_dims),
                address: address + i * unfiltered_size,
            })
            .collect(),

        ChunkIndex::FixedArray { .. } => {
            let elements = read_fixed_array(ctx, address, total as usize)?;
            elements_to_records(elements, &grid, chunk_dims, unfiltered_size)
        }

        ChunkIndex::ExtensibleArray { .. } => {
            let elements = read_extensible_array(ctx, address)?;
            elements_to_records(elements, &grid, chunk_dims, unfiltered_size)
        }

        ChunkIndex::BtreeV2 { .. } => {
            let mut records = read_btree2_chunks(ctx, address, chunk_dims, filtered)?;
            // An unfiltered record stores no size; it is the full chunk.
            for r in records.iter_mut() {
                if r.size == 0 {
                    r.size = unfiltered_size as u32;
                }
            }
            records
        }

        ChunkIndex::BtreeV1 => {
            return Err(Error::malformed(
                "a version 1 B-tree index reached the version 4 resolver",
            ))
        }
    };

    records.sort_by(|a, b| a.offset.cmp(&b.offset));
    Ok(records)
}

/// One entry of an array-style index.
struct Element {
    address: Option<u64>,
    size: u32,
    filter_mask: u32,
}

fn elements_to_records(
    elements: Vec<Element>,
    grid: &[u64],
    chunk_dims: &[u64],
    unfiltered_size: u64,
) -> Vec<ChunkRecord> {
    elements
        .into_iter()
        .enumerate()
        .filter_map(|(i, e)| {
            let address = e.address?;
            Some(ChunkRecord {
                size: if e.size == 0 {
                    unfiltered_size as u32
                } else {
                    e.size
                },
                filter_mask: e.filter_mask,
                offset: chunk_offset(i as u64, grid, chunk_dims),
                address,
            })
        })
        .collect()
}

/// Number of chunks along each axis.
fn chunk_grid(shape: &[u64], chunk_dims: &[u64]) -> Vec<u64> {
    shape
        .iter()
        .zip(chunk_dims.iter())
        .map(|(&d, &c)| if c == 0 { 0 } else { d.div_ceil(c) })
        .collect()
}

/// Element offset of chunk number `index`, in row-major chunk order.
fn chunk_offset(index: u64, grid: &[u64], chunk_dims: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; grid.len()];
    let mut rest = index;
    for axis in (0..grid.len()).rev() {
        let extent = grid[axis].max(1);
        out[axis] = (rest % extent) * chunk_dims[axis];
        rest /= extent;
    }
    out
}

/// Read a fixed array index.
fn read_fixed_array(ctx: Ctx<'_>, address: u64, expected: usize) -> Result<Vec<Element>> {
    let sizes = ctx.sizes();
    let header_len = 4 + 1 + 1 + 1 + 1 + sizes.length as usize + sizes.offset as usize + 4;
    let buf = ctx.read(address, header_len)?;
    let mut cur = Cursor::new(&buf);

    expect_signature(&mut cur, FAHD_SIGNATURE, "fixed array header")?;
    let version = cur.u8()?;
    if version != 0 {
        return Err(Error::unsupported(format!(
            "fixed array header version {version}"
        )));
    }
    let client_id = cur.u8()?;
    let entry_size = cur.u8()? as usize;
    let page_bits = cur.u8()?;
    let max_entries = cur.length(sizes)? as usize;
    let data_block_address = cur.address(sizes)?;
    verify_checksum(&buf, header_len - 4, "fixed array header")?;

    let Some(data_block_address) = data_block_address else {
        return Ok(Vec::new());
    };

    let count = max_entries.min(expected.max(max_entries));
    // A paged data block interleaves a bitmap and per-page checksums.
    if page_bits > 0 && max_entries > (1usize << page_bits) {
        return Err(Error::unsupported("paged fixed array data block"));
    }

    let block_len = 4 + 1 + 1 + sizes.offset as usize + count * entry_size + 4;
    let block = ctx.read(data_block_address, block_len)?;
    let mut cur = Cursor::new(&block);

    expect_signature(&mut cur, FADB_SIGNATURE, "fixed array data block")?;
    let version = cur.u8()?;
    if version != 0 {
        return Err(Error::unsupported(format!(
            "fixed array data block version {version}"
        )));
    }
    cur.skip(1)?; // client id, already known
    cur.skip(sizes.offset as usize)?; // header address
    verify_checksum(&block, block_len - 4, "fixed array data block")?;

    read_elements(&mut cur, count, entry_size, client_id, sizes)
}

/// Read an extensible array index.
///
/// The elements are spread across three places, in index order:
///
/// 1. the first `idx_blk_elmts` sit inside the index block;
/// 2. the next few live in data blocks the index block points at directly;
/// 3. anything beyond that hangs off secondary blocks.
///
/// Super block `u` holds `2^(u/2)` data blocks of `2^((u+1)/2) * min` elements
/// each, so the run each data block covers grows as the array does.
fn read_extensible_array(ctx: Ctx<'_>, address: u64) -> Result<Vec<Element>> {
    let sizes = ctx.sizes();
    let l = sizes.length as usize;
    let o = sizes.offset as usize;
    let header_len = 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + l * 6 + o + 4;
    let buf = ctx.read(address, header_len)?;
    let mut cur = Cursor::new(&buf);

    expect_signature(&mut cur, EAHD_SIGNATURE, "extensible array header")?;
    let version = cur.u8()?;
    if version != 0 {
        return Err(Error::unsupported(format!(
            "extensible array header version {version}"
        )));
    }
    let client_id = cur.u8()?;
    let element_size = cur.u8()? as usize;
    let max_nelmts_bits = cur.u8()? as u32;
    let index_block_elements = cur.u8()? as usize;
    let data_block_min_elements = cur.u8()? as usize;
    let _secondary_block_min_pointers = cur.u8()?;
    let _max_data_block_page_bits = cur.u8()?;
    let _num_secondary_blocks = cur.length(sizes)?;
    let _secondary_block_size = cur.length(sizes)?;
    let _num_data_blocks = cur.length(sizes)?;
    let _data_block_size = cur.length(sizes)?;
    let max_index_set = cur.length(sizes)? as usize;
    let _num_elements = cur.length(sizes)?;
    let index_block_address = cur.address(sizes)?;
    verify_checksum(&buf, header_len - 4, "extensible array header")?;

    let Some(index_block_address) = index_block_address else {
        return Ok(Vec::new());
    };
    if element_size == 0 || data_block_min_elements == 0 {
        return Err(Error::malformed(
            "extensible array header declares a zero element or block size",
        ));
    }

    // Super block layout, which fixes how many elements each data block holds.
    let nsblks = 1 + (max_nelmts_bits as usize - log2_exact(data_block_min_elements));
    let total_addresses = SBLK_FIRST_IDX + nsblks;
    let block_offset_size = max_nelmts_bits.div_ceil(8) as usize;

    let index_len = 4 + 1 + 1 + o + index_block_elements * element_size + total_addresses * o + 4;
    let block = ctx.read(index_block_address, index_len)?;
    let mut cur = Cursor::new(&block);

    expect_signature(&mut cur, EAIB_SIGNATURE, "extensible array index block")?;
    let version = cur.u8()?;
    if version != 0 {
        return Err(Error::unsupported(format!(
            "extensible array index block version {version}"
        )));
    }
    cur.skip(1)?; // client id
    cur.skip(o)?; // header address
    verify_checksum(&block, index_len - 4, "extensible array index block")?;

    // 1. Elements stored inline.
    let inline = index_block_elements.min(max_index_set);
    let mut out = read_elements(&mut cur, inline, element_size, client_id, sizes)?;
    cur.skip((index_block_elements - inline) * element_size)?;

    if out.len() >= max_index_set {
        return Ok(out);
    }

    // 2. Data blocks the index block points at directly.
    let mut data_block_addresses = Vec::with_capacity(SBLK_FIRST_IDX);
    for _ in 0..SBLK_FIRST_IDX {
        data_block_addresses.push(cur.address(sizes)?);
    }

    for (sblk, data_block_address) in data_block_addresses.into_iter().enumerate() {
        if out.len() >= max_index_set {
            break;
        }
        // Super blocks 0 and 1 each hold exactly one data block.
        let capacity = data_block_capacity(sblk, data_block_min_elements);
        let Some(data_block_address) = data_block_address else {
            // An unallocated data block leaves that whole run unwritten.
            for _ in 0..capacity.min(max_index_set - out.len()) {
                out.push(Element {
                    address: None,
                    size: 0,
                    filter_mask: 0,
                });
            }
            continue;
        };

        let wanted = capacity.min(max_index_set - out.len());
        out.extend(read_data_block(
            ctx,
            data_block_address,
            wanted,
            capacity,
            element_size,
            client_id,
            block_offset_size,
        )?);
    }

    if out.len() < max_index_set {
        // 3. Secondary blocks. Only reached by arrays larger than the first two
        // super blocks cover, and not implemented; say so rather than return a
        // short index, which would read as missing chunks.
        return Err(Error::unsupported(format!(
            "extensible array with {} elements beyond its directly addressed data blocks; \
             secondary blocks are not implemented",
            max_index_set - out.len()
        )));
    }

    Ok(out)
}

/// Elements held by the single data block of super block `sblk`.
fn data_block_capacity(sblk: usize, min_elements: usize) -> usize {
    // Super block u holds data blocks of 2^((u+1)/2) * min elements.
    min_elements << sblk.div_ceil(2)
}

/// Read `wanted` elements out of one extensible array data block.
#[allow(clippy::too_many_arguments)]
fn read_data_block(
    ctx: Ctx<'_>,
    address: u64,
    wanted: usize,
    capacity: usize,
    element_size: usize,
    client_id: u8,
    block_offset_size: usize,
) -> Result<Vec<Element>> {
    let sizes = ctx.sizes();
    let len = 4 + 1 + 1 + sizes.offset as usize + block_offset_size + capacity * element_size + 4;
    let block = ctx.read(address, len)?;
    let mut cur = Cursor::new(&block);

    expect_signature(&mut cur, EADB_SIGNATURE, "extensible array data block")?;
    let version = cur.u8()?;
    if version != 0 {
        return Err(Error::unsupported(format!(
            "extensible array data block version {version}"
        )));
    }
    cur.skip(1)?; // client id
    cur.skip(sizes.offset as usize)?; // header address
    cur.skip(block_offset_size)?;
    verify_checksum(&block, len - 4, "extensible array data block")?;

    read_elements(&mut cur, wanted, element_size, client_id, sizes)
}

/// Exact base-2 logarithm of a power of two.
fn log2_exact(value: usize) -> usize {
    (usize::BITS - 1 - value.leading_zeros()) as usize
}

/// Read `count` index entries, whose shape depends on whether chunks are
/// filtered.
fn read_elements(
    cur: &mut Cursor<'_>,
    count: usize,
    entry_size: usize,
    client_id: u8,
    sizes: crate::cursor::Sizes,
) -> Result<Vec<Element>> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        if client_id == CLIENT_FILTERED_CHUNKS {
            // Address, then the stored chunk size, then the filter mask.
            let address = cur.address(sizes)?;
            let size_width = entry_size
                .checked_sub(sizes.offset as usize + 4)
                .ok_or_else(|| Error::malformed("chunk index entry is too small"))?;
            let size = cur.uint(size_width as u8)? as u32;
            let filter_mask = cur.u32()?;
            out.push(Element {
                address,
                size,
                filter_mask,
            });
        } else {
            out.push(Element {
                address: cur.address(sizes)?,
                size: 0,
                filter_mask: 0,
            });
        }
    }
    Ok(out)
}

/// Read a version 2 B-tree chunk index.
///
/// Record type 10 holds unfiltered chunks and type 11 filtered ones. Both end
/// with one scaled offset per dimension.
fn read_btree2_chunks(
    ctx: Ctx<'_>,
    address: u64,
    chunk_dims: &[u64],
    filtered: bool,
) -> Result<Vec<ChunkRecord>> {
    let sizes = ctx.sizes();
    let btree = BtreeV2::read(ctx, address)?;
    let offset_width = sizes.offset as usize;

    // The record carries one 8-byte offset per stored dimension, and the layout
    // counts a trailing element-size axis, so the number of offsets is derived
    // from the record size rather than assumed.
    let fixed = if filtered {
        // Address, plus a size field, plus a 4-byte filter mask.
        offset_width + 4
    } else {
        offset_width
    };
    let offsets_bytes = btree.record_size.saturating_sub(fixed);
    let stored_dims = offsets_bytes / 8;
    if stored_dims < chunk_dims.len() {
        return Err(Error::malformed(format!(
            "chunk B-tree record holds {stored_dims} offsets for a rank-{} dataset",
            chunk_dims.len()
        )));
    }
    // A filtered record's size field takes whatever is left over.
    let size_width = btree
        .record_size
        .saturating_sub(offset_width + 4 + stored_dims * 8);

    let mut out = Vec::with_capacity(btree.records.len());
    for record in &btree.records {
        let mut cur = Cursor::new(record);
        let chunk_address = cur.address_required(sizes, "chunk B-tree record")?;

        let (size, filter_mask) = if filtered {
            let size = cur.uint(size_width as u8)? as u32;
            let mask = cur.u32()?;
            (size, mask)
        } else {
            (0, 0)
        };

        // Offsets are *scaled*: they count chunks, not elements.
        let mut offset = Vec::with_capacity(chunk_dims.len());
        for axis in 0..stored_dims {
            let scaled = cur.u64()?;
            if axis < chunk_dims.len() {
                offset.push(scaled * chunk_dims[axis]);
            }
        }

        out.push(ChunkRecord {
            size,
            filter_mask,
            offset,
            address: chunk_address,
        });
    }
    Ok(out)
}

fn expect_signature(cur: &mut Cursor<'_>, want: &[u8; 4], what: &str) -> Result<()> {
    let sig = cur.take(4)?;
    if sig != want {
        return Err(Error::malformed(format!("{what} has the wrong signature")));
    }
    Ok(())
}

fn verify_checksum(buf: &[u8], len: usize, what: &'static str) -> Result<()> {
    let stored = u32::from_le_bytes([buf[len], buf[len + 1], buf[len + 2], buf[len + 3]]);
    let computed = checksum::metadata(&buf[..len]);
    if stored != computed {
        return Err(Error::ChecksumMismatch {
            what,
            stored,
            computed,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_block_capacity_grows_with_the_super_block() {
        // Super blocks 0 and 1 hold one data block each; the run doubles every
        // other super block.
        assert_eq!(data_block_capacity(0, 16), 16);
        assert_eq!(data_block_capacity(1, 16), 32);
    }

    #[test]
    fn log2_of_a_power_of_two_is_exact() {
        assert_eq!(log2_exact(16), 4);
        assert_eq!(log2_exact(1), 0);
        assert_eq!(log2_exact(1024), 10);
    }

    #[test]
    fn the_chunk_grid_rounds_up() {
        assert_eq!(chunk_grid(&[40, 6], &[7, 4]), vec![6, 2]);
        assert_eq!(chunk_grid(&[40, 6], &[40, 6]), vec![1, 1]);
    }

    #[test]
    fn chunk_offsets_walk_in_row_major_order() {
        let grid = [6u64, 2];
        let dims = [7u64, 4];
        assert_eq!(chunk_offset(0, &grid, &dims), vec![0, 0]);
        assert_eq!(chunk_offset(1, &grid, &dims), vec![0, 4]);
        assert_eq!(chunk_offset(2, &grid, &dims), vec![7, 0]);
        assert_eq!(chunk_offset(11, &grid, &dims), vec![35, 4]);
    }
}
