//! Fractal heaps.
//!
//! A fractal heap stores many small records: the links of a large group, or the
//! attributes of an object with many of them. Records live in *direct* blocks;
//! *indirect* blocks form a tree over those, with block sizes doubling row by
//! row after the first two.
//!
//! # Why this walks the heap instead of the B-tree
//!
//! Dense storage pairs the heap with a version 2 B-tree that maps a name to a
//! heap ID. That index exists to answer "find the record called X". This reader
//! wants every record, so it walks the heap directly and skips the B-tree
//! entirely, which is both simpler and fewer reads.
//!
//! The cost is that records must be self-delimiting, and a heap with internal
//! free space would desynchronise the walk. The heap header records how many
//! managed objects it holds, so the walk is checked against that count and
//! fails loudly if it disagrees. Under-reporting a group's children silently
//! would be far worse than an error.

use crate::checksum;
use crate::cursor::Cursor;
use crate::error::{Error, Result};
use crate::hdf5::context::Ctx;

/// Signature of a fractal heap header.
pub const FRHP_SIGNATURE: &[u8; 4] = b"FRHP";
/// Signature of a fractal heap direct block.
pub const FHDB_SIGNATURE: &[u8; 4] = b"FHDB";
/// Signature of a fractal heap indirect block.
pub const FHIB_SIGNATURE: &[u8; 4] = b"FHIB";

/// Heap flag: direct blocks carry a checksum.
const FLAG_CHECKSUM_DIRECT_BLOCKS: u8 = 0x02;

/// Guard against a cyclic or absurd block tree.
const MAX_BLOCKS: usize = 1 << 20;

/// A fractal heap, with every direct block's payload already loaded.
#[derive(Debug, Clone)]
pub struct FractalHeap {
    /// Payload of each direct block, paired with the heap offset it starts at.
    blocks: Vec<(u64, Vec<u8>)>,
    /// How many managed objects the header says the heap holds.
    managed_object_count: u64,
    /// Width of a heap ID, from the header.
    heap_id_len: u16,
    /// Width of the offset field inside a heap ID.
    offset_size: usize,
}

impl FractalHeap {
    /// Read the heap whose header is at `address`.
    pub fn read(ctx: Ctx<'_>, address: u64) -> Result<Self> {
        let header = Header::read(ctx, address)?;

        if header.io_filter_encoded_len > 0 {
            return Err(Error::unsupported(
                "fractal heap with filtered direct blocks",
            ));
        }

        let mut blocks = Vec::new();
        if let Some(root) = header.root_block_address {
            if header.current_root_rows == 0 {
                // The root block is a direct block of the starting size.
                read_direct_block(ctx, &header, root, header.starting_block_size, &mut blocks)?;
            } else {
                read_indirect_block(ctx, &header, root, header.current_root_rows, &mut blocks, 0)?;
            }
        }

        blocks.sort_by_key(|(offset, _)| *offset);
        Ok(Self {
            blocks,
            managed_object_count: header.managed_object_count,
            heap_id_len: header.heap_id_len,
            offset_size: header.block_offset_size,
        })
    }

    /// The payload of each direct block, in heap order.
    pub fn blocks(&self) -> Vec<&[u8]> {
        self.blocks.iter().map(|(_, b)| b.as_slice()).collect()
    }

    /// Resolve a managed-object heap ID to its bytes.
    ///
    /// A heap ID is a flags byte, an offset into the heap's managed space, and
    /// a length. Only managed objects are resolved here; huge and tiny objects
    /// live elsewhere and are reported rather than guessed at.
    pub fn get(&self, heap_id: &[u8]) -> Result<&[u8]> {
        if heap_id.len() < self.heap_id_len as usize {
            return Err(Error::malformed(format!(
                "heap ID is {} bytes but this heap uses {}",
                heap_id.len(),
                self.heap_id_len
            )));
        }
        // Bits 4 and 5 of the first byte give the object's storage class.
        let kind = (heap_id[0] >> 4) & 0x03;
        if kind != 0 {
            return Err(Error::unsupported(format!(
                "fractal heap object of class {kind} (only managed objects are read)"
            )));
        }

        let length_size = self.heap_id_len as usize - 1 - self.offset_size;
        let mut cur = Cursor::at(heap_id, 1);
        let offset = cur.uint(self.offset_size as u8)?;
        let length = cur.uint(length_size as u8)? as usize;

        for (block_offset, payload) in &self.blocks {
            let end = block_offset + payload.len() as u64;
            if offset >= *block_offset && offset < end {
                let start = (offset - block_offset) as usize;
                if start + length > payload.len() {
                    return Err(Error::malformed(
                        "fractal heap object runs past the end of its direct block",
                    ));
                }
                return Ok(&payload[start..start + length]);
            }
        }

        Err(Error::malformed(format!(
            "fractal heap offset {offset} falls in no direct block"
        )))
    }

    /// Parse every record named by a version 2 B-tree index.
    ///
    /// This is the reliable path for a heap with internal free space, which a
    /// sequential walk cannot enumerate. `heap_id_of` extracts the heap ID from
    /// a B-tree record, whose layout depends on the record type.
    pub fn walk_indexed<T, F, G>(
        &self,
        btree: &crate::hdf5::btree2::BtreeV2,
        mut heap_id_of: G,
        mut parse: F,
    ) -> Result<Vec<T>>
    where
        F: FnMut(&mut Cursor<'_>) -> Result<T>,
        G: FnMut(&[u8]) -> &[u8],
    {
        let mut out = Vec::with_capacity(btree.records.len());
        for record in &btree.records {
            let bytes = self.get(heap_id_of(record))?;
            let mut cur = Cursor::new(bytes);
            out.push(parse(&mut cur)?);
        }
        Ok(out)
    }

    /// How many managed objects the heap header claims.
    pub fn managed_object_count(&self) -> u64 {
        self.managed_object_count
    }

    /// Walk every record in the heap.
    ///
    /// `parse` is called at each record boundary and must consume exactly one
    /// record from the cursor. The walk stops at a record that fails to parse or
    /// that begins with a zero byte, which marks unused space.
    ///
    /// The number of records found is checked against the header's count, so a
    /// heap this reader cannot walk correctly produces an error rather than a
    /// short list.
    pub fn walk<T, F>(&self, what: &str, parse: F) -> Result<Vec<T>>
    where
        F: FnMut(&mut Cursor<'_>) -> Result<T>,
    {
        let (out, complete) = self.walk_lenient(what, parse)?;
        if !complete {
            return Err(Error::unsupported(format!(
                "fractal heap walk found {} {what} records but the header records {}. \
                 The heap has internal free space, so enumerating it needs the version 2 \
                 B-tree index, which this reader does not implement yet",
                out.len(),
                self.managed_object_count
            )));
        }
        Ok(out)
    }

    /// Walk the heap, reporting whether every record was found.
    ///
    /// Returns the records plus a completeness flag. Use this where a partial
    /// answer is still useful and the caller can act on the flag; use
    /// [`FractalHeap::walk`] where a short list would be a correctness bug.
    ///
    /// A heap whose records have been rewritten in place develops gaps. Records
    /// after a gap are not reachable by a sequential walk, because there is no
    /// way to tell a gap from the end of the data without the B-tree.
    pub fn walk_lenient<T, F>(&self, what: &str, mut parse: F) -> Result<(Vec<T>, bool)>
    where
        F: FnMut(&mut Cursor<'_>) -> Result<T>,
    {
        let mut out = Vec::new();

        for (_, block) in &self.blocks {
            let mut cur = Cursor::new(block);
            while cur.remaining() > 0 && (out.len() as u64) < self.managed_object_count {
                // Unused space is zeroed, and every record this reader walks
                // starts with a non-zero version byte.
                if cur.peek(1)?[0] == 0 {
                    break;
                }
                let before = cur.pos();
                // A parse failure is a real defect, not the end of the records.
                // Swallowing it here would silently drop a group's children.
                let value = parse(&mut cur).map_err(|e| {
                    Error::malformed(format!(
                        "failed to parse {what} record {} at heap offset {before}: {e}",
                        out.len() + 1
                    ))
                })?;
                out.push(value);
                if cur.pos() == before {
                    return Err(Error::malformed(format!(
                        "{what} record parser consumed no bytes; refusing to loop"
                    )));
                }
            }
        }

        let complete = out.len() as u64 == self.managed_object_count;
        Ok((out, complete))
    }
}

/// The parsed fractal heap header.
#[derive(Debug, Clone)]
struct Header {
    heap_header_address: u64,
    heap_id_len: u16,
    flags: u8,
    io_filter_encoded_len: u16,
    managed_object_count: u64,
    table_width: u16,
    starting_block_size: u64,
    max_direct_block_size: u64,
    /// Width in bytes of the block-offset field inside each block.
    block_offset_size: usize,
    root_block_address: Option<u64>,
    current_root_rows: u16,
}

impl Header {
    fn read(ctx: Ctx<'_>, address: u64) -> Result<Self> {
        let sizes = ctx.sizes();
        let o = sizes.offset as usize;
        let l = sizes.length as usize;

        // Fixed part plus the filter fields; read generously and checksum the
        // exact prefix once the length is known.
        let max_len = 4 + 1 + 2 + 2 + 1 + 4 + l + o + l + o + l * 8 + 2 + l * 2 + 2 + 2 + o + 2 + l + 4 + 4;
        let buf = ctx.read_upto(address, max_len)?;
        let mut cur = Cursor::new(&buf);

        let sig = cur.take(4)?;
        if sig != FRHP_SIGNATURE {
            return Err(Error::malformed(
                "fractal heap header is missing its FRHP signature",
            ));
        }
        let version = cur.u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "fractal heap header version {version}"
            )));
        }

        let heap_id_len = cur.u16()?;
        let io_filter_encoded_len = cur.u16()?;
        let flags = cur.u8()?;
        let _max_managed_object_size = cur.u32()?;

        let _next_huge_id = cur.length(sizes)?;
        let _huge_btree_address = cur.address(sizes)?;
        let _free_space = cur.length(sizes)?;
        let _free_space_manager = cur.address(sizes)?;
        let _managed_space = cur.length(sizes)?;
        let _allocated_managed_space = cur.length(sizes)?;
        let _direct_block_iterator_offset = cur.length(sizes)?;
        let managed_object_count = cur.length(sizes)?;
        let _huge_size = cur.length(sizes)?;
        let _huge_count = cur.length(sizes)?;
        let _tiny_size = cur.length(sizes)?;
        let _tiny_count = cur.length(sizes)?;

        let table_width = cur.u16()?;
        let starting_block_size = cur.length(sizes)?;
        let max_direct_block_size = cur.length(sizes)?;
        let max_heap_size_bits = cur.u16()?;
        let _starting_root_rows = cur.u16()?;
        let root_block_address = cur.address(sizes)?;
        let current_root_rows = cur.u16()?;

        if io_filter_encoded_len > 0 {
            let _filtered_root_size = cur.length(sizes)?;
            let _filter_mask = cur.u32()?;
            cur.skip(io_filter_encoded_len as usize)?;
        }

        let checksum_pos = cur.pos();
        let stored = cur.u32()?;
        let computed = checksum::metadata(&buf[..checksum_pos]);
        if stored != computed {
            return Err(Error::ChecksumMismatch {
                what: "fractal heap header",
                stored,
                computed,
            });
        }

        if table_width == 0 || starting_block_size == 0 {
            return Err(Error::malformed(
                "fractal heap header declares a zero table width or block size",
            ));
        }

        Ok(Self {
            heap_header_address: address,
            heap_id_len,
            flags,
            io_filter_encoded_len,
            managed_object_count,
            table_width,
            starting_block_size,
            max_direct_block_size,
            block_offset_size: (max_heap_size_bits as usize).div_ceil(8),
            root_block_address,
            current_root_rows,
        })
    }

    /// Size of the direct blocks in row `row`.
    ///
    /// Rows 0 and 1 both use the starting size; each later row doubles.
    fn row_block_size(&self, row: u16) -> u64 {
        if row < 2 {
            self.starting_block_size
        } else {
            self.starting_block_size << (row as u32 - 1)
        }
    }

    /// Number of rows whose blocks are direct rather than indirect.
    fn max_direct_rows(&self) -> u16 {
        // The first row where the doubling size would exceed the maximum direct
        // block size is the first indirect row.
        let mut row: u16 = 0;
        while row < u16::MAX && self.row_block_size(row) <= self.max_direct_block_size {
            row += 1;
        }
        row
    }

    /// Header size of a direct block.
    fn direct_block_header_len(&self, sizes: crate::cursor::Sizes) -> usize {
        let checksum = if self.flags & FLAG_CHECKSUM_DIRECT_BLOCKS != 0 {
            4
        } else {
            0
        };
        4 + 1 + sizes.offset as usize + self.block_offset_size + checksum
    }
}

/// Read one direct block and push its payload.
fn read_direct_block(
    ctx: Ctx<'_>,
    header: &Header,
    address: u64,
    block_size: u64,
    out: &mut Vec<(u64, Vec<u8>)>,
) -> Result<()> {
    if out.len() > MAX_BLOCKS {
        return Err(Error::malformed(
            "fractal heap has implausibly many blocks; the file may be cyclic",
        ));
    }
    let sizes = ctx.sizes();
    let block = ctx.read(address, block_size as usize)?;
    let mut cur = Cursor::new(&block);

    let sig = cur.take(4)?;
    if sig != FHDB_SIGNATURE {
        return Err(Error::malformed(
            "fractal heap direct block is missing its FHDB signature",
        ));
    }
    let version = cur.u8()?;
    if version != 0 {
        return Err(Error::unsupported(format!(
            "fractal heap direct block version {version}"
        )));
    }
    let heap_address = cur.address_required(sizes, "direct block heap header")?;
    if heap_address != header.heap_header_address {
        return Err(Error::malformed(
            "fractal heap direct block points at a different heap header",
        ));
    }
    let block_offset = cur.uint(header.block_offset_size as u8)?;

    if header.flags & FLAG_CHECKSUM_DIRECT_BLOCKS != 0 {
        let checksum_pos = cur.pos();
        let stored = cur.u32()?;
        // The checksum covers the block with its own checksum field zeroed.
        let mut scratch = block.clone();
        scratch[checksum_pos..checksum_pos + 4].fill(0);
        let computed = checksum::metadata(&scratch);
        if stored != computed {
            return Err(Error::ChecksumMismatch {
                what: "fractal heap direct block",
                stored,
                computed,
            });
        }
    }

    let payload_start = header.direct_block_header_len(sizes);
    if payload_start > block.len() {
        return Err(Error::malformed(
            "fractal heap direct block is smaller than its own header",
        ));
    }
    // A heap ID's offset is measured in the heap's linear space, which counts
    // each block's header as well as its payload. Record where the payload
    // starts in that space so a lookup lands in the right place.
    out.push((
        block_offset + payload_start as u64,
        block[payload_start..].to_vec(),
    ));
    Ok(())
}

/// Walk an indirect block, descending into its children.
fn read_indirect_block(
    ctx: Ctx<'_>,
    header: &Header,
    address: u64,
    rows: u16,
    out: &mut Vec<(u64, Vec<u8>)>,
    depth: usize,
) -> Result<()> {
    if depth > 64 {
        return Err(Error::malformed(
            "fractal heap indirect blocks nest too deeply; the file may be cyclic",
        ));
    }
    let sizes = ctx.sizes();
    let max_direct_rows = header.max_direct_rows();

    let direct_rows = rows.min(max_direct_rows);
    let indirect_rows = rows.saturating_sub(max_direct_rows);
    let direct_entries = direct_rows as usize * header.table_width as usize;
    let indirect_entries = indirect_rows as usize * header.table_width as usize;

    let entry_len = sizes.offset as usize;
    let block_len = 4
        + 1
        + sizes.offset as usize
        + header.block_offset_size
        + direct_entries * entry_len
        + indirect_entries * entry_len
        + 4;

    let block = ctx.read(address, block_len)?;
    let mut cur = Cursor::new(&block);

    let sig = cur.take(4)?;
    if sig != FHIB_SIGNATURE {
        return Err(Error::malformed(
            "fractal heap indirect block is missing its FHIB signature",
        ));
    }
    let version = cur.u8()?;
    if version != 0 {
        return Err(Error::unsupported(format!(
            "fractal heap indirect block version {version}"
        )));
    }
    let heap_address = cur.address_required(sizes, "indirect block heap header")?;
    if heap_address != header.heap_header_address {
        return Err(Error::malformed(
            "fractal heap indirect block points at a different heap header",
        ));
    }
    cur.skip(header.block_offset_size)?;

    let checksum_pos = block.len() - 4;
    let stored = u32::from_le_bytes([
        block[checksum_pos],
        block[checksum_pos + 1],
        block[checksum_pos + 2],
        block[checksum_pos + 3],
    ]);
    let computed = checksum::metadata(&block[..checksum_pos]);
    if stored != computed {
        return Err(Error::ChecksumMismatch {
            what: "fractal heap indirect block",
            stored,
            computed,
        });
    }

    // Direct children first, row by row, so block sizes follow the row.
    for entry in 0..direct_entries {
        let row = (entry / header.table_width as usize) as u16;
        let child = cur.address(sizes)?;
        if let Some(child) = child {
            read_direct_block(ctx, header, child, header.row_block_size(row), out)?;
        }
    }

    // Then indirect children, which each cover as many rows as their size
    // implies.
    for entry in 0..indirect_entries {
        let row = max_direct_rows + (entry / header.table_width as usize) as u16;
        let child = cur.address(sizes)?;
        if let Some(child) = child {
            // A child indirect block covers the rows below its own size.
            let child_rows = row.saturating_sub(max_direct_rows) + 1;
            read_indirect_block(ctx, header, child, child_rows, out, depth + 1)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdf5::superblock::Superblock;
    use crate::hdf5::ObjectHeader;
    use crate::source::FileSource;

    const V2_FILE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_files/test_file.nc"
    );

    #[test]
    fn reads_the_root_group_link_heap_of_a_netcdf_file() {
        let src = FileSource::open(V2_FILE).unwrap();
        let sb = Superblock::read(&src).unwrap();
        let ctx = Ctx::new(&src, &sb);

        let root = ObjectHeader::read(ctx, sb.root_object_header_address().unwrap()).unwrap();
        let info = root.link_info(sb.sizes).unwrap().unwrap();
        let heap_address = info
            .fractal_heap_address
            .expect("this file stores its root links densely");

        let heap = FractalHeap::read(ctx, heap_address).unwrap();
        assert!(
            heap.managed_object_count() > 0,
            "the root group has children"
        );
        assert!(!heap.blocks().is_empty(), "the heap has direct blocks");
    }

    #[test]
    fn row_block_sizes_double_after_the_second_row() {
        let h = Header {
            heap_header_address: 0,
            heap_id_len: 8,
            flags: 0,
            io_filter_encoded_len: 0,
            managed_object_count: 0,
            table_width: 4,
            starting_block_size: 512,
            max_direct_block_size: 65536,
            block_offset_size: 2,
            root_block_address: None,
            current_root_rows: 0,
        };
        assert_eq!(h.row_block_size(0), 512);
        assert_eq!(h.row_block_size(1), 512, "rows 0 and 1 share a size");
        assert_eq!(h.row_block_size(2), 1024);
        assert_eq!(h.row_block_size(3), 2048);
        // 512 << (7-1) = 32768, still within the max; row 8 would be 65536.
        assert_eq!(h.max_direct_rows(), 9);
    }
}
