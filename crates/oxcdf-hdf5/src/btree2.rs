//! Version 2 B-trees.
//!
//! These index a fractal heap by name, mapping a record to a heap ID. This
//! reader needs them for exactly one reason: a heap that has internal free
//! space cannot be walked sequentially, because there is no way to tell a gap
//! from the end of the data. The B-tree lists every live record, so it can.
//!
//! netcdf-c creates that situation routinely: it rewrites `REFERENCE_LIST` in
//! place as more variables come to share a dimension, and the old record
//! becomes a hole.
//!
//! # Node sizing
//!
//! Internal nodes store a child's record count in a field whose width is
//! computed, not stored. The width depends on how many records a node at that
//! depth can hold, which depends on the width itself at the level below. So the
//! sizes have to be derived level by level from the leaf upwards, exactly as the
//! library does.

use crate::checksum;
use crate::context::Ctx;
use crate::cursor::Cursor;
use crate::error::{Error, Result};

/// Signature of a version 2 B-tree header.
pub const BTHD_SIGNATURE: &[u8; 4] = b"BTHD";
/// Signature of a version 2 B-tree internal node.
pub const BTIN_SIGNATURE: &[u8; 4] = b"BTIN";
/// Signature of a version 2 B-tree leaf node.
pub const BTLF_SIGNATURE: &[u8; 4] = b"BTLF";

/// Bytes of prefix and checksum around a node's records.
const NODE_OVERHEAD: usize = 4 + 1 + 1 + 4;

/// Guard against a cyclic tree.
const MAX_NODES: usize = 1 << 20;

/// A parsed version 2 B-tree, with every record already collected.
#[derive(Debug, Clone)]
pub struct BtreeV2 {
    /// The record type this tree indexes.
    pub record_type: u8,
    /// Width of one record in bytes.
    pub record_size: usize,
    /// Every record, as raw bytes.
    pub records: Vec<Vec<u8>>,
}

impl BtreeV2 {
    /// Read the tree whose header is at `address` and collect every record.
    pub fn read(ctx: Ctx<'_>, address: u64) -> Result<Self> {
        let sizes = ctx.sizes();
        let header_len =
            4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + sizes.offset as usize + 2 + sizes.length as usize + 4;
        let buf = ctx.read_upto(address, header_len)?;
        let mut cur = Cursor::new(&buf);

        let sig = cur.take(4)?;
        if sig != BTHD_SIGNATURE {
            return Err(Error::malformed(
                "version 2 B-tree header is missing its BTHD signature",
            ));
        }
        let version = cur.u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!(
                "version 2 B-tree header version {version}"
            )));
        }
        let record_type = cur.u8()?;
        let node_size = cur.u32()? as usize;
        let record_size = cur.u16()? as usize;
        let depth = cur.u16()?;
        let _split_percent = cur.u8()?;
        let _merge_percent = cur.u8()?;
        let root_address = cur.address(sizes)?;
        let root_records = cur.u16()? as usize;
        let _total_records = cur.length(sizes)?;

        let checksum_pos = cur.pos();
        let stored = cur.u32()?;
        let computed = checksum::metadata(&buf[..checksum_pos]);
        if stored != computed {
            return Err(Error::ChecksumMismatch {
                what: "version 2 B-tree header",
                stored,
                computed,
            });
        }

        if record_size == 0 || node_size <= NODE_OVERHEAD {
            return Err(Error::malformed(
                "version 2 B-tree declares an impossible node or record size",
            ));
        }

        let layout = NodeLayout::derive(node_size, record_size, depth, sizes.offset as usize);

        let mut records = Vec::new();
        if let Some(root) = root_address {
            let mut visited = 0usize;
            collect(
                ctx,
                root,
                depth,
                root_records,
                record_size,
                &layout,
                sizes.offset as usize,
                &mut records,
                &mut visited,
            )?;
        }

        Ok(Self {
            record_type,
            record_size,
            records,
        })
    }
}

/// Per-depth field widths, derived from the leaf upwards.
#[derive(Debug)]
struct NodeLayout {
    /// Width of the "number of records" field in a child pointer.
    max_nrec_size: usize,
    /// Width of the "total number of records" field, per depth.
    cum_max_nrec_size: Vec<usize>,
}

impl NodeLayout {
    fn derive(node_size: usize, record_size: usize, depth: u16, offset_size: usize) -> Self {
        // A leaf holds this many records at most.
        let leaf_max = (node_size - NODE_OVERHEAD) / record_size;
        let max_nrec_size = enc_size(leaf_max as u64);

        let mut cum_max_nrec = vec![leaf_max as u64];
        let mut cum_max_nrec_size = vec![0usize];

        for d in 1..=depth as usize {
            let pointer_size =
                offset_size + max_nrec_size + if d > 1 { cum_max_nrec_size[d - 1] } else { 0 };
            let denominator = record_size + pointer_size;
            let numerator = node_size.saturating_sub(NODE_OVERHEAD + max_nrec_size);
            let max = if denominator == 0 {
                0
            } else {
                numerator / denominator
            } as u64;

            let cum = (max + 1) * cum_max_nrec[d - 1] + max;
            cum_max_nrec.push(cum);
            cum_max_nrec_size.push(enc_size(cum));
        }

        Self {
            max_nrec_size,
            cum_max_nrec_size,
        }
    }
}

/// Bytes needed to hold `value`, matching the library's own encoding rule.
fn enc_size(value: u64) -> usize {
    let log2 = if value == 0 {
        0
    } else {
        u64::BITS - 1 - value.leading_zeros()
    };
    ((log2 + 8) / 8) as usize
}

/// Read one node and, for an internal node, descend into its children.
#[allow(clippy::too_many_arguments)]
fn collect(
    ctx: Ctx<'_>,
    address: u64,
    depth: u16,
    record_count: usize,
    record_size: usize,
    layout: &NodeLayout,
    offset_size: usize,
    out: &mut Vec<Vec<u8>>,
    visited: &mut usize,
) -> Result<()> {
    *visited += 1;
    if *visited > MAX_NODES {
        return Err(Error::malformed(
            "version 2 B-tree has implausibly many nodes; the file may be cyclic",
        ));
    }

    if depth == 0 {
        // Leaf: a run of records between the prefix and the checksum.
        let len = NODE_OVERHEAD + record_count * record_size;
        let buf = ctx.read(address, len)?;
        let mut cur = Cursor::new(&buf);

        let sig = cur.take(4)?;
        if sig != BTLF_SIGNATURE {
            return Err(Error::malformed(
                "version 2 B-tree leaf is missing its BTLF signature",
            ));
        }
        cur.skip(2)?; // version and type

        verify_tail_checksum(&buf, "version 2 B-tree leaf")?;

        for _ in 0..record_count {
            out.push(cur.take(record_size)?.to_vec());
        }
        return Ok(());
    }

    // Internal: records, then one more child pointer than records.
    let child_nrec_size = layout.max_nrec_size;
    let child_total_size = if depth > 1 {
        layout.cum_max_nrec_size[depth as usize - 1]
    } else {
        0
    };
    let pointer_size = offset_size + child_nrec_size + child_total_size;
    let len = NODE_OVERHEAD + record_count * record_size + (record_count + 1) * pointer_size;

    let buf = ctx.read(address, len)?;
    let mut cur = Cursor::new(&buf);

    let sig = cur.take(4)?;
    if sig != BTIN_SIGNATURE {
        return Err(Error::malformed(
            "version 2 B-tree internal node is missing its BTIN signature",
        ));
    }
    cur.skip(2)?; // version and type

    verify_tail_checksum(&buf, "version 2 B-tree internal node")?;

    let mut records = Vec::with_capacity(record_count);
    for _ in 0..record_count {
        records.push(cur.take(record_size)?.to_vec());
    }

    let mut children = Vec::with_capacity(record_count + 1);
    for _ in 0..=record_count {
        let child_address = cur.uint(offset_size as u8)?;
        let child_records = cur.uint(child_nrec_size as u8)? as usize;
        if child_total_size > 0 {
            cur.skip(child_total_size)?;
        }
        children.push((child_address, child_records));
    }

    // In-order: child, record, child, record, ..., child.
    for (i, (child_address, child_records)) in children.iter().enumerate() {
        collect(
            ctx,
            *child_address,
            depth - 1,
            *child_records,
            record_size,
            layout,
            offset_size,
            out,
            visited,
        )?;
        if i < records.len() {
            out.push(records[i].clone());
        }
    }

    Ok(())
}

fn verify_tail_checksum(buf: &[u8], what: &'static str) -> Result<()> {
    let split = buf.len() - 4;
    let stored = u32::from_le_bytes([buf[split], buf[split + 1], buf[split + 2], buf[split + 3]]);
    let computed = checksum::metadata(&buf[..split]);
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
    fn encoding_widths_follow_the_library_rule() {
        assert_eq!(enc_size(0), 1);
        assert_eq!(enc_size(1), 1);
        assert_eq!(enc_size(255), 1);
        assert_eq!(enc_size(256), 2);
        assert_eq!(enc_size(65535), 2);
        assert_eq!(enc_size(65536), 3);
    }

    #[test]
    fn a_depth_zero_layout_only_needs_the_leaf_width() {
        let layout = NodeLayout::derive(512, 11, 0, 8);
        // (512 - 10) / 11 = 45 records, which fits in one byte.
        assert_eq!(layout.max_nrec_size, 1);
        assert_eq!(layout.cum_max_nrec_size.len(), 1);
    }

    #[test]
    fn deeper_layouts_add_a_width_per_level() {
        let layout = NodeLayout::derive(4096, 17, 2, 8);
        assert_eq!(layout.cum_max_nrec_size.len(), 3);
        assert!(layout.cum_max_nrec_size[1] >= 1);
    }
}
