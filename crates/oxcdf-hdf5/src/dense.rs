//! Dense link and attribute storage.
//!
//! Once a group has more than a handful of children, HDF5 stops writing one
//! link message per child into the object header and moves them all into a
//! fractal heap. Attributes get the same treatment. netcdf-c crosses that
//! threshold for almost every real file, so this path is the common one, not an
//! edge case.

use crate::btree2::BtreeV2;
use crate::context::Ctx;
use crate::cursor::Sizes;
use crate::error::Result;
use crate::fractal::FractalHeap;
use crate::message::{Attribute, Link};

/// Offset of the heap ID inside a "link name for indexed group" record.
///
/// The record is a 4-byte name hash followed by the heap ID.
const LINK_RECORD_HEAP_ID_OFFSET: usize = 4;
/// The heap ID opens an "attribute name for indexed attributes" record.
const ATTRIBUTE_RECORD_HEAP_ID_OFFSET: usize = 0;
/// Where that record holds the creation order: after the 8-byte heap ID and the
/// flags byte.
const ATTRIBUTE_RECORD_CREATION_ORDER: std::ops::Range<usize> = 9..13;

/// Read every link out of a group's fractal heap.
///
/// The name index lists every live record, so it is the source when the group
/// has one. A heap that lost a record keeps the old bytes, and a walk of the
/// heap alone reads them as a link.
pub fn read_dense_links(
    ctx: Ctx<'_>,
    heap_address: u64,
    name_btree_address: Option<u64>,
) -> Result<Vec<Link>> {
    let sizes = ctx.sizes();
    let heap = FractalHeap::read(ctx, heap_address)?;

    let Some(btree_address) = name_btree_address else {
        return heap.walk("link", |cur| Link::parse_at(cur, sizes));
    };
    let btree = BtreeV2::read(ctx, btree_address)?;
    let mut links = heap.walk_indexed(
        ctx,
        &btree,
        |record| &record[LINK_RECORD_HEAP_ID_OFFSET..],
        |cur| Link::parse_at(cur, sizes),
    )?;
    // The index is in hash order. Restore the order the links were made in,
    // or name order when the group does not track it, as the HDF5 library does.
    if links.iter().all(|link| link.creation_order.is_some()) {
        links.sort_by_key(|link| link.creation_order);
    } else {
        links.sort_by(|a, b| a.name.cmp(&b.name));
    }
    Ok(links)
}

/// Read every attribute out of an object's fractal heap.
///
/// Returns the attributes found plus whether the heap was fully enumerated.
/// The name index lists every live record, so it is the source when the object
/// has one. It also reaches an attribute too large for a heap block, which the
/// file stores outside the blocks.
///
/// A short list is reported rather than hidden, so a caller can fall back to
/// netcdf-c for that object's attributes.
pub fn read_dense_attributes(
    ctx: Ctx<'_>,
    heap_address: u64,
    name_btree_address: Option<u64>,
    sizes: Sizes,
) -> Result<(Vec<Attribute>, bool)> {
    let heap = FractalHeap::read(ctx, heap_address)?;

    let Some(btree_address) = name_btree_address else {
        return heap.walk_lenient("attribute", |cur| Attribute::parse_at(cur, sizes));
    };
    let mut btree = BtreeV2::read(ctx, btree_address)?;
    // The index is in hash order. netCDF reports attributes in creation order.
    btree.records.sort_by_key(|record| {
        record
            .get(ATTRIBUTE_RECORD_CREATION_ORDER)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    });
    match heap.walk_indexed(
        ctx,
        &btree,
        |record| &record[ATTRIBUTE_RECORD_HEAP_ID_OFFSET..],
        |cur| Attribute::parse_at(cur, sizes),
    ) {
        Ok(indexed) => Ok((indexed, true)),
        // The asynchronous engine replays this walk until every byte is in
        // memory. Report a short read. Do not mistake it for a bad index.
        Err(crate::error::Error::Incomplete) => Err(crate::error::Error::Incomplete),
        // Keep whatever a walk of the heap finds rather than lose it to an
        // index this reader cannot follow.
        Err(_) => {
            let (attributes, _) =
                heap.walk_lenient("attribute", |cur| Attribute::parse_at(cur, sizes))?;
            Ok((attributes, false))
        }
    }
}
