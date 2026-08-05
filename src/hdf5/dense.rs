//! Dense link and attribute storage.
//!
//! Once a group has more than a handful of children, HDF5 stops writing one
//! link message per child into the object header and moves them all into a
//! fractal heap. Attributes get the same treatment. netcdf-c crosses that
//! threshold for almost every real file, so this path is the common one, not an
//! edge case.

use crate::cursor::Sizes;
use crate::error::Result;
use crate::hdf5::btree2::BtreeV2;
use crate::hdf5::context::Ctx;
use crate::hdf5::fractal::FractalHeap;
use crate::hdf5::message::{Attribute, Link};

/// Offset of the heap ID inside a "link name for indexed group" record.
///
/// The record is a 4-byte name hash followed by the heap ID.
const LINK_RECORD_HEAP_ID_OFFSET: usize = 4;
/// The heap ID opens an "attribute name for indexed attributes" record.
const ATTRIBUTE_RECORD_HEAP_ID_OFFSET: usize = 0;

/// Read every link out of a group's fractal heap.
pub fn read_dense_links(
    ctx: Ctx<'_>,
    heap_address: u64,
    name_btree_address: Option<u64>,
) -> Result<Vec<Link>> {
    let sizes = ctx.sizes();
    let heap = FractalHeap::read(ctx, heap_address)?;

    // A sequential walk is cheaper and works whenever the heap is unfragmented,
    // which is the common case. Fall back to the index only when it is not.
    let (links, complete) = heap.walk_lenient("link", |cur| Link::parse_at(cur, sizes))?;
    if complete {
        return Ok(links);
    }

    let btree_address = name_btree_address.ok_or_else(|| {
        crate::error::Error::malformed(
            "a group's link heap has gaps but the group records no name index",
        )
    })?;
    let btree = BtreeV2::read(ctx, btree_address)?;
    heap.walk_indexed(
        &btree,
        |record| &record[LINK_RECORD_HEAP_ID_OFFSET..],
        |cur| Link::parse_at(cur, sizes),
    )
}

/// Read every attribute out of an object's fractal heap.
///
/// Returns the attributes found plus whether the heap was fully enumerated.
/// Attribute heaps do develop gaps: netcdf-c rewrites `REFERENCE_LIST` in place
/// as more variables come to share a dimension, which frees the old record.
/// A short list is reported rather than hidden, so a caller can fall back to
/// netcdf-c for that object's attributes.
pub fn read_dense_attributes(
    ctx: Ctx<'_>,
    heap_address: u64,
    name_btree_address: Option<u64>,
    sizes: Sizes,
) -> Result<(Vec<Attribute>, bool)> {
    let heap = FractalHeap::read(ctx, heap_address)?;

    let (attributes, complete) =
        heap.walk_lenient("attribute", |cur| Attribute::parse_at(cur, sizes))?;
    if complete {
        return Ok((attributes, true));
    }

    // The heap has gaps, so the sequential walk cannot see past them. The name
    // index lists every live record, so use it.
    let Some(btree_address) = name_btree_address else {
        return Ok((attributes, false));
    };
    let btree = BtreeV2::read(ctx, btree_address)?;
    match heap.walk_indexed(
        &btree,
        |record| &record[ATTRIBUTE_RECORD_HEAP_ID_OFFSET..],
        |cur| Attribute::parse_at(cur, sizes),
    ) {
        Ok(indexed) => Ok((indexed, true)),
        // The asynchronous engine replays this walk until every byte is in
        // memory. Report a short read. Do not mistake it for a bad index.
        Err(crate::error::Error::Incomplete) => Err(crate::error::Error::Incomplete),
        // Keep whatever the walk found rather than losing it to an index this
        // reader cannot follow.
        Err(_) => Ok((attributes, false)),
    }
}
