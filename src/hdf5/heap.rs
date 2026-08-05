//! Local and global heaps.
//!
//! A *local* heap holds the link names of an old-style group. Symbol table
//! entries store a byte offset into it rather than the name itself.
//!
//! A *global* heap holds variable-length values. A vlen element in a dataset or
//! attribute is a small descriptor naming a collection and an object inside it,
//! and the bytes live here. netCDF needs this for `DIMENSION_LIST`, which is a
//! vlen sequence of object references.

use crate::cursor::Cursor;
use crate::error::{Error, Result};
use crate::hdf5::context::Ctx;

/// Signature of a local heap.
pub const HEAP_SIGNATURE: &[u8; 4] = b"HEAP";
/// Signature of a global heap collection.
pub const GCOL_SIGNATURE: &[u8; 4] = b"GCOL";

/// A local heap: a flat byte segment addressed by offset.
#[derive(Debug, Clone)]
pub struct LocalHeap {
    data: Vec<u8>,
}

impl LocalHeap {
    /// Read the local heap at `address`.
    pub fn read(ctx: Ctx<'_>, address: u64) -> Result<Self> {
        let sizes = ctx.sizes();
        // Header: signature, version, 3 reserved, then two lengths and one
        // address.
        let header_len = 8 + 2 * sizes.length as usize + sizes.offset as usize;
        let header = ctx.read(address, header_len)?;
        let mut cur = Cursor::new(&header);

        let sig = cur.take(4)?;
        if sig != HEAP_SIGNATURE {
            return Err(Error::malformed("local heap is missing its HEAP signature"));
        }
        let version = cur.u8()?;
        if version != 0 {
            return Err(Error::unsupported(format!("local heap version {version}")));
        }
        cur.skip(3)?; // reserved

        let segment_size = cur.length(sizes)? as usize;
        let _free_list_head = cur.length(sizes)?;
        let segment_address = cur.address_required(sizes, "local heap data segment")?;

        Ok(Self {
            data: ctx.read(segment_address, segment_size)?,
        })
    }

    /// The NUL-terminated name stored at `offset`.
    pub fn name_at(&self, offset: u64) -> Result<String> {
        let offset = offset as usize;
        if offset >= self.data.len() {
            return Err(Error::malformed(format!(
                "local heap offset {offset} is past the end of the {}-byte segment",
                self.data.len()
            )));
        }
        let rest = &self.data[offset..];
        let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        Ok(String::from_utf8_lossy(&rest[..end]).into_owned())
    }

    /// Size of the heap's data segment.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the heap is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// One object inside a global heap collection.
#[derive(Debug, Clone)]
pub struct GlobalHeapObject {
    /// Index of this object within its collection.
    pub index: u16,
    /// Number of references to it.
    pub reference_count: u16,
    /// The stored bytes.
    pub data: Vec<u8>,
}

/// A global heap collection.
#[derive(Debug, Clone)]
pub struct GlobalHeap {
    /// Objects, keyed by their index within the collection.
    pub objects: Vec<GlobalHeapObject>,
}

impl GlobalHeap {
    /// Read the global heap collection at `address`.
    pub fn read(ctx: Ctx<'_>, address: u64) -> Result<Self> {
        let sizes = ctx.sizes();
        let header_len = 8 + sizes.length as usize;
        let header = ctx.read(address, header_len)?;
        let mut cur = Cursor::new(&header);

        let sig = cur.take(4)?;
        if sig != GCOL_SIGNATURE {
            return Err(Error::malformed("global heap is missing its GCOL signature"));
        }
        let version = cur.u8()?;
        if version != 1 {
            return Err(Error::unsupported(format!("global heap version {version}")));
        }
        cur.skip(3)?; // reserved
        let collection_size = cur.length(sizes)? as usize;

        if collection_size < header_len {
            return Err(Error::malformed(
                "global heap collection size is smaller than its own header",
            ));
        }

        let whole = ctx.read(address, collection_size)?;
        let mut cur = Cursor::at(&whole, header_len);

        // Each object header is 2 + 2 + 4 bytes plus one length field.
        let object_header_len = 8 + sizes.length as usize;
        let mut objects = Vec::new();

        while cur.remaining() >= object_header_len {
            let index = cur.u16()?;
            let reference_count = cur.u16()?;
            cur.skip(4)?; // reserved
            let size = cur.length(sizes)? as usize;

            // Index zero marks the free space that closes the collection.
            if index == 0 {
                break;
            }
            if size > cur.remaining() {
                return Err(Error::malformed(format!(
                    "global heap object {index} claims {size} bytes but only {} remain",
                    cur.remaining()
                )));
            }

            let data = cur.take(size)?.to_vec();
            objects.push(GlobalHeapObject {
                index,
                reference_count,
                data,
            });

            // Objects are padded out to an 8-byte boundary.
            let pad = (8 - (size % 8)) % 8;
            if pad > 0 {
                if cur.remaining() < pad {
                    break;
                }
                cur.skip(pad)?;
            }
        }

        Ok(Self { objects })
    }

    /// The object with the given index.
    pub fn object(&self, index: u16) -> Option<&GlobalHeapObject> {
        self.objects.iter().find(|o| o.index == index)
    }
}

/// A pointer to a variable-length value.
///
/// This is the descriptor stored inline in a dataset or attribute: a length, a
/// collection address and an index within that collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VlenDescriptor {
    /// Number of elements in the sequence, or bytes in the string.
    pub length: u32,
    /// Address of the global heap collection holding the bytes.
    pub collection_address: u64,
    /// Index of the object within that collection.
    pub object_index: u32,
}

impl VlenDescriptor {
    /// Parse a vlen descriptor from the inline bytes of one element.
    pub fn parse(data: &[u8], sizes: crate::cursor::Sizes) -> Result<Self> {
        let mut cur = Cursor::new(data);
        let length = cur.u32()?;
        let collection_address = cur.address_required(sizes, "vlen global heap collection")?;
        let object_index = cur.u32()?;
        Ok(Self {
            length,
            collection_address,
            object_index,
        })
    }

    /// Width of a descriptor on disk.
    pub fn encoded_len(sizes: crate::cursor::Sizes) -> usize {
        4 + sizes.offset as usize + 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::Sizes;
    use crate::hdf5::superblock::Superblock;
    use crate::source::{FileSource, MemorySource};

    const LEGACY_FILE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_files/legacy_v1_objheader.h5"
    );

    #[test]
    fn reads_names_out_of_a_local_heap() {
        // Build a file holding a heap header followed by its data segment.
        let mut data = Vec::new();
        data.extend_from_slice(HEAP_SIGNATURE);
        data.push(0); // version
        data.extend_from_slice(&[0, 0, 0]); // reserved
        data.extend_from_slice(&32u64.to_le_bytes()); // segment size
        data.extend_from_slice(&0u64.to_le_bytes()); // free list head
        data.extend_from_slice(&32u64.to_le_bytes()); // segment address
        assert_eq!(data.len(), 32);
        let mut segment = vec![0u8; 32];
        segment[..4].copy_from_slice(b"lat\0");
        segment[8..13].copy_from_slice(b"time\0");
        data.extend_from_slice(&segment);

        let src = MemorySource::new(data);
        let sb = fake_superblock();
        let ctx = Ctx::new(&src, &sb);
        let heap = LocalHeap::read(ctx, 0).unwrap();

        assert_eq!(heap.name_at(0).unwrap(), "lat");
        assert_eq!(heap.name_at(8).unwrap(), "time");
        assert_eq!(heap.len(), 32);
    }

    #[test]
    fn rejects_a_local_heap_offset_past_the_segment() {
        let heap = LocalHeap {
            data: b"abc\0".to_vec(),
        };
        assert!(heap.name_at(99).is_err());
    }

    #[test]
    fn reads_the_real_local_heaps_of_the_legacy_fixture() {
        let src = FileSource::open(LEGACY_FILE).unwrap();
        let sb = Superblock::read(&src).unwrap();
        let ctx = Ctx::new(&src, &sb);

        let root = crate::hdf5::ObjectHeader::read(ctx, sb.root_object_header_address().unwrap())
            .unwrap();
        let st = root.symbol_table(sb.sizes).unwrap().unwrap();
        let heap = LocalHeap::read(ctx, st.local_heap_address).unwrap();

        assert!(!heap.is_empty(), "the root group has named children");
        // Offset 0 of a local heap is always an empty string; real names follow.
        let joined: String = heap.data.iter().map(|&b| b as char).collect();
        assert!(
            joined.contains("contig_f64"),
            "the heap should hold the fixture's dataset names, got {joined:?}"
        );
    }

    #[test]
    fn parses_a_vlen_descriptor() {
        let mut d = Vec::new();
        d.extend_from_slice(&3u32.to_le_bytes());
        d.extend_from_slice(&0x1000u64.to_le_bytes());
        d.extend_from_slice(&2u32.to_le_bytes());
        let v = VlenDescriptor::parse(&d, Sizes::EIGHT).unwrap();
        assert_eq!(
            v,
            VlenDescriptor {
                length: 3,
                collection_address: 0x1000,
                object_index: 2
            }
        );
        assert_eq!(VlenDescriptor::encoded_len(Sizes::EIGHT), 16);
    }

    #[test]
    fn reads_objects_out_of_a_global_heap() {
        let mut data = Vec::new();
        data.extend_from_slice(GCOL_SIGNATURE);
        data.push(1); // version
        data.extend_from_slice(&[0, 0, 0]);
        // Collection size is filled in once the body is known.
        let size_pos = data.len();
        data.extend_from_slice(&0u64.to_le_bytes());

        // Object 1: 5 bytes, padded to 8.
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes());
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(&5u64.to_le_bytes());
        data.extend_from_slice(b"hello");
        data.extend_from_slice(&[0; 3]); // padding

        let total = data.len() as u64;
        data[size_pos..size_pos + 8].copy_from_slice(&total.to_le_bytes());

        let src = MemorySource::new(data);
        let sb = fake_superblock();
        let ctx = Ctx::new(&src, &sb);
        let heap = GlobalHeap::read(ctx, 0).unwrap();

        assert_eq!(heap.objects.len(), 1);
        assert_eq!(heap.object(1).unwrap().data, b"hello");
        assert!(heap.object(9).is_none());
    }

    /// A superblock is needed only for its `sizes` and base address here.
    fn fake_superblock() -> Superblock {
        let mut data = crate::HDF5_SIGNATURE.to_vec();
        data.push(2); // version
        data.push(8); // offset size
        data.push(8); // length size
        data.push(0); // flags
        data.extend_from_slice(&0u64.to_le_bytes()); // base address
        data.extend_from_slice(&u64::MAX.to_le_bytes()); // extension
        data.extend_from_slice(&0u64.to_le_bytes()); // eof
        data.extend_from_slice(&0u64.to_le_bytes()); // root header
        let checksum = crate::checksum::metadata(&data);
        data.extend_from_slice(&checksum.to_le_bytes());
        let src = MemorySource::new(data);
        Superblock::read(&src).unwrap()
    }
}
