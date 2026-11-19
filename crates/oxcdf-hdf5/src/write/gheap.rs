//! Global heap collections for variable-length values.
//!
//! The format is the inverse of [`crate::heap::GlobalHeap`].

use super::alloc::Allocator;
use super::encode::Encoder;
use super::sink::ByteSink;
use super::to_u64;
use crate::error::{Error, Result};
use crate::heap::GCOL_SIGNATURE;

/// Size of one element of a variable-length datatype: a heap reference.
pub const DESCRIPTOR_BYTES: usize = 16;

/// Default size of one collection in bytes.
const COLLECTION_BYTES: usize = 65536;

/// Size of the collection header, and of each object header.
const HEADER_BYTES: usize = 16;

/// Stores variable-length values in global heap collections.
///
/// One collection is open in memory. It goes to the sink when it is full,
/// and at [`Self::flush`].
#[derive(Debug, Default)]
pub struct HeapWriter {
    open: Option<Collection>,
}

/// A collection with reserved file space that is not in the sink yet.
#[derive(Debug)]
struct Collection {
    address: u64,
    /// Size of the reserved space in bytes.
    size: usize,
    /// The header and the objects up to now.
    bytes: Encoder,
    /// The number of objects. The next object index is one more.
    objects: u16,
}

impl HeapWriter {
    /// Store `value` and return the reference that one element holds.
    ///
    /// `count` is the length field of the reference: the number of base
    /// elements in `value`. An empty value needs no heap space.
    pub fn put<S: ByteSink>(
        &mut self,
        sink: &mut S,
        alloc: &mut Allocator,
        value: &[u8],
        count: u32,
    ) -> Result<[u8; DESCRIPTOR_BYTES]> {
        if value.is_empty() {
            return Ok([0; DESCRIPTOR_BYTES]);
        }
        self.insert(sink, alloc, value, count)
    }

    /// Store an empty value as an object and return its reference.
    ///
    /// The HDF5 library reads this reference as an empty string. It reads a
    /// reference of zeros as no string.
    pub fn put_empty<S: ByteSink>(
        &mut self,
        sink: &mut S,
        alloc: &mut Allocator,
    ) -> Result<[u8; DESCRIPTOR_BYTES]> {
        self.insert(sink, alloc, &[], 0)
    }

    /// Store `value` as an object, also when it is empty.
    fn insert<S: ByteSink>(
        &mut self,
        sink: &mut S,
        alloc: &mut Allocator,
        value: &[u8],
        count: u32,
    ) -> Result<[u8; DESCRIPTOR_BYTES]> {
        // An object is padded to a multiple of 8 bytes.
        let need = value
            .len()
            .checked_next_multiple_of(8)
            .and_then(|padded| padded.checked_add(HEADER_BYTES))
            .ok_or_else(|| Error::unsupported("a variable-length value is too large"))?;
        let fits = self.open.as_ref().is_some_and(|c| {
            let rest = c.size - c.bytes.len();
            // The rest after this object is empty or holds the free space header.
            c.objects < u16::MAX && (need == rest || need.saturating_add(HEADER_BYTES) <= rest)
        });
        if !fits {
            self.flush(sink)?;
        }
        let collection = match &mut self.open {
            Some(collection) => collection,
            open => {
                let exact = need
                    .checked_add(HEADER_BYTES)
                    .ok_or_else(|| Error::unsupported("a variable-length value is too large"))?;
                let size = exact.max(COLLECTION_BYTES);
                let address = alloc.alloc_data(to_u64(size)?)?;
                let mut bytes = Encoder::with_capacity(size);
                bytes.bytes(GCOL_SIGNATURE);
                bytes.u8(1); // version
                bytes.zeros(3);
                bytes.u64(to_u64(size)?);
                open.insert(Collection {
                    address,
                    size,
                    bytes,
                    objects: 0,
                })
            }
        };
        collection.objects += 1;
        let bytes = &mut collection.bytes;
        bytes.u16(collection.objects);
        bytes.u16(0); // reference count, as the HDF5 library writes it
        bytes.zeros(4);
        bytes.u64(to_u64(value.len())?);
        bytes.bytes(value);
        bytes.zeros(need - HEADER_BYTES - value.len());

        // The reference holds the count, the collection address and the object index.
        let mut out = [0; DESCRIPTOR_BYTES];
        out[..4].copy_from_slice(&count.to_le_bytes());
        out[4..12].copy_from_slice(&collection.address.to_le_bytes());
        out[12..].copy_from_slice(&u32::from(collection.objects).to_le_bytes());
        Ok(out)
    }

    /// Write the open collection to the sink.
    ///
    /// After an error the collection stays open, so a later call can write it.
    pub fn flush<S: ByteSink>(&mut self, sink: &mut S) -> Result<()> {
        let Some(collection) = &self.open else {
            return Ok(());
        };
        let mut tail = Encoder::new();
        let rest = collection.size - collection.bytes.len();
        if rest >= HEADER_BYTES {
            // Object 0 is the free space. Its size includes this header.
            tail.zeros(8);
            tail.u64(to_u64(rest)?);
        }
        let used = to_u64(collection.bytes.len())?;
        sink.write_all_at(collection.address, collection.bytes.as_slice())?;
        sink.write_all_at(collection.address + used, &tail.into_bytes())?;
        self.open = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Ctx;
    use crate::heap::{GlobalHeap, VlenDescriptor};
    use crate::source::MemorySource;
    use crate::superblock::Superblock;
    use crate::write::header::encode_superblock;
    use crate::write::MemorySink;

    fn read(file: Vec<u8>, address: u64) -> GlobalHeap {
        let superblock = Superblock::read(&MemorySource::new(encode_superblock(48, 4096))).unwrap();
        let source = MemorySource::new(file);
        GlobalHeap::read(Ctx::new(&source, &superblock), address).unwrap()
    }

    fn parse(descriptor: [u8; 16]) -> VlenDescriptor {
        VlenDescriptor::parse(&descriptor, crate::cursor::Sizes::EIGHT).unwrap()
    }

    #[test]
    fn values_read_back_from_one_collection() {
        let mut heap = HeapWriter::default();
        let mut sink = MemorySink::new();
        let mut alloc = Allocator::new(0, 0);
        let a = parse(heap.put(&mut sink, &mut alloc, b"hello", 5).unwrap());
        let b = parse(heap.put(&mut sink, &mut alloc, b"12345678", 8).unwrap());
        assert_eq!(heap.put(&mut sink, &mut alloc, b"", 0).unwrap(), [0; 16]);
        heap.flush(&mut sink).unwrap();
        sink.set_len(alloc.end_of_file()).unwrap();
        assert_eq!((a.length, a.collection_address, a.object_index), (5, 0, 1));
        assert_eq!((b.length, b.collection_address, b.object_index), (8, 0, 2));
        let file = sink.into_inner();
        assert_eq!(file.len(), 65536);
        // The free space object starts after two objects and holds the rest.
        assert_eq!(file[64..66], [0, 0]);
        assert_eq!(file[72..80], (65536u64 - 64).to_le_bytes());
        let heap = read(file, 0);
        assert_eq!(heap.object(1).unwrap().data, b"hello");
        assert_eq!(heap.object(2).unwrap().data, b"12345678");
    }

    #[test]
    fn a_full_collection_starts_a_new_collection() {
        let mut heap = HeapWriter::default();
        let mut sink = MemorySink::new();
        let mut alloc = Allocator::new(0, 0);
        let value = vec![7u8; 30000];
        let addresses: Vec<u64> = (0..5)
            .map(|_| {
                parse(heap.put(&mut sink, &mut alloc, &value, 30000).unwrap()).collection_address
            })
            .collect();
        assert_eq!(addresses, [0, 0, 65536, 65536, 131072]);
        heap.flush(&mut sink).unwrap();
        sink.set_len(alloc.end_of_file()).unwrap();
        let file = sink.into_inner();
        assert_eq!(read(file.clone(), 65536).object(2).unwrap().data, value);
        assert_eq!(read(file, 131072).objects.len(), 1);
    }

    #[test]
    fn a_large_value_gets_a_collection_of_its_exact_size() {
        let mut heap = HeapWriter::default();
        let mut sink = MemorySink::new();
        let mut alloc = Allocator::new(0, 0);
        let value = vec![1u8; 100_001];
        let d = parse(heap.put(&mut sink, &mut alloc, &value, 100_001).unwrap());
        assert_eq!(alloc.end_of_file(), 16 + 16 + 100_008);
        let next = parse(heap.put(&mut sink, &mut alloc, b"x", 1).unwrap());
        assert_eq!(next.collection_address, 100_040);
        heap.flush(&mut sink).unwrap();
        sink.set_len(alloc.end_of_file()).unwrap();
        let file = sink.into_inner();
        assert_eq!(
            read(file, d.collection_address).object(1).unwrap().data,
            value
        );
    }
}
