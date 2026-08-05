//! The read context: a byte source plus the superblock that interprets it.
//!
//! Every address stored inside an HDF5 file is relative to the superblock's
//! base address. Funnelling reads through one place means no parser has to
//! remember that.
//!
//! The context borrows and holds no mutable state, so it is `Copy`-cheap to
//! pass around and safe to use from several threads at once.

use crate::error::Result;
use crate::hdf5::superblock::Superblock;
use crate::source::ByteSource;

/// A file being read: the bytes, and the superblock that describes them.
#[derive(Clone, Copy)]
pub struct Ctx<'a> {
    /// Where the bytes come from.
    pub source: &'a dyn ByteSource,
    /// The parsed superblock.
    pub superblock: &'a Superblock,
    /// Decoded-chunk cache for this file, when one is in use.
    pub cache: Option<&'a crate::cache::ChunkCache>,
    /// How aggressively to merge neighbouring byte-range requests.
    pub io: crate::io::IoConfig,
    /// Raw byte cache, when one is in use.
    pub io_cache: Option<&'a crate::cache::IoCache>,
}

impl std::fmt::Debug for Ctx<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ctx")
            .field("size", &self.source.size())
            .field("superblock_version", &self.superblock.version)
            .finish()
    }
}

impl<'a> Ctx<'a> {
    /// Build a context.
    pub fn new(source: &'a dyn ByteSource, superblock: &'a Superblock) -> Self {
        Self {
            source,
            superblock,
            cache: None,
            io: crate::io::IoConfig::default(),
            io_cache: None,
        }
    }

    /// The same context, serving reads from `cache` where it can.
    pub fn with_io_cache(mut self, cache: Option<&'a crate::cache::IoCache>) -> Self {
        self.io_cache = cache;
        self
    }

    /// The same context, merging byte-range requests as `io` describes.
    pub fn with_io(mut self, io: crate::io::IoConfig) -> Self {
        self.io = io;
        self
    }

    /// The same context, consulting `cache` for decoded chunks.
    pub fn with_cache(mut self, cache: Option<&'a crate::cache::ChunkCache>) -> Self {
        self.cache = cache;
        self
    }

    /// Read `len` bytes at a stored address, applying the base address.
    pub fn read(&self, address: u64, len: usize) -> Result<Vec<u8>> {
        let offset = self.superblock.resolve(address);
        match self.io_cache {
            // Metadata reads cluster tightly, so most of them hit a page that a
            // neighbouring read already pulled in.
            Some(cache) => Ok(cache.read(self.source, offset, len)?.to_vec()),
            None => self.source.read_vec(offset, len),
        }
    }

    /// Read several stored ranges in one call.
    ///
    /// A remote source coalesces neighbouring ranges and issues the rest
    /// concurrently, so batching here turns one round trip per chunk into a
    /// handful for the whole read.
    pub fn read_ranges(&self, ranges: &[(u64, usize)]) -> Result<Vec<Vec<u8>>> {
        let resolved: Vec<(u64, usize)> = ranges
            .iter()
            .map(|&(address, len)| (self.superblock.resolve(address), len))
            .collect();

        // With a page cache, paging already merges neighbours and reuses what
        // earlier reads pulled in, so go straight through it.
        if let Some(cache) = self.io_cache {
            return resolved
                .iter()
                .map(|&(offset, len)| Ok(cache.read(self.source, offset, len)?.to_vec()))
                .collect();
        }

        // Without one, merge neighbours explicitly: chunks of one variable
        // usually sit close together, and one larger read beats several small.
        let plan = crate::io::plan(&resolved, self.io);
        let merged: Vec<(u64, usize)> = plan.iter().map(|r| (r.offset, r.len)).collect();
        let fetched = self.source.read_ranges(&merged)?;

        let parts = crate::io::scatter(
            &plan,
            fetched.into_iter().map(bytes::Bytes::from).collect(),
            resolved.len(),
        )?;
        Ok(parts.into_iter().map(|b| b.to_vec()).collect())
    }

    /// Read up to `len` bytes at a stored address, stopping at the end of the
    /// file rather than failing.
    ///
    /// Some structures do not record their own length, so the parser reads a
    /// generous window and stops when the content ends. Near the end of a file
    /// that window can overrun, which is not an error.
    pub fn read_upto(&self, address: u64, len: usize) -> Result<Vec<u8>> {
        let start = self.superblock.resolve(address);
        let available = self.source.size().saturating_sub(start);
        let want = (len as u64).min(available) as usize;
        match self.io_cache {
            Some(cache) => Ok(cache.read(self.source, start, want)?.to_vec()),
            None => self.source.read_vec(start, want),
        }
    }

    /// Address width and length width for this file.
    pub fn sizes(&self) -> crate::cursor::Sizes {
        self.superblock.sizes
    }
}
