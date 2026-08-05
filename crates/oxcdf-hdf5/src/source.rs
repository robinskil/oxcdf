//! Byte sources for random access.
//!
//! This trait makes parallel reads possible. Every method takes `&self`. No
//! implementation holds a position. Many threads can read different byte ranges
//! of one file at the same time. There is no position to share, so there is no
//! lock.
//!
//! This trait is also the seam for remote storage. An `ObjectStore`
//! implementation serves ranges over HTTP. Nothing above this module changes.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::error::{Error, Result};

/// A source of bytes addressed by absolute file offset.
///
/// Implementations must be cheap to share and safe to call concurrently.
pub trait ByteSource: Send + Sync + std::fmt::Debug {
    /// Total size of the source in bytes.
    fn size(&self) -> u64;

    /// Fill `buf` with the bytes starting at `offset`.
    ///
    /// Returns [`Error::OutOfBounds`] when the range runs past the end. A short
    /// read is an error, never a silent truncation.
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()>;

    /// Read `len` bytes at `offset` into a fresh buffer.
    fn read_vec(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.read_exact_at(offset, &mut buf)?;
        Ok(buf)
    }

    /// Read several ranges at once.
    ///
    /// The default implementation issues them one by one. A remote
    /// implementation should override this to coalesce and pipeline requests,
    /// which is where most of the win is on object storage.
    fn read_ranges(&self, ranges: &[(u64, usize)]) -> Result<Vec<Vec<u8>>> {
        ranges
            .iter()
            .map(|&(offset, len)| self.read_vec(offset, len))
            .collect()
    }
}

impl ByteSource for Arc<dyn ByteSource> {
    fn size(&self) -> u64 {
        (**self).size()
    }
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        (**self).read_exact_at(offset, buf)
    }
    fn read_vec(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        (**self).read_vec(offset, len)
    }
    fn read_ranges(&self, ranges: &[(u64, usize)]) -> Result<Vec<Vec<u8>>> {
        (**self).read_ranges(ranges)
    }
}

/// A byte source backed by an open file handle.
///
/// Reads go through positional I/O (`pread` on Unix, `seek_read` on Windows).
/// Those take `&self` and do not move a shared file cursor, so concurrent reads
/// on one handle are safe and need no lock.
#[derive(Debug)]
pub struct FileSource {
    file: File,
    size: u64,
}

impl FileSource {
    /// Open `path` for reading.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        Ok(Self { file, size })
    }

    /// Wrap an already-open file.
    pub fn from_file(file: File) -> Result<Self> {
        let size = file.metadata()?.len();
        Ok(Self { file, size })
    }
}

impl ByteSource for FileSource {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let want = buf.len() as u64;
        if offset.saturating_add(want) > self.size {
            return Err(Error::OutOfBounds {
                what: "file",
                offset,
                len: want,
                available: self.size.saturating_sub(offset),
            });
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(buf, offset)?;
        }

        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            let mut done = 0usize;
            while done < buf.len() {
                let n = self
                    .file
                    .seek_read(&mut buf[done..], offset + done as u64)?;
                if n == 0 {
                    return Err(Error::OutOfBounds {
                        what: "file",
                        offset,
                        len: want,
                        available: done as u64,
                    });
                }
                done += n;
            }
        }

        Ok(())
    }
}

/// A byte source backed by an in-memory buffer. Useful for tests and for files
/// small enough to hold whole.
#[derive(Debug)]
pub struct MemorySource {
    data: Vec<u8>,
}

impl MemorySource {
    /// Wrap an owned buffer.
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Read an entire file into memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::new(std::fs::read(path)?))
    }
}

impl ByteSource for MemorySource {
    fn size(&self) -> u64 {
        self.data.len() as u64
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let end = offset.saturating_add(buf.len() as u64);
        if end > self.data.len() as u64 {
            return Err(Error::OutOfBounds {
                what: "buffer",
                offset,
                len: buf.len() as u64,
                available: (self.data.len() as u64).saturating_sub(offset),
            });
        }
        buf.copy_from_slice(&self.data[offset as usize..end as usize]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_source_reads_a_range() {
        let src = MemorySource::new(vec![1, 2, 3, 4, 5]);
        assert_eq!(src.read_vec(1, 3).unwrap(), vec![2, 3, 4]);
        assert_eq!(src.size(), 5);
    }

    #[test]
    fn memory_source_rejects_a_read_past_the_end() {
        let src = MemorySource::new(vec![1, 2, 3]);
        let err = src.read_vec(2, 5).unwrap_err();
        assert!(matches!(err, Error::OutOfBounds { .. }), "got {err:?}");
    }

    #[test]
    fn memory_source_rejects_an_offset_past_the_end() {
        let src = MemorySource::new(vec![1, 2, 3]);
        assert!(src.read_vec(99, 1).is_err());
    }
}
