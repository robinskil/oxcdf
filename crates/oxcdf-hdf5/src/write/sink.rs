//! Byte destinations for the writer.
//!
//! A sink takes writes at absolute offsets. It holds no cursor, so the writer
//! can place each structure at its allocated address in any order.

use std::fs::File;
use std::path::Path;

use crate::error::{Error, Result};

/// A destination for bytes addressed by absolute file offset.
pub trait ByteSink: Send + std::fmt::Debug {
    /// Write all of `data` at `offset`. A write past the end grows the sink.
    ///
    /// An empty write changes nothing.
    fn write_all_at(&mut self, offset: u64, data: &[u8]) -> Result<()>;

    /// Set the size of the sink to `len` bytes. New bytes hold zeros.
    fn set_len(&mut self, len: u64) -> Result<()>;

    /// Flush all written bytes to durable storage.
    fn sync(&mut self) -> Result<()>;
}

/// Return the end of the range at `offset` with `len` bytes, or a bad request.
fn range_end(offset: u64, len: usize) -> Result<u64> {
    offset
        .checked_add(len as u64)
        .ok_or_else(|| Error::bad_request(format!("write of {len} bytes at {offset} overflows")))
}

/// A byte sink backed by an in-memory buffer.
#[derive(Debug, Default)]
pub struct MemorySink {
    data: Vec<u8>,
}

impl MemorySink {
    /// Start an empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Take the written bytes.
    pub fn into_inner(self) -> Vec<u8> {
        self.data
    }

    /// Resize the buffer to `len` bytes without a panic on a huge length.
    fn resize(&mut self, len: u64) -> Result<()> {
        let len = usize::try_from(len)
            .map_err(|_| Error::bad_request(format!("length {len} exceeds the address space")))?;
        if let Some(extra) = len.checked_sub(self.data.len()) {
            self.data
                .try_reserve(extra)
                .map_err(|_| Error::Io(std::io::ErrorKind::OutOfMemory.into()))?;
        }
        self.data.resize(len, 0);
        Ok(())
    }
}

impl ByteSink for MemorySink {
    fn write_all_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let end = range_end(offset, data.len())?;
        if end > self.data.len() as u64 {
            self.resize(end)?;
        }
        // The resize above makes the range fit, so these casts cannot truncate.
        self.data[offset as usize..end as usize].copy_from_slice(data);
        Ok(())
    }

    fn set_len(&mut self, len: u64) -> Result<()> {
        self.resize(len)
    }

    fn sync(&mut self) -> Result<()> {
        Ok(())
    }
}

/// A byte sink backed by an open file handle.
///
/// Writes go through positional I/O (`pwrite` on Unix, `seek_write` on Windows),
/// as [`crate::source::FileSource`] does for reads. On another platform each
/// write is [`Error::Unsupported`].
#[derive(Debug)]
pub struct FileSink {
    file: File,
}

impl FileSink {
    /// Create `path` for writing. An existing file is truncated.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            file: File::create(path)?,
        })
    }
}

impl ByteSink for FileSink {
    fn write_all_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        range_end(offset, data.len())?;
        write_at(&self.file, offset, data)
    }

    fn set_len(&mut self, len: u64) -> Result<()> {
        self.file.set_len(len)?;
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }
}

/// Write all of `data` to `file` at `offset` with `pwrite`.
#[cfg(unix)]
fn write_at(file: &File, offset: u64, data: &[u8]) -> Result<()> {
    use std::os::unix::fs::FileExt;
    file.write_all_at(data, offset)?;
    Ok(())
}

/// Write all of `data` to `file` at `offset` with `seek_write`.
#[cfg(windows)]
fn write_at(file: &File, offset: u64, data: &[u8]) -> Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < data.len() {
        let n = file.seek_write(&data[done..], offset + done as u64)?;
        if n == 0 {
            return Err(Error::Io(std::io::ErrorKind::WriteZero.into()));
        }
        done += n;
    }
    Ok(())
}

/// Return [`Error::Unsupported`]: this platform has no positional file write.
#[cfg(not(any(unix, windows)))]
fn write_at(_file: &File, _offset: u64, _data: &[u8]) -> Result<()> {
    Err(Error::unsupported(
        "positional file writes on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_memory_sink_grows_and_zero_fills_the_gap() {
        let mut s = MemorySink::new();
        s.write_all_at(4, b"ab").unwrap();
        s.write_all_at(0, b"x").unwrap();
        assert_eq!(s.into_inner(), b"x\0\0\0ab");
    }

    #[test]
    fn an_empty_write_changes_nothing() {
        let mut s = MemorySink::new();
        s.write_all_at(0, b"ab").unwrap();
        s.write_all_at(10, b"").unwrap();
        assert_eq!(s.into_inner(), b"ab");
    }

    #[test]
    fn set_len_extends_and_truncates() {
        let mut s = MemorySink::new();
        s.write_all_at(0, b"abcd").unwrap();
        s.set_len(6).unwrap();
        s.set_len(5).unwrap();
        assert_eq!(s.into_inner(), b"abcd\0");
    }

    #[test]
    fn a_file_sink_writes_at_offsets() {
        let path = std::env::temp_dir().join(format!("oxcdf-sink-{}.bin", std::process::id()));
        let mut s = FileSink::create(&path).unwrap();
        s.write_all_at(4, b"ab").unwrap();
        s.write_all_at(0, b"x").unwrap();
        s.set_len(8).unwrap();
        s.sync().unwrap();
        drop(s);
        assert_eq!(std::fs::read(&path).unwrap(), b"x\0\0\0ab\0\0");
        std::fs::remove_file(&path).unwrap();
    }
}
