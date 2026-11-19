//! Byte destinations for the writer.
//!
//! A sink takes writes at absolute offsets. It holds no cursor, so the writer
//! can place each structure at its allocated address in any order.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::source::ByteSource;

/// A destination for bytes addressed by absolute file offset.
pub trait ByteSink: Send + std::fmt::Debug {
    /// Write all of `data` at `offset`. A write past the end grows the sink.
    ///
    /// Bytes below the end that get no write hold zeros. An empty write
    /// changes nothing.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BadRequest`] when `offset` plus the length of `data`
    /// passes `u64::MAX`. Returns [`Error::Io`] when the storage fails.
    fn write_all_at(&mut self, offset: u64, data: &[u8]) -> Result<()>;

    /// Set the size of the sink to `len` bytes. New bytes hold zeros.
    ///
    /// # Errors
    ///
    /// Returns an error when the storage cannot hold `len` bytes.
    fn set_len(&mut self, len: u64) -> Result<()>;

    /// Flush all written bytes to durable storage.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the storage fails.
    fn sync(&mut self) -> Result<()>;

    /// Whether [`Self::read_exact_at`] reads the bytes of this sink.
    ///
    /// The writer needs a read to change a part of a chunk that is in the
    /// file with its filters.
    fn can_read(&self) -> bool {
        false
    }

    /// Fill `buf` with the bytes at `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] for a sink that cannot read. Returns
    /// [`Error::Io`] when the storage fails or the range passes the end.
    fn read_exact_at(&mut self, _offset: u64, _buf: &mut [u8]) -> Result<()> {
        Err(Error::unsupported("a read from this sink"))
    }
}

/// Return the end of the range at `offset` with `len` bytes, or a bad request.
fn range_end(offset: u64, len: usize) -> Result<u64> {
    offset
        .checked_add(len as u64)
        .ok_or_else(|| Error::bad_request(format!("write of {len} bytes at {offset} overflows")))
}

/// A byte sink backed by an in-memory buffer.
///
/// # Examples
///
/// ```
/// use oxcdf_hdf5::write::{ByteSink, MemorySink};
///
/// let mut sink = MemorySink::new();
/// sink.write_all_at(2, b"ab")?;
/// assert_eq!(sink.into_inner(), b"\0\0ab");
/// # Ok::<(), oxcdf_hdf5::Error>(())
/// ```
#[derive(Debug, Default)]
pub struct MemorySink {
    data: Vec<u8>,
}

impl MemorySink {
    /// Start an empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start a sink that holds `data`, the bytes of a file.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { data }
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

    fn can_read(&self) -> bool {
        true
    }

    fn read_exact_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let range = usize::try_from(offset)
            .ok()
            .and_then(|start| Some(start..start.checked_add(buf.len())?));
        match range.and_then(|range| self.data.get(range)) {
            Some(bytes) => buf.copy_from_slice(bytes),
            None => return Err(Error::Io(std::io::ErrorKind::UnexpectedEof.into())),
        }
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
    file: Arc<File>,
}

/// The read side of a [`FileSink`], on the same file handle.
#[derive(Debug)]
struct SharedFile {
    file: Arc<File>,
    size: u64,
}

impl ByteSource for SharedFile {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        range_end(offset, buf.len())?;
        read_at(&self.file, offset, buf)
    }
}

impl FileSink {
    /// Create `path` for writing. An existing file is truncated.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the file cannot be created.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let mut options = File::options();
        options.read(true).write(true).create(true).truncate(true);
        Ok(Self {
            file: Arc::new(options.open(path)?),
        })
    }

    /// Open the file `path` for reads and writes, as its only writer.
    ///
    /// The sink stops a second writer of the file. Readers can open and
    /// read the file at the same time. The protection ends when the sink is
    /// dropped. The OS ends it when the process stops, so a crash leaves no
    /// state that stops a later writer.
    ///
    /// * On Windows, the handle shares the file for reads only. Each other
    ///   handle with write access fails to open, and another handle cannot
    ///   rename or remove the file. A handle that reads the file and lets
    ///   others write, as [`std::fs::File::open`] does, opens.
    /// * On Unix, the sink takes an exclusive advisory lock (`flock`) on the
    ///   file. A reader takes no lock, so the lock does not stop it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the file cannot be opened, or when another
    /// handle has it open for writes or holds a lock on it.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            file: Arc::new(open_as_only_writer(path.as_ref())?),
        })
    }

    /// A source that reads the file through the handle of this sink.
    ///
    /// The protection of [`Self::open`] does not stop these reads. The
    /// source has the size of the file at this call.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] when the file gives no metadata.
    pub fn source(&self) -> Result<Arc<dyn ByteSource>> {
        Ok(Arc::new(SharedFile {
            file: Arc::clone(&self.file),
            size: self.file.metadata()?.len(),
        }))
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

    fn can_read(&self) -> bool {
        cfg!(any(unix, windows))
    }

    fn read_exact_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        range_end(offset, buf.len())?;
        read_at(&self.file, offset, buf)
    }
}

/// Open `path` for reads and writes with a share mode that stops other writers.
///
/// A mandatory lock on Windows stops readers too, so the share mode does the work of the lock.
#[cfg(windows)]
fn open_as_only_writer(path: &Path) -> Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    /// `FILE_SHARE_READ` of the Windows API.
    const SHARE_READ: u32 = 0x1;
    /// `ERROR_SHARING_VIOLATION` of the Windows API.
    const SHARING_VIOLATION: i32 = 32;
    let mut options = File::options();
    options.read(true).write(true).share_mode(SHARE_READ);
    options
        .open(path)
        .map_err(|error| match error.raw_os_error() {
            Some(SHARING_VIOLATION) => Error::Io(std::io::Error::other(
                "another handle has the file open for writes, or does not share it with a writer",
            )),
            _ => Error::Io(error),
        })
}

/// Open `path` for reads and writes with an exclusive advisory lock.
#[cfg(not(windows))]
fn open_as_only_writer(path: &Path) -> Result<File> {
    let file = File::options().read(true).write(true).open(path)?;
    file.try_lock().map_err(|error| match error {
        std::fs::TryLockError::Error(error) => Error::Io(error),
        std::fs::TryLockError::WouldBlock => Error::Io(std::io::Error::other(
            "another handle holds a lock on the file",
        )),
    })?;
    Ok(file)
}

/// Fill `buf` from `file` at `offset` with `pread`.
#[cfg(unix)]
fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)?;
    Ok(())
}

/// Fill `buf` from `file` at `offset` with `seek_read`.
#[cfg(windows)]
fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < buf.len() {
        let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(Error::Io(std::io::ErrorKind::UnexpectedEof.into()));
        }
        done += n;
    }
    Ok(())
}

/// Return [`Error::Unsupported`]: this platform has no positional file read.
#[cfg(not(any(unix, windows)))]
fn read_at(_file: &File, _offset: u64, _buf: &mut [u8]) -> Result<()> {
    Err(Error::unsupported("positional file reads on this platform"))
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

    #[test]
    fn an_open_file_sink_stops_a_writer_and_not_a_reader() {
        let name = format!("oxcdf-sink-open-{}.bin", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, b"abcd").unwrap();
        let mut sink = FileSink::open(&path).unwrap();
        assert!(matches!(FileSink::open(&path), Err(Error::Io(_))));
        sink.write_all_at(4, b"ef").unwrap();
        let reader = crate::source::FileSource::open(&path).unwrap();
        assert_eq!(reader.read_vec(0, 6).unwrap(), b"abcdef");
        drop(reader);
        drop(sink);
        // The drop frees the file for the next writer.
        drop(FileSink::open(&path).unwrap());
        std::fs::remove_file(&path).unwrap();
    }
}
