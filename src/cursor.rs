//! Little-endian cursor over an in-memory metadata block.
//!
//! HDF5 stores all of its own metadata little-endian, whatever byte order the
//! dataset values use. So every integer read here is little-endian. Dataset
//! values are a separate concern and get byte-swapped in `data`.
//!
//! Offsets and lengths have a width that the superblock declares (almost always
//! 8 bytes). [`Sizes`] carries that width so the message parsers stay readable.

use crate::error::{Error, Result};

/// Width in bytes of file offsets and of lengths, as declared by the superblock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sizes {
    /// Width of a file address.
    pub offset: u8,
    /// Width of a length.
    pub length: u8,
}

impl Sizes {
    /// The near-universal 64-bit configuration.
    pub const EIGHT: Sizes = Sizes {
        offset: 8,
        length: 8,
    };

    /// Whether `value` is the "undefined address" for this offset width, which
    /// HDF5 encodes as all bits set.
    pub fn is_undefined_address(&self, value: u64) -> bool {
        undefined_for_width(self.offset) == value
    }
}

/// The all-bits-set sentinel for an integer of `width` bytes.
fn undefined_for_width(width: u8) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width as u32 * 8)) - 1
    }
}

/// A read cursor over a metadata buffer.
#[derive(Debug, Clone)]
pub struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// Start a cursor at the beginning of `buf`.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Start a cursor at byte `pos` of `buf`.
    pub fn at(buf: &'a [u8], pos: usize) -> Self {
        Self { buf, pos }
    }

    /// Current byte position.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Move to an absolute byte position.
    pub fn seek(&mut self, pos: usize) -> Result<()> {
        if pos > self.buf.len() {
            return Err(self.oob(pos as u64, 0));
        }
        self.pos = pos;
        Ok(())
    }

    /// Bytes left after the cursor.
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    /// Whether the cursor sits at or past the end.
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// The whole underlying buffer.
    pub fn buffer(&self) -> &'a [u8] {
        self.buf
    }

    fn oob(&self, offset: u64, len: u64) -> Error {
        Error::OutOfBounds {
            what: "metadata",
            offset,
            len,
            available: self.buf.len() as u64,
        }
    }

    /// Take the next `n` bytes.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| self.oob(self.pos as u64, n as u64))?;
        if end > self.buf.len() {
            return Err(self.oob(self.pos as u64, n as u64));
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    /// Look at the next `n` bytes without advancing.
    pub fn peek(&self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| self.oob(self.pos as u64, n as u64))?;
        if end > self.buf.len() {
            return Err(self.oob(self.pos as u64, n as u64));
        }
        Ok(&self.buf[self.pos..end])
    }

    /// Skip `n` bytes.
    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.take(n).map(|_| ())
    }

    /// Advance until the position is a multiple of `align`.
    ///
    /// Version 1 object headers pad every message to an 8-byte boundary.
    pub fn align_to(&mut self, align: usize) -> Result<()> {
        debug_assert!(align.is_power_of_two());
        let rem = self.pos % align;
        if rem != 0 {
            self.skip(align - rem)?;
        }
        Ok(())
    }

    /// Read one byte.
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// Read a little-endian `u16`.
    pub fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    /// Read a little-endian `u32`.
    pub fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read a little-endian `u64`.
    pub fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Read a little-endian unsigned integer `width` bytes wide, up to 8.
    pub fn uint(&mut self, width: u8) -> Result<u64> {
        if width == 0 {
            return Ok(0);
        }
        if width > 8 {
            return Err(Error::unsupported(format!(
                "integer width {width} bytes exceeds 8"
            )));
        }
        let bytes = self.take(width as usize)?;
        let mut value = 0u64;
        for (i, b) in bytes.iter().enumerate() {
            value |= (*b as u64) << (8 * i);
        }
        Ok(value)
    }

    /// Read a file address. Returns `None` for the undefined-address sentinel.
    pub fn address(&mut self, sizes: Sizes) -> Result<Option<u64>> {
        let raw = self.uint(sizes.offset)?;
        Ok(if sizes.is_undefined_address(raw) {
            None
        } else {
            Some(raw)
        })
    }

    /// Read a file address that must be defined.
    pub fn address_required(&mut self, sizes: Sizes, what: &str) -> Result<u64> {
        self.address(sizes)?
            .ok_or_else(|| Error::malformed(format!("{what} has an undefined address")))
    }

    /// Read a length field.
    pub fn length(&mut self, sizes: Sizes) -> Result<u64> {
        self.uint(sizes.length)
    }

    /// Read a NUL-terminated string and advance past the terminator.
    pub fn cstring(&mut self) -> Result<String> {
        let start = self.pos;
        let end = self.buf[start..]
            .iter()
            .position(|&b| b == 0)
            .map(|p| start + p)
            .ok_or_else(|| Error::malformed("unterminated string in metadata"))?;
        let s = decode_utf8(&self.buf[start..end])?;
        self.pos = end + 1;
        Ok(s)
    }

    /// Read a NUL-terminated string, then pad the total consumed length up to a
    /// multiple of `align`. Version 1 link names use 8-byte padding.
    pub fn cstring_padded(&mut self, align: usize) -> Result<String> {
        let start = self.pos;
        let s = self.cstring()?;
        let consumed = self.pos - start;
        let rem = consumed % align;
        if rem != 0 {
            self.skip(align - rem)?;
        }
        Ok(s)
    }

    /// Read exactly `n` bytes and decode them as a string, stopping at the
    /// first NUL. HDF5 pads fixed-width name fields this way.
    pub fn fixed_string(&mut self, n: usize) -> Result<String> {
        let raw = self.take(n)?;
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        decode_utf8(&raw[..end])
    }
}

/// Decode bytes as UTF-8. HDF5 permits ASCII and UTF-8 character sets, and
/// real files occasionally carry stray high bytes in ASCII fields, so fall back
/// to a lossy decode rather than failing a whole file over one bad name.
fn decode_utf8(bytes: &[u8]) -> Result<String> {
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => Ok(String::from_utf8_lossy(bytes).into_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_little_endian_integers() {
        let buf = [0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let mut c = Cursor::new(&buf);
        assert_eq!(c.u16().unwrap(), 0x0201);
        assert_eq!(c.u16().unwrap(), 0x0403);
        assert_eq!(c.u32().unwrap(), 0x0807_0605);
        assert!(c.is_empty());
    }

    #[test]
    fn reads_a_narrow_integer() {
        let buf = [0xAAu8, 0xBB, 0xCC];
        let mut c = Cursor::new(&buf);
        assert_eq!(c.uint(3).unwrap(), 0x00CC_BBAA);
    }

    #[test]
    fn detects_the_undefined_address() {
        let buf = [0xFFu8; 8];
        let mut c = Cursor::new(&buf);
        assert_eq!(c.address(Sizes::EIGHT).unwrap(), None);

        let buf4 = [0xFFu8; 4];
        let sizes = Sizes {
            offset: 4,
            length: 4,
        };
        let mut c = Cursor::new(&buf4);
        assert_eq!(c.address(sizes).unwrap(), None);
    }

    #[test]
    fn reads_a_defined_address() {
        let buf = 0x1234u64.to_le_bytes();
        let mut c = Cursor::new(&buf);
        assert_eq!(c.address(Sizes::EIGHT).unwrap(), Some(0x1234));
    }

    #[test]
    fn aligns_to_a_boundary() {
        let buf = [0u8; 16];
        let mut c = Cursor::new(&buf);
        c.skip(3).unwrap();
        c.align_to(8).unwrap();
        assert_eq!(c.pos(), 8);
        c.align_to(8).unwrap();
        assert_eq!(c.pos(), 8, "already aligned, must not move");
    }

    #[test]
    fn reads_a_nul_terminated_string() {
        let buf = b"lat\0rest";
        let mut c = Cursor::new(buf);
        assert_eq!(c.cstring().unwrap(), "lat");
        assert_eq!(c.pos(), 4);
    }

    #[test]
    fn pads_a_string_to_the_alignment() {
        // "lat\0" is 4 bytes consumed, so 4 more bytes of padding to reach 8.
        let buf = b"lat\0\0\0\0\0tail";
        let mut c = Cursor::new(buf);
        assert_eq!(c.cstring_padded(8).unwrap(), "lat");
        assert_eq!(c.pos(), 8);
    }

    #[test]
    fn reads_a_fixed_width_string() {
        let buf = b"abc\0\0\0\0\0";
        let mut c = Cursor::new(buf);
        assert_eq!(c.fixed_string(8).unwrap(), "abc");
        assert_eq!(c.pos(), 8);
    }

    #[test]
    fn rejects_a_read_past_the_end() {
        let buf = [0u8; 2];
        let mut c = Cursor::new(&buf);
        assert!(c.u32().is_err());
    }

    #[test]
    fn rejects_an_unterminated_string() {
        let buf = b"no terminator";
        let mut c = Cursor::new(buf);
        assert!(c.cstring().is_err());
    }
}
