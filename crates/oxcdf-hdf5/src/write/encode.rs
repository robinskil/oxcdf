//! Little-endian byte encoder, the inverse of [`crate::cursor::Cursor`].

/// The address HDF5 writes for "no address": all bits set.
pub const UNDEFINED_ADDRESS: u64 = u64::MAX;

/// A growable buffer that appends little-endian metadata fields.
#[derive(Debug, Default)]
pub struct Encoder {
    buf: Vec<u8>,
}

impl Encoder {
    /// Start an empty encoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start an empty encoder with space for `capacity` bytes.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
        }
    }

    /// Append one byte.
    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    /// Append a little-endian `u16`.
    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a little-endian `u32`.
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append a little-endian `u64`.
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Append the low `width` bytes of `v`, little-endian.
    ///
    /// HDF5 uses widths of 1, 2, 4 or 8. A width above 8 writes 8 bytes.
    pub fn uint(&mut self, v: u64, width: u8) {
        let width = usize::from(width.min(8));
        self.buf.extend_from_slice(&v.to_le_bytes()[..width]);
    }

    /// Append an 8-byte file address. `None` writes [`UNDEFINED_ADDRESS`].
    pub fn address(&mut self, v: Option<u64>) {
        self.u64(v.unwrap_or(UNDEFINED_ADDRESS));
    }

    /// Append raw bytes.
    pub fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }

    /// Append `n` zero bytes.
    pub fn zeros(&mut self, n: usize) {
        self.buf.resize(self.buf.len() + n, 0);
    }

    /// Append the metadata checksum of every byte written so far.
    pub fn checksum(&mut self) {
        let sum = crate::checksum::metadata(&self.buf);
        self.u32(sum);
    }

    /// The number of bytes written so far.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// The bytes written so far.
    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    /// Take the encoded bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::{Cursor, Sizes};

    #[test]
    fn integers_read_back_through_cursor() {
        let mut e = Encoder::new();
        e.u8(7);
        e.u16(0x1234);
        e.u32(0xDEAD_BEEF);
        e.u64(1 << 40);
        e.uint(0x0102, 2);
        e.address(None);
        e.address(Some(48));
        let b = e.into_bytes();
        let mut c = Cursor::new(&b);
        assert_eq!(c.u8().unwrap(), 7);
        assert_eq!(c.u16().unwrap(), 0x1234);
        assert_eq!(c.u32().unwrap(), 0xDEAD_BEEF);
        assert_eq!(c.u64().unwrap(), 1 << 40);
        assert_eq!(c.uint(2).unwrap(), 0x0102);
        assert_eq!(c.address(Sizes::EIGHT).unwrap(), None);
        assert_eq!(c.address(Sizes::EIGHT).unwrap(), Some(48));
    }

    #[test]
    fn checksum_covers_every_earlier_byte() {
        let mut e = Encoder::new();
        e.bytes(b"OHDR");
        e.checksum();
        let b = e.into_bytes();
        assert_eq!(b.len(), 8);
        assert_eq!(
            u32::from_le_bytes(b[4..].try_into().unwrap()),
            crate::checksum::metadata(b"OHDR")
        );
    }
}
