//! The fill value message: what unwritten storage reads as.
//!
//! This is easy to overlook and produces silently wrong numbers when it is.
//! A dataset that was created but never written has no bytes on disk at all.
//! Returning zeros for it looks entirely plausible, and is wrong: netCDF
//! defines its own fill values, such as `-2147483647` for a 32-bit integer, and
//! writes them into this message. The same applies to chunks that were never
//! allocated inside an otherwise-written dataset.

use crate::cursor::Cursor;
use crate::error::{Error, Result};

/// A dataset's fill value.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FillValue {
    /// The fill bytes, exactly one element wide, in the file's byte order.
    ///
    /// `None` means the file defines no fill value, in which case zero bytes
    /// are the defined behaviour.
    pub bytes: Option<Vec<u8>>,
}

impl FillValue {
    /// Fill `buf` with repetitions of the fill value.
    ///
    /// Falls back to zeroing when no fill value is defined, which is what the
    /// format specifies.
    ///
    /// A read only needs this for the part of a selection no chunk covers. See
    /// [`crate::read::read_hyperslab`], which skips the call entirely when the
    /// stored chunks cover everything.
    pub fn fill(&self, buf: &mut [u8], element_size: usize) {
        let Some(bytes) = self.bytes.as_ref() else {
            buf.fill(0);
            return;
        };
        if bytes.is_empty() || element_size == 0 {
            buf.fill(0);
            return;
        }
        // A fill value narrower than an element cannot be tiled meaningfully.
        if bytes.len() != element_size {
            buf.fill(0);
            return;
        }

        // Tile by doubling: write one element, then copy everything already
        // written over the rest, twice as much each time. That is a handful of
        // `memcpy` calls over the buffer instead of one four-byte copy for
        // every element, which showed up as 8.5% of a whole scan.
        let head = element_size.min(buf.len());
        buf[..head].copy_from_slice(&bytes[..head]);
        let mut done = head;
        while done < buf.len() {
            let take = done.min(buf.len() - done);
            let (written, rest) = buf.split_at_mut(done);
            rest[..take].copy_from_slice(&written[..take]);
            done += take;
        }
    }

    /// Whether the file defines a fill value.
    pub fn is_defined(&self) -> bool {
        self.bytes.is_some()
    }

    /// Parse a fill value message body.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut cur = Cursor::new(data);
        let version = cur.u8()?;

        match version {
            1 | 2 => {
                cur.skip(1)?; // space allocation time
                cur.skip(1)?; // fill value write time
                let defined = cur.u8()?;
                // Version 1 always carries the size and value. Version 2 only
                // carries them when the flag says so.
                if version == 1 || defined != 0 {
                    if cur.remaining() < 4 {
                        return Ok(Self { bytes: None });
                    }
                    let size = cur.u32()? as usize;
                    if size == 0 || cur.remaining() < size {
                        return Ok(Self { bytes: None });
                    }
                    return Ok(Self {
                        bytes: Some(cur.take(size)?.to_vec()),
                    });
                }
                Ok(Self { bytes: None })
            }
            3 => {
                let flags = cur.u8()?;
                // Bit 5 says a fill value follows; bit 4 says it is explicitly
                // undefined.
                if flags & 0x20 == 0 {
                    return Ok(Self { bytes: None });
                }
                if cur.remaining() < 4 {
                    return Ok(Self { bytes: None });
                }
                let size = cur.u32()? as usize;
                if size == 0 || cur.remaining() < size {
                    return Ok(Self { bytes: None });
                }
                Ok(Self {
                    bytes: Some(cur.take(size)?.to_vec()),
                })
            }
            other => Err(Error::unsupported(format!(
                "fill value message version {other}"
            ))),
        }
    }

    /// Parse the original, pre-version fill value message (`FillValueOld`),
    /// which is only a size and a value.
    pub fn parse_old(data: &[u8]) -> Result<Self> {
        let mut cur = Cursor::new(data);
        if cur.remaining() < 4 {
            return Ok(Self { bytes: None });
        }
        let size = cur.u32()? as usize;
        if size == 0 || cur.remaining() < size {
            return Ok(Self { bytes: None });
        }
        Ok(Self {
            bytes: Some(cur.take(size)?.to_vec()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_version_three_defined_fill_value() {
        let mut d = vec![3u8, 0x20];
        d.extend_from_slice(&4u32.to_le_bytes());
        d.extend_from_slice(&(-2147483647i32).to_le_bytes());
        let f = FillValue::parse(&d).unwrap();
        assert!(f.is_defined());
        assert_eq!(f.bytes.unwrap(), (-2147483647i32).to_le_bytes());
    }

    #[test]
    fn a_version_three_message_without_the_flag_defines_nothing() {
        let f = FillValue::parse(&[3u8, 0x00]).unwrap();
        assert!(!f.is_defined());
    }

    #[test]
    fn parses_a_version_two_defined_fill_value() {
        let mut d = vec![2u8, 2, 2, 1];
        d.extend_from_slice(&2u32.to_le_bytes());
        d.extend_from_slice(&7i16.to_le_bytes());
        let f = FillValue::parse(&d).unwrap();
        assert_eq!(f.bytes.unwrap(), 7i16.to_le_bytes());
    }

    #[test]
    fn a_version_two_message_marked_undefined_defines_nothing() {
        let f = FillValue::parse(&[2u8, 2, 2, 0]).unwrap();
        assert!(!f.is_defined());
    }

    #[test]
    fn fill_tiles_the_value_across_the_buffer() {
        let f = FillValue {
            bytes: Some(vec![0xAA, 0xBB]),
        };
        let mut buf = vec![0u8; 6];
        f.fill(&mut buf, 2);
        assert_eq!(buf, vec![0xAA, 0xBB, 0xAA, 0xBB, 0xAA, 0xBB]);
    }

    #[test]
    fn fill_tiles_a_buffer_that_is_not_whole_elements() {
        // The doubling must not run past the end, and the part of an element
        // that fits gets the start of the value.
        let f = FillValue {
            bytes: Some(vec![1, 2, 3, 4]),
        };
        let mut buf = vec![0u8; 10];
        f.fill(&mut buf, 4);
        assert_eq!(buf, vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2]);
    }

    #[test]
    fn fill_tiles_a_buffer_shorter_than_one_element() {
        let f = FillValue {
            bytes: Some(vec![9, 8, 7, 6]),
        };
        let mut buf = vec![0u8; 3];
        f.fill(&mut buf, 4);
        assert_eq!(buf, vec![9, 8, 7]);
    }

    #[test]
    fn fill_tiles_a_long_buffer_exactly() {
        // Long enough that the doubling takes several rounds.
        let value = (-2147483647i32).to_ne_bytes();
        let f = FillValue {
            bytes: Some(value.to_vec()),
        };
        let mut buf = vec![0u8; 4 * 1000];
        f.fill(&mut buf, 4);
        assert!(buf.chunks_exact(4).all(|c| c == value));
    }

    #[test]
    fn fill_leaves_an_empty_buffer_alone() {
        let f = FillValue {
            bytes: Some(vec![1, 2]),
        };
        f.fill(&mut [], 2);
    }

    #[test]
    fn fill_zeroes_when_nothing_is_defined() {
        let f = FillValue { bytes: None };
        let mut buf = vec![9u8; 4];
        f.fill(&mut buf, 2);
        assert_eq!(buf, vec![0, 0, 0, 0]);
    }

    #[test]
    fn fill_zeroes_when_the_value_width_does_not_match_the_element() {
        // Refusing to tile a mismatched width is safer than guessing.
        let f = FillValue {
            bytes: Some(vec![1, 2, 3]),
        };
        let mut buf = vec![9u8; 4];
        f.fill(&mut buf, 2);
        assert_eq!(buf, vec![0, 0, 0, 0]);
    }

    #[test]
    fn rejects_an_unknown_version() {
        assert!(FillValue::parse(&[9u8, 0, 0, 0]).is_err());
    }
}
