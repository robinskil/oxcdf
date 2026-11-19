//! File space allocation for the writer.

use std::collections::BTreeMap;

use crate::error::{Error, Result};

/// The largest end of the file, so that no address equals the undefined address.
const MAX_END: u64 = u64::MAX - 1;

/// Gives file addresses to metadata and data. No method panics.
#[derive(Debug)]
pub struct Allocator {
    /// Next free address in the current metadata block.
    cursor: u64,
    /// First address after the current metadata block.
    block_end: u64,
    /// Size of one metadata block.
    block_bytes: u64,
    /// First address after all reserved space.
    eof: u64,
    /// Free ranges as start address to length, disjoint and not adjacent.
    free: BTreeMap<u64, u64>,
}

impl Allocator {
    /// Reserve `[start, start + metadata_block_bytes)` as the first metadata block.
    ///
    /// If that block would end past `u64::MAX - 1`, the block is empty,
    /// as with `metadata_block_bytes == 0`.
    pub fn new(start: u64, metadata_block_bytes: u64) -> Self {
        let block_bytes = match start.checked_add(metadata_block_bytes) {
            Some(end) if end <= MAX_END => metadata_block_bytes,
            _ => 0,
        };
        let block_end = start + block_bytes;
        Self {
            cursor: start,
            block_end,
            block_bytes,
            eof: block_end,
            free: BTreeMap::new(),
        }
    }

    /// Reserve `len` bytes for metadata and return their address.
    ///
    /// A request that moves the end of the file past `u64::MAX - 1` returns
    /// [`Error::BadRequest`] and changes nothing.
    pub fn alloc_metadata(&mut self, len: u64) -> Result<u64> {
        if len == 0 {
            return self.take_end(0);
        }
        if len > self.block_bytes {
            return self.take_end(len);
        }
        // The cursor never passes the block end.
        let rest_len = self.block_end - self.cursor;
        if len > rest_len {
            let block_start = self.take_end(self.block_bytes)?;
            // The rest of the old block becomes free space for data.
            let rest_start = self.cursor;
            self.cursor = block_start;
            self.block_end = self.eof;
            self.free(rest_start, rest_len);
        }
        let address = self.cursor;
        self.cursor += len;
        Ok(address)
    }

    /// Reserve `len` bytes for data and return their address.
    ///
    /// A request that moves the end of the file past `u64::MAX - 1` returns
    /// [`Error::BadRequest`] and changes nothing.
    pub fn alloc_data(&mut self, len: u64) -> Result<u64> {
        if len == 0 {
            return self.take_end(0);
        }
        let fit = self
            .free
            .iter()
            .find(|(_, &free_len)| free_len >= len)
            .map(|(&start, &free_len)| (start, free_len));
        match fit {
            Some((start, free_len)) => {
                self.free.remove(&start);
                if free_len > len {
                    self.free.insert(start + len, free_len - len);
                }
                Ok(start)
            }
            None => self.take_end(len),
        }
    }

    /// Return `[address, address + len)` to the free list and merge it with its neighbors.
    ///
    /// The range is clipped to the end of the file. A double free, or a free
    /// of live space, is a caller error that this method does not detect.
    pub fn free(&mut self, address: u64, len: u64) {
        let mut start = address;
        // Space past the end of the file is never reserved, so it is not free space.
        let mut end = address.saturating_add(len).min(self.eof);
        if start >= end {
            return;
        }
        // Free ranges lie below the end of the file, so their ends do not overflow.
        if let Some((&prev_start, &prev_len)) = self.free.range(..=start).next_back() {
            if prev_start + prev_len >= start {
                start = prev_start;
                end = end.max(prev_start + prev_len);
                self.free.remove(&prev_start);
            }
        }
        while let Some((&next_start, &next_len)) = self.free.range(start..).next() {
            if next_start > end {
                break;
            }
            end = end.max(next_start + next_len);
            self.free.remove(&next_start);
        }
        self.free.insert(start, end - start);
    }

    /// First address after all reserved space.
    pub fn end_of_file(&self) -> u64 {
        self.eof
    }

    /// Reserve `len` bytes at the end of the file and return their address.
    fn take_end(&mut self, len: u64) -> Result<u64> {
        let end = self
            .eof
            .checked_add(len)
            .filter(|&end| end <= MAX_END)
            .ok_or_else(|| {
                Error::bad_request(format!(
                    "{len} bytes at address {} pass the largest file address",
                    self.eof
                ))
            })?;
        let address = self.eof;
        self.eof = end;
        Ok(address)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn metadata_comes_from_the_front_block_and_data_follows_it() {
        let mut a = Allocator::new(48, 1024);
        assert_eq!(a.alloc_metadata(100).unwrap(), 48);
        assert_eq!(a.alloc_metadata(100).unwrap(), 148);
        assert_eq!(a.alloc_data(500).unwrap(), 1072);
        assert_eq!(a.end_of_file(), 1572);
    }

    #[test]
    fn a_full_block_makes_a_new_block_at_the_end() {
        let mut a = Allocator::new(48, 1024);
        assert_eq!(a.alloc_metadata(1000).unwrap(), 48);
        assert_eq!(a.alloc_metadata(100).unwrap(), 1072);
        assert_eq!(a.end_of_file(), 2096);
        assert_eq!(a.alloc_data(24).unwrap(), 1048); // the rest of the first block
    }

    #[test]
    fn an_oversize_metadata_request_goes_to_the_end() {
        let mut a = Allocator::new(48, 1024);
        assert_eq!(a.alloc_metadata(5000).unwrap(), 1072);
        assert_eq!(a.alloc_metadata(10).unwrap(), 48);
    }

    #[test]
    fn freed_ranges_merge_and_are_reused() {
        let mut a = Allocator::new(0, 0);
        let x = a.alloc_data(100).unwrap();
        let y = a.alloc_data(100).unwrap();
        let z = a.alloc_data(100).unwrap();
        a.free(x, 100);
        a.free(y, 100);
        assert_eq!(a.alloc_data(150).unwrap(), 0);
        assert_eq!(a.alloc_data(50).unwrap(), 150);
        assert_eq!(a.alloc_data(10).unwrap(), z + 100);
    }

    #[test]
    fn an_overflowing_request_is_a_bad_request_and_changes_nothing() {
        let mut a = Allocator::new(0, 0);
        assert_eq!(a.alloc_data(u64::MAX - 10).unwrap(), 0);
        assert!(matches!(a.alloc_data(100), Err(Error::BadRequest(_))));
        assert!(matches!(a.alloc_metadata(100), Err(Error::BadRequest(_))));
        assert_eq!(a.end_of_file(), u64::MAX - 10);
        assert_eq!(a.alloc_data(5).unwrap(), u64::MAX - 10);
    }
}
