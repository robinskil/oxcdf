//! File space allocation for the writer.

use std::collections::BTreeMap;

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
    pub fn new(start: u64, metadata_block_bytes: u64) -> Self {
        let block_end = start.saturating_add(metadata_block_bytes);
        Self {
            cursor: start,
            block_end,
            block_bytes: metadata_block_bytes,
            eof: block_end,
            free: BTreeMap::new(),
        }
    }

    /// Reserve `len` bytes for metadata and return their address.
    pub fn alloc_metadata(&mut self, len: u64) -> u64 {
        if len == 0 {
            return self.eof;
        }
        if len > self.block_bytes {
            return self.take_end(len);
        }
        let rest_len = self.block_end.saturating_sub(self.cursor);
        if len > rest_len {
            // The rest of the old block becomes free space for data.
            let rest_start = self.cursor;
            let block_start = self.take_end(self.block_bytes);
            self.cursor = block_start;
            self.block_end = block_start.saturating_add(self.block_bytes);
            self.free(rest_start, rest_len);
        }
        let address = self.cursor;
        self.cursor = self.cursor.saturating_add(len);
        address
    }

    /// Reserve `len` bytes for data and return their address.
    pub fn alloc_data(&mut self, len: u64) -> u64 {
        if len == 0 {
            return self.eof;
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
                    self.free.insert(start.saturating_add(len), free_len - len);
                }
                start
            }
            None => self.take_end(len),
        }
    }

    /// Return `[address, address + len)` to the free list and merge it with its neighbors.
    pub fn free(&mut self, address: u64, len: u64) {
        // Space past the end of the file is never reserved, so it is not free space.
        let mut start = address;
        let mut end = address.saturating_add(len).min(self.eof);
        if start >= end {
            return;
        }
        if let Some((&prev_start, &prev_len)) = self.free.range(..=start).next_back() {
            if prev_start.saturating_add(prev_len) >= start {
                start = prev_start;
                end = end.max(prev_start.saturating_add(prev_len));
                self.free.remove(&prev_start);
            }
        }
        while let Some((&next_start, &next_len)) = self.free.range(start..).next() {
            if next_start > end {
                break;
            }
            end = end.max(next_start.saturating_add(next_len));
            self.free.remove(&next_start);
        }
        self.free.insert(start, end - start);
    }

    /// First address after all reserved space.
    pub fn end_of_file(&self) -> u64 {
        self.eof
    }

    /// Reserve `len` bytes at the end of the file and return their address.
    fn take_end(&mut self, len: u64) -> u64 {
        let address = self.eof;
        self.eof = self.eof.saturating_add(len);
        address
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_comes_from_the_front_block_and_data_follows_it() {
        let mut a = Allocator::new(48, 1024);
        assert_eq!(a.alloc_metadata(100), 48);
        assert_eq!(a.alloc_metadata(100), 148);
        assert_eq!(a.alloc_data(500), 1072);
        assert_eq!(a.end_of_file(), 1572);
    }

    #[test]
    fn a_full_block_makes_a_new_block_at_the_end() {
        let mut a = Allocator::new(48, 1024);
        assert_eq!(a.alloc_metadata(1000), 48);
        assert_eq!(a.alloc_metadata(100), 1072);
        assert_eq!(a.end_of_file(), 2096);
        assert_eq!(a.alloc_data(24), 1048); // the rest of the first block
    }

    #[test]
    fn an_oversize_metadata_request_goes_to_the_end() {
        let mut a = Allocator::new(48, 1024);
        assert_eq!(a.alloc_metadata(5000), 1072);
        assert_eq!(a.alloc_metadata(10), 48);
    }

    #[test]
    fn freed_ranges_merge_and_are_reused() {
        let mut a = Allocator::new(0, 0);
        let x = a.alloc_data(100);
        let y = a.alloc_data(100);
        let z = a.alloc_data(100);
        a.free(x, 100);
        a.free(y, 100);
        assert_eq!(a.alloc_data(150), 0);
        assert_eq!(a.alloc_data(50), 150);
        assert_eq!(a.alloc_data(10), z + 100);
    }
}
