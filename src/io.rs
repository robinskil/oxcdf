//! Merge of byte-range requests.
//!
//! A chunked read asks for many small ranges. One request for each range is
//! acceptable on a local file. It is slow on object storage. Each range is one
//! HTTP request. Each request costs milliseconds.
//!
//! Chunks of one variable sit close together. Small gaps separate them. A merge
//! of neighbours trades a little waste for fewer requests.
//!
//! The plan here is pure. Ranges go in. Merged requests and a map come out.
//! Both engines use it. A slice of a merged buffer copies nothing, because
//! [`bytes::Bytes`] shares the allocation.
//!
//! A page cache makes this redundant. Pages merge neighbours already. See
//! [`crate::cache::IoCache`].

use bytes::Bytes;

/// How aggressively to merge neighbouring ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoConfig {
    /// Merge two ranges when the unwanted gap between them is at most this.
    ///
    /// The gap's bytes are fetched and discarded, so this is the amount of
    /// waste tolerated to save one request. Set to zero to merge only ranges
    /// that actually touch.
    pub max_gap: usize,

    /// Never let a merged request exceed this size.
    ///
    /// Bounds both peak memory and the damage a single slow request can do.
    /// A range larger than this on its own is still issued whole; the limit
    /// only stops *merging* past it.
    pub max_request_size: usize,
}

impl IoConfig {
    /// Tuned for object storage, where a request costs milliseconds and
    /// bandwidth is cheap.
    pub const REMOTE: IoConfig = IoConfig {
        max_gap: 1 << 20,          // 1 MiB
        max_request_size: 1 << 23, // 8 MiB
    };

    /// Tuned for local files, where a request is a `pread` and reading bytes
    /// nobody wants is pure loss.
    pub const LOCAL: IoConfig = IoConfig {
        max_gap: 1 << 14,          // 16 KiB
        max_request_size: 1 << 22, // 4 MiB
    };

    /// Merge nothing; issue every range as asked.
    pub const NONE: IoConfig = IoConfig {
        max_gap: 0,
        max_request_size: usize::MAX,
    };
}

impl Default for IoConfig {
    fn default() -> Self {
        Self::LOCAL
    }
}

/// One request covering a run of original ranges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// Offset to read from.
    pub offset: u64,
    /// Number of bytes to read.
    pub len: usize,
    /// Which original ranges this satisfies: `(index, offset within this
    /// request, length)`.
    pub parts: Vec<(usize, usize, usize)>,
}

/// Plan the reads for `ranges`.
///
/// The result is in ascending offset order, which is also the order a store
/// serves most efficiently. Every original range is covered exactly once, and
/// the mapping records where to find it.
pub fn plan(ranges: &[(u64, usize)], config: IoConfig) -> Vec<Request> {
    if ranges.is_empty() {
        return Vec::new();
    }

    // Sort by offset, remembering where each range came from.
    let mut order: Vec<usize> = (0..ranges.len()).collect();
    order.sort_by_key(|&i| ranges[i].0);

    let mut out: Vec<Request> = Vec::new();

    for &i in &order {
        let (offset, len) = ranges[i];
        if len == 0 {
            // A zero-length range needs no bytes; record it against whatever
            // request is current so the mapping stays total.
            match out.last_mut() {
                Some(req) => req.parts.push((i, 0, 0)),
                None => out.push(Request {
                    offset,
                    len: 0,
                    parts: vec![(i, 0, 0)],
                }),
            }
            continue;
        }
        let end = offset + len as u64;

        if let Some(last) = out.last_mut() {
            let last_end = last.offset + last.len as u64;
            // Overlapping or close enough to be worth merging?
            let gap = offset.saturating_sub(last_end);
            let merged_len = (end.max(last_end) - last.offset) as usize;

            if offset >= last.offset
                && gap <= config.max_gap as u64
                && merged_len <= config.max_request_size
            {
                last.len = merged_len;
                last.parts.push((i, (offset - last.offset) as usize, len));
                continue;
            }
        }

        out.push(Request {
            offset,
            len,
            parts: vec![(i, 0, len)],
        });
    }

    out
}

/// Slice fetched request buffers back into one buffer per original range.
///
/// Zero-copy: each result shares the merged allocation rather than copying.
pub fn scatter(requests: &[Request], fetched: Vec<Bytes>, count: usize) -> crate::Result<Vec<Bytes>> {
    if requests.len() != fetched.len() {
        return Err(crate::Error::malformed(format!(
            "fetched {} buffers for {} requests",
            fetched.len(),
            requests.len()
        )));
    }

    let mut out = vec![Bytes::new(); count];
    for (request, buffer) in requests.iter().zip(fetched) {
        for &(index, at, len) in &request.parts {
            if index >= count {
                return Err(crate::Error::malformed(
                    "coalesced plan names a range that was not requested",
                ));
            }
            if at + len > buffer.len() {
                return Err(crate::Error::malformed(format!(
                    "coalesced request returned {} bytes, too few for a part at {at}+{len}",
                    buffer.len()
                )));
            }
            out[index] = buffer.slice(at..at + len);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_ranges_merge_into_one_request() {
        let ranges = [(0u64, 10usize), (10, 10), (20, 10)];
        let reqs = plan(&ranges, IoConfig::LOCAL);
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].offset, 0);
        assert_eq!(reqs[0].len, 30);
    }

    #[test]
    fn a_small_gap_is_bridged_and_a_large_one_is_not() {
        let ranges = [(0u64, 10usize), (100, 10)];

        let near = plan(&ranges, IoConfig { max_gap: 128, max_request_size: 1 << 20 });
        assert_eq!(near.len(), 1, "a 90-byte gap under the threshold merges");
        assert_eq!(near[0].len, 110);

        let far = plan(&ranges, IoConfig { max_gap: 16, max_request_size: 1 << 20 });
        assert_eq!(far.len(), 2, "a 90-byte gap over the threshold does not");
    }

    #[test]
    fn merging_stops_at_the_request_size_limit() {
        let ranges = [(0u64, 100usize), (100, 100), (200, 100)];
        let reqs = plan(&ranges, IoConfig { max_gap: 1024, max_request_size: 250 });
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].len, 200);
        assert_eq!(reqs[1].len, 100);
    }

    #[test]
    fn a_disabled_config_merges_only_touching_ranges() {
        let ranges = [(0u64, 10usize), (10, 10), (30, 10)];
        let reqs = plan(&ranges, IoConfig::NONE);
        assert_eq!(reqs.len(), 2, "touching merges, a gap does not");
    }

    #[test]
    fn out_of_order_ranges_are_sorted_and_still_map_back() {
        let ranges = [(20u64, 4usize), (0, 4), (10, 4)];
        let reqs = plan(&ranges, IoConfig::LOCAL);
        assert_eq!(reqs.len(), 1, "all within the gap threshold");

        // The merged buffer is offsets 0..24.
        let data: Vec<u8> = (0..24u8).collect();
        let got = scatter(&reqs, vec![Bytes::from(data)], 3).unwrap();
        assert_eq!(&got[0][..], &[20, 21, 22, 23], "original index 0 was offset 20");
        assert_eq!(&got[1][..], &[0, 1, 2, 3]);
        assert_eq!(&got[2][..], &[10, 11, 12, 13]);
    }

    #[test]
    fn overlapping_ranges_are_handled() {
        let ranges = [(0u64, 20usize), (10, 20)];
        let reqs = plan(&ranges, IoConfig::LOCAL);
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].len, 30);

        let data: Vec<u8> = (0..30u8).collect();
        let got = scatter(&reqs, vec![Bytes::from(data)], 2).unwrap();
        assert_eq!(got[0].len(), 20);
        assert_eq!(got[1].len(), 20);
        assert_eq!(got[1][0], 10);
    }

    #[test]
    fn scatter_is_zero_copy() {
        let ranges = [(0u64, 4usize), (4, 4)];
        let reqs = plan(&ranges, IoConfig::LOCAL);
        let buffer = Bytes::from((0..8u8).collect::<Vec<_>>());
        let got = scatter(&reqs, vec![buffer.clone()], 2).unwrap();
        // Slices of the same allocation, not copies.
        assert_eq!(got[0].as_ptr(), buffer.as_ptr());
        assert_eq!(got[1].as_ptr(), unsafe { buffer.as_ptr().add(4) });
    }

    #[test]
    fn an_empty_request_list_plans_nothing() {
        assert!(plan(&[], IoConfig::LOCAL).is_empty());
    }

    #[test]
    fn a_short_buffer_is_reported() {
        let reqs = plan(&[(0u64, 10usize)], IoConfig::LOCAL);
        let err = scatter(&reqs, vec![Bytes::from_static(b"short")], 1).unwrap_err();
        assert!(matches!(err, crate::Error::Malformed(_)), "got {err:?}");
    }

    #[test]
    fn every_range_is_covered_exactly_once() {
        let ranges: Vec<(u64, usize)> =
            (0..50).map(|i| (i as u64 * 37 % 900, 8usize)).collect();
        let reqs = plan(&ranges, IoConfig::LOCAL);

        let mut seen = vec![0usize; ranges.len()];
        for r in &reqs {
            for &(i, _, _) in &r.parts {
                seen[i] += 1;
            }
        }
        assert!(seen.iter().all(|&n| n == 1), "each range maps exactly once");
    }
}
