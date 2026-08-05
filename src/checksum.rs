//! Checksums that HDF5 stores in its own metadata.
//!
//! Version 2 object headers, version 2 B-trees, fractal heaps and version 2+
//! superblocks each end with a Jenkins lookup3 checksum over the preceding
//! bytes. Verifying it catches a mis-parse immediately, which is worth a lot
//! while a from-scratch reader is young.
//!
//! Fletcher-32 is a separate thing: an optional *filter* applied to dataset
//! chunks. It lives here because it is the same kind of routine.

/// Rotate left, as the reference implementation defines it.
#[inline(always)]
fn rot(x: u32, k: u32) -> u32 {
    x.rotate_left(k)
}

#[inline(always)]
fn mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*c);
    *a ^= rot(*c, 4);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= rot(*a, 6);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= rot(*b, 8);
    *b = b.wrapping_add(*a);
    *a = a.wrapping_sub(*c);
    *a ^= rot(*c, 16);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= rot(*a, 19);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= rot(*b, 4);
    *b = b.wrapping_add(*a);
}

#[inline(always)]
fn final_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *c ^= *b;
    *c = c.wrapping_sub(rot(*b, 14));
    *a ^= *c;
    *a = a.wrapping_sub(rot(*c, 11));
    *b ^= *a;
    *b = b.wrapping_sub(rot(*a, 25));
    *c ^= *b;
    *c = c.wrapping_sub(rot(*b, 16));
    *a ^= *c;
    *a = a.wrapping_sub(rot(*c, 4));
    *b ^= *a;
    *b = b.wrapping_sub(rot(*a, 14));
    *c ^= *b;
    *c = c.wrapping_sub(rot(*b, 24));
}

/// Jenkins lookup3 `hashlittle`, which HDF5 uses for metadata checksums.
pub fn lookup3(data: &[u8], initval: u32) -> u32 {
    let len = data.len();
    let seed = 0xdead_beefu32
        .wrapping_add(len as u32)
        .wrapping_add(initval);
    let (mut a, mut b, mut c) = (seed, seed, seed);

    let mut k = data;
    while k.len() > 12 {
        a = a.wrapping_add(u32::from_le_bytes([k[0], k[1], k[2], k[3]]));
        b = b.wrapping_add(u32::from_le_bytes([k[4], k[5], k[6], k[7]]));
        c = c.wrapping_add(u32::from_le_bytes([k[8], k[9], k[10], k[11]]));
        mix(&mut a, &mut b, &mut c);
        k = &k[12..];
    }

    // The tail falls through from the high byte down, exactly as the reference
    // switch statement does.
    let n = k.len();
    if n == 0 {
        return c;
    }
    if n >= 12 {
        c = c.wrapping_add((k[11] as u32) << 24);
    }
    if n >= 11 {
        c = c.wrapping_add((k[10] as u32) << 16);
    }
    if n >= 10 {
        c = c.wrapping_add((k[9] as u32) << 8);
    }
    if n >= 9 {
        c = c.wrapping_add(k[8] as u32);
    }
    if n >= 8 {
        b = b.wrapping_add((k[7] as u32) << 24);
    }
    if n >= 7 {
        b = b.wrapping_add((k[6] as u32) << 16);
    }
    if n >= 6 {
        b = b.wrapping_add((k[5] as u32) << 8);
    }
    if n >= 5 {
        b = b.wrapping_add(k[4] as u32);
    }
    if n >= 4 {
        a = a.wrapping_add((k[3] as u32) << 24);
    }
    if n >= 3 {
        a = a.wrapping_add((k[2] as u32) << 16);
    }
    if n >= 2 {
        a = a.wrapping_add((k[1] as u32) << 8);
    }
    a = a.wrapping_add(k[0] as u32);

    final_mix(&mut a, &mut b, &mut c);
    c
}

/// The metadata checksum HDF5 appends to its structures.
pub fn metadata(data: &[u8]) -> u32 {
    lookup3(data, 0)
}

/// Fletcher-32 as HDF5 computes it for the `fletcher32` filter.
///
/// Note the big-endian pairing of bytes into 16-bit words. That detail is what
/// makes this differ from other Fletcher-32 implementations.
pub fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;

    let words = data.len() / 2;
    let mut remaining = words;
    let mut pos = 0usize;

    while remaining > 0 {
        // Reduce every 360 words so the 32-bit accumulators cannot overflow.
        let block = remaining.min(360);
        remaining -= block;
        for _ in 0..block {
            let w = ((data[pos] as u32) << 8) | (data[pos + 1] as u32);
            sum1 = sum1.wrapping_add(w);
            sum2 = sum2.wrapping_add(sum1);
            pos += 2;
        }
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    if data.len() % 2 == 1 {
        sum1 = sum1.wrapping_add((data[pos] as u32) << 8);
        sum2 = sum2.wrapping_add(sum1);
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum2 << 16) | sum1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference vectors from Bob Jenkins' lookup3.c self-test.
    #[test]
    fn lookup3_matches_reference_vectors() {
        assert_eq!(lookup3(b"", 0), 0xdeadbeef);
        assert_eq!(lookup3(b"", 0xdeadbeef), 0xbd5b7dde);
        assert_eq!(lookup3(b"Four score and seven years ago", 0), 0x17770551);
        assert_eq!(
            lookup3(b"Four score and seven years ago", 1),
            0xcd628161,
            "initval must feed the seed"
        );
    }

    #[test]
    fn lookup3_handles_every_tail_length() {
        // Exercises each fallthrough arm. The check is that no length panics
        // and that lengths produce distinct results.
        let data = b"abcdefghijklmnopqrstuvwxyz";
        let mut seen = std::collections::HashSet::new();
        for n in 0..=26 {
            seen.insert(lookup3(&data[..n], 0));
        }
        assert_eq!(seen.len(), 27, "each length should hash differently");
    }

    #[test]
    fn fletcher32_of_empty_input_is_zero() {
        assert_eq!(fletcher32(&[]), 0);
    }

    #[test]
    fn fletcher32_pairs_bytes_big_endian() {
        // One 16-bit word 0x0102: sum1 = 0x0102, sum2 = 0x0102.
        assert_eq!(fletcher32(&[0x01, 0x02]), (0x0102 << 16) | 0x0102);
    }

    #[test]
    fn fletcher32_handles_an_odd_length() {
        // Trailing byte is padded into the high half of a word.
        let got = fletcher32(&[0x01, 0x02, 0x03]);
        let sum1 = 0x0102u32 + 0x0300;
        let sum2 = 0x0102u32 + sum1;
        assert_eq!(got, (sum2 << 16) | sum1);
    }

    #[test]
    fn fletcher32_reduces_across_block_boundaries() {
        // 400 words exceeds the 360-word reduction block, so this exercises the
        // outer loop running twice.
        let data = vec![0xFFu8; 800];
        let got = fletcher32(&data);
        assert_ne!(got, 0);
    }
}
