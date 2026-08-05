//! Which container format a file uses.
//!
//! A netCDF file is either an HDF5 container (netCDF-4) or one of the three
//! classic containers. Only the first is HDF5, so this crate reads only the
//! first. It still recognises the others, so a caller gets a clear message
//! instead of a confusing parse failure, and so the netCDF layer above can
//! route the file to its classic reader.

use crate::error::{Error, Result};
use crate::source::ByteSource;

/// The 8-byte signature at the front of every HDF5 file.
pub const HDF5_SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0d, 0x0a, 0x1a, 0x0a];

/// The 4-byte signatures of the netCDF classic formats.
pub const CDF_SIGNATURES: [[u8; 4]; 3] = [
    *b"CDF\x01", // classic
    *b"CDF\x02", // 64-bit offset
    *b"CDF\x05", // 64-bit data (CDF-5)
];

/// Which container a file uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Container {
    /// netCDF-4, stored in an HDF5 container.
    Hdf5,
    /// netCDF classic, 32-bit offsets.
    Cdf1,
    /// netCDF classic, 64-bit offsets.
    Cdf2,
    /// netCDF classic, 64-bit data.
    Cdf5,
}

/// Identify the container by its magic bytes.
///
/// HDF5 permits a user block before the signature. The block size is a power of
/// two of at least 512, so probe those offsets too.
pub fn detect_container(source: &dyn ByteSource) -> Result<Container> {
    let mut magic = [0u8; 8];
    if source.size() >= 8 {
        source.read_exact_at(0, &mut magic)?;
        if magic == HDF5_SIGNATURE {
            return Ok(Container::Hdf5);
        }
        let four = [magic[0], magic[1], magic[2], magic[3]];
        if four == CDF_SIGNATURES[0] {
            return Ok(Container::Cdf1);
        }
        if four == CDF_SIGNATURES[1] {
            return Ok(Container::Cdf2);
        }
        if four == CDF_SIGNATURES[2] {
            return Ok(Container::Cdf5);
        }
    }

    let mut probe = 512u64;
    while probe + 8 <= source.size() {
        source.read_exact_at(probe, &mut magic)?;
        if magic == HDF5_SIGNATURE {
            return Ok(Container::Hdf5);
        }
        probe *= 2;
    }

    Err(Error::malformed(
        "file is neither HDF5 (netCDF-4) nor a netCDF classic container",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{FileSource, MemorySource};

    #[test]
    fn detects_hdf5_at_offset_zero() {
        let mut data = HDF5_SIGNATURE.to_vec();
        data.extend_from_slice(&[0u8; 64]);
        let src = MemorySource::new(data);
        assert_eq!(detect_container(&src).unwrap(), Container::Hdf5);
    }

    #[test]
    fn detects_hdf5_behind_a_user_block() {
        let mut data = vec![0u8; 512];
        data.extend_from_slice(&HDF5_SIGNATURE);
        data.extend_from_slice(&[0u8; 64]);
        let src = MemorySource::new(data);
        assert_eq!(detect_container(&src).unwrap(), Container::Hdf5);
    }

    #[test]
    fn detects_the_classic_containers() {
        for (sig, want) in [
            (CDF_SIGNATURES[0], Container::Cdf1),
            (CDF_SIGNATURES[1], Container::Cdf2),
            (CDF_SIGNATURES[2], Container::Cdf5),
        ] {
            let mut data = sig.to_vec();
            data.extend_from_slice(&[0u8; 64]);
            let src = MemorySource::new(data);
            assert_eq!(detect_container(&src).unwrap(), want);
        }
    }

    #[test]
    fn rejects_an_unknown_container() {
        let src = MemorySource::new(vec![0u8; 4096]);
        assert!(detect_container(&src).is_err());
    }

    #[test]
    fn detects_hdf5_in_the_real_corpus() {
        for path in crate::test_corpus::paths() {
            let src = FileSource::open(&path).unwrap();
            assert_eq!(
                detect_container(&src).unwrap(),
                Container::Hdf5,
                "{path} should be netCDF-4"
            );
        }
    }
}
