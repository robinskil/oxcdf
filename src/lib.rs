//! A pure-Rust reader for netCDF-4 files.
//!
//! The reader parses the HDF5 container directly. It does not call the netcdf-c
//! library. Many threads can read one file at the same time.
//!
//! # Purpose
//!
//! netCDF-4 files are HDF5 files. The netcdf-c library is not thread safe. The
//! Rust bindings put one process-global mutex around every library call.
//!
//! That mutex covers the expensive work. Chunk input, decompression and type
//! conversion all run while the mutex is held. A query engine that reads many
//! files gets no parallel reads.
//!
//! This crate removes the mutex. It removes the C library.
//!
//! # Design
//!
//! The reader separates one parse from many reads.
//!
//! * An open parses the file metadata once. The result is an immutable index.
//! * The index is `Send + Sync`. Share it with [`std::sync::Arc`].
//! * A read is a pure function of the index and a request.
//! * A read holds no lock. A read changes no shared state.
//!
//! All input goes through [`ByteSource`]. Its methods take `&self`. Its methods
//! address bytes by absolute offset. There is no file position to share.
//!
//! # Two engines
//!
//! The crate has a synchronous engine and an asynchronous engine. Both engines
//! share every pure part: the parsers, the filters, the chunk arithmetic and
//! the netCDF layer. The engines differ only in how bytes arrive.
//!
//! ```text
//! plan     shared    Decide which byte ranges the read needs.
//! fetch    differs   ByteSource (sync) or AsyncByteSource (async).
//! decode   shared    Decompress, unshuffle and assemble the values.
//! ```
//!
//! The decode step stays synchronous. Decompression uses the processor. An
//! async decode would block the runtime.
//!
//! The crate holds one parser, not two. The asynchronous open runs the
//! synchronous walk over pages held in memory. It fetches the pages the walk
//! asks for. It then runs the walk again. See [`replay`].
//!
//! # Layers
//!
//! * [`netcdf`] applies the netCDF conventions. Use it for variables,
//!   dimensions and attributes.
//! * [`async_file`] is the same layer, asynchronous.
//! * [`index`] holds the HDF5 view. Use it for datasets and raw storage.
//! * [`classic`] reads the netCDF classic formats. Those files are not HDF5.
//!
//! # Example
//!
//! ```no_run
//! let file = oxcdf::open("argo.nc")?;
//! let temp = file.variable("TEMP").unwrap();
//!
//! println!("{:?} {:?}", temp.shape, temp.dimensions);
//! let values = temp.read()?.get::<f64>()?;
//! # Ok::<(), oxcdf::Error>(())
//! ```
//!
//! The asynchronous form matches it.
//!
//! ```no_run
//! # async fn run(source: std::sync::Arc<dyn oxcdf::AsyncByteSource>) -> oxcdf::Result<()> {
//! let file = oxcdf::open_async(source).await?;
//! let temp = file.variable("TEMP").unwrap();
//!
//! println!("{:?} {:?}", temp.shape, temp.dimensions);
//! let values = temp.read().await?.get::<f64>()?;
//! # Ok(()) }
//! ```
//!
//! # Writes
//!
//! The crate does not write files. A read is a parser. A write needs B-tree
//! insertion and free space management. Keep netcdf-c for writes.
//!
//! Writes lose nothing to the mutex. A query writes one file. A query reads
//! many files.
//!
//! # Scope
//!
//! The reader targets the subset of HDF5 that netcdf-c writes. It does not
//! target the whole specification.
//!
//! A feature outside that subset returns [`Error::Unsupported`]. Match on
//! [`Error::is_fallback_worthy`] to send one variable to netcdf-c. Every other
//! error marks a damaged file or a defect here. A fall back then never hides a
//! defect.

// docs.rs builds with `--cfg docsrs` on nightly, so every feature-gated item
// carries a badge that names the feature. A stable build ignores this.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![deny(rust_2018_idioms)]

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub mod async_file;
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub mod async_source;
pub mod cache;
pub mod checksum;
pub mod classic;
pub mod cursor;
pub mod dtype;
pub mod error;
pub mod extent;
pub mod filters;
pub mod hdf5;
pub mod index;
pub mod io;
#[cfg(feature = "object-store")]
#[cfg_attr(docsrs, doc(cfg(feature = "object-store")))]
pub mod object_store_source;
pub mod netcdf;
pub mod read;
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub mod replay;
pub mod source;

pub use error::{Error, Result};
pub use source::{ByteSource, FileSource, MemorySource};

// The types most callers need, at the crate root.
pub use extent::{Extent, Extents};
pub use index::OpenOptions;
pub use dtype::{DType, Element};
pub use netcdf::{
    AttributeValue, NcAttribute, NcDimension, NcVariable, NetcdfFile, Values,
    Variable,
};
pub use read::Hyperslab;

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub use async_file::{AsyncFile, AsyncVariable};
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub use async_source::{AsyncByteSource, SyncAsAsync};

/// Open a netCDF file for reading.
///
/// The file may be netCDF-4 or netCDF classic. This is the synchronous entry
/// point. Use [`open_async`] for an asynchronous one.
///
/// ```no_run
/// let file = oxcdf::open("argo.nc")?;
/// let temp = file.variable("TEMP").unwrap();
/// let values = temp.read()?.get::<f64>()?;
/// # Ok::<(), oxcdf::Error>(())
/// ```
pub fn open(path: impl AsRef<std::path::Path>) -> Result<NetcdfFile> {
    NetcdfFile::open(path)
}

/// Open a netCDF file for reading, with explicit options.
///
/// Use [`OpenOptions::remote`] for object storage.
pub fn open_with(path: impl AsRef<std::path::Path>, options: OpenOptions) -> Result<NetcdfFile> {
    NetcdfFile::open_with(path, options)
}

/// Open a netCDF-4 file over an asynchronous byte source.
///
/// The open reads the metadata. The returned file answers every metadata
/// question without further input. A read of values awaits.
///
/// ```no_run
/// # async fn run(source: std::sync::Arc<dyn oxcdf::AsyncByteSource>) -> oxcdf::Result<()> {
/// let file = oxcdf::open_async(source).await?;
/// let temp = file.variable("TEMP").unwrap();
/// let values = temp.read().await?.get::<f64>()?;
/// # Ok(()) }
/// ```
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub async fn open_async(source: std::sync::Arc<dyn AsyncByteSource>) -> Result<AsyncFile> {
    AsyncFile::open(source).await
}

/// Open a netCDF-4 file over an asynchronous byte source, with explicit options.
///
/// Use [`OpenOptions::remote`] for object storage.
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub async fn open_async_with(
    source: std::sync::Arc<dyn AsyncByteSource>,
    options: OpenOptions,
) -> Result<AsyncFile> {
    AsyncFile::open_with(source, options).await
}

/// The 8-byte signature at the front of every HDF5 file.
pub const HDF5_SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0d, 0x0a, 0x1a, 0x0a];

/// The 4-byte signatures of the netCDF classic formats.
///
/// This crate does not read them yet. It recognises them so it can return a
/// clear message instead of a confusing parse failure.
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

#[cfg(test)]
pub(crate) mod test_corpus {
    /// Absolute paths to the netCDF-4 files checked into this repository.
    ///
    /// They cover the feature matrix this reader targets: superblock v0 and v2,
    /// chunked with shuffle+deflate, contiguous, big- and little-endian floats,
    /// and fixed-length strings.
    pub fn paths() -> Vec<String> {
        let root = concat!(env!("CARGO_MANIFEST_DIR"), "/test_files");
        [
            "test_file.nc",
            "gridded-example.nc",
            "wod_ctd_1964.nc",
        ]
        .iter()
        .map(|p| format!("{root}/{p}"))
        .filter(|p| std::path::Path::new(p).exists())
        .collect()
    }
}
