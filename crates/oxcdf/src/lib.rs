//! A pure-Rust reader for netCDF-4 and netCDF classic files.
//!
//! The reader parses the container directly. It does not call the netcdf-c
//! library. Many threads read one file at the same time.
//!
//! The interface matches the `netcdf` crate. A program moves across with few
//! changes.
//!
//! # Example
//!
//! ```no_run
//! let file = oxcdf::open("argo.nc")?;
//! let temp = file.variable("TEMP").unwrap();
//!
//! println!("{:?} {:?}", temp.vartype(), temp.shape);
//! let values = temp.get_values::<f64, _>(..)?;
//! let slice = temp.get_values::<f64, _>([0..8, 10..30])?;
//! # Ok::<(), oxcdf::Error>(())
//! ```
//!
//! The asynchronous interface is the same list. Only the reads await.
//!
//! ```no_run
//! # async fn run(source: std::sync::Arc<dyn oxcdf::AsyncByteSource>) -> oxcdf::Result<()> {
//! let file = oxcdf::open_async(source).await?;
//! let temp = file.variable("TEMP").unwrap();
//! let values = temp.get_values::<f64, _>(..).await?;
//! # Ok(()) }
//! ```
//!
//! # Why
//!
//! netCDF-4 files are HDF5 files. The netcdf-c library is not thread safe. Its
//! Rust bindings hold one process-global mutex for every call.
//!
//! That mutex covers the expensive work: input, decompression and conversion.
//! A query engine that reads many files gets no parallel reads.
//!
//! This crate has no mutex. This crate has no C library.
//!
//! # Design
//!
//! An open parses the metadata once. The result is an immutable index. Share it
//! with [`std::sync::Arc`].
//!
//! A read is a pure function of the index and a request. A read holds no lock.
//!
//! All input goes through [`ByteSource`]. Its methods take `&self` and address
//! bytes by absolute offset. There is no file position to share.
//!
//! # Two crates
//!
//! [`oxcdf_hdf5`] reads the HDF5 container. It knows nothing about netCDF.
//! This crate applies the netCDF conventions on top and re-exports it, under
//! the name `hdf5`. One dependency is enough.
//!
//! Use `oxcdf::hdf5` for storage this interface does not model, such as the
//! chunk grid and ragged arrays. The `netcdf` crate models neither.
//!
//! # Layout
//!
//! * [`netcdf`] holds the netCDF view: variables, dimensions and attributes.
//! * `async_netcdf` is the same view, asynchronous.
//! * [`classic`] reads the netCDF classic formats. Those files are not HDF5.
//!
//! [`open`] reads the magic bytes and picks the container. netCDF-4 and classic
//! then use the same interface.
//!
//! # Scope
//!
//! The reader targets the subset of HDF5 that netcdf-c writes.
//!
//! A feature outside that subset returns [`Error::Unsupported`]. Match on
//! [`Error::is_fallback_worthy`] to send one variable to netcdf-c. Every other
//! error marks a damaged file or a defect here.
//!
//! The crate does not write files. Keep netcdf-c for writes.

// docs.rs builds with `--cfg docsrs` on nightly, so every feature-gated item
// carries a badge that names the feature. A stable build ignores this.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![deny(rust_2018_idioms)]

pub mod classic;
pub mod extent;
pub mod netcdf;

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub mod async_netcdf;

// ─── the layer below ───────────────────────────────────────────────────────
//
// The HDF5 half lives in its own crate. These re-exports keep one dependency
// enough for a caller, and keep every path stable.

pub use oxcdf_hdf5;
pub use oxcdf_hdf5 as hdf5;
pub use oxcdf_hdf5::{cache, checksum, cursor, dtype, error, filters, index, io, read, source};

#[cfg(feature = "object-store")]
#[cfg_attr(docsrs, doc(cfg(feature = "object-store")))]
pub use oxcdf_hdf5::object_store_source;
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub use oxcdf_hdf5::{async_source, replay};

// ─── the types most callers need, at the crate root ────────────────────────

pub use oxcdf_hdf5::{
    detect_container, ByteSource, Container, DType, Element, Error, FileSource, Hyperslab,
    MemorySource, OpenOptions, Result, CDF_SIGNATURES, HDF5_SIGNATURE,
};

pub use extent::{Extent, Extents};
pub use netcdf::{AttributeValue, NcAttribute, NcDimension, NcVariable, NetcdfFile, Variable};

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub use async_netcdf::{AsyncNetcdfFile, AsyncVariable};
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub use oxcdf_hdf5::{AsyncByteSource, SyncAsAsync};

/// Open a netCDF file for reading.
///
/// The file may be netCDF-4 or netCDF classic. This is the synchronous entry
/// point. Use [`open_async`] for an asynchronous one.
///
/// ```no_run
/// let file = oxcdf::open("argo.nc")?;
/// let temp = file.variable("TEMP").unwrap();
/// let values = temp.get_values::<f64, _>(..)?;
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

/// Open a netCDF file over an asynchronous byte source.
///
/// The file may be netCDF-4 or netCDF classic. The open reads the metadata. The
/// returned file answers every metadata question without further input. A read
/// of values awaits.
///
/// ```no_run
/// # async fn run(source: std::sync::Arc<dyn oxcdf::AsyncByteSource>) -> oxcdf::Result<()> {
/// let file = oxcdf::open_async(source).await?;
/// let temp = file.variable("TEMP").unwrap();
/// let values = temp.get_values::<f64, _>(..).await?;
/// # Ok(()) }
/// ```
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub async fn open_async(source: std::sync::Arc<dyn AsyncByteSource>) -> Result<AsyncNetcdfFile> {
    AsyncNetcdfFile::open(source).await
}

/// Open a netCDF file over an asynchronous byte source, with explicit options.
///
/// Use [`OpenOptions::remote`] for object storage.
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub async fn open_async_with(
    source: std::sync::Arc<dyn AsyncByteSource>,
    options: OpenOptions,
) -> Result<AsyncNetcdfFile> {
    AsyncNetcdfFile::open_with(source, options).await
}
