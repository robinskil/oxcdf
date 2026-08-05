//! A pure-Rust, parallel-safe reader for the HDF5 container format.
//!
//! This crate reads the file as HDF5. It knows nothing about netCDF. It calls
//! no C library, so it holds no global mutex. Many threads read one file at the
//! same time.
//!
//! For netCDF-4 and netCDF classic files, use [`oxcdf`]. That crate applies the
//! netCDF conventions on top of this one.
//!
//! [`oxcdf`]: https://docs.rs/oxcdf
//!
//! # Example
//!
//! ```no_run
//! use oxcdf_hdf5::index::Hdf5File;
//! use oxcdf_hdf5::read::{read_hyperslab, Hyperslab};
//!
//! let file = Hdf5File::open("data.h5")?;
//! let dataset = file.dataset("/temperature").unwrap();
//! dataset.prepare(file.ctx())?;
//!
//! let slab = Hyperslab::all(&dataset.shape);
//! let values = read_hyperslab(file.ctx(), dataset, &slab)?.get::<f64>(dataset)?;
//! # Ok::<(), oxcdf_hdf5::Error>(())
//! ```
//!
//! `AsyncHdf5File` is the asynchronous form, behind the `async` feature.
//! Only the reads await.
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
//! Both engines share every pure part. Only the fetch differs. The decode stays
//! synchronous, because decompression uses the processor.
//!
//! The crate holds one parser, not two. An asynchronous open runs the
//! synchronous walk over pages in memory, then fetches what the walk asks for
//! and runs it again. See the `replay` module.
//!
//! # Layout
//!
//! * [`index`] holds the parsed view of a file. Start here.
//! * [`read`] turns a dataset and a selection into bytes.
//! * `async_hdf5` is the same pair, asynchronous.
//! * [`superblock`], [`objheader`] and [`message`] parse the container.
//! * [`btree1`], [`btree2`], [`fractal`], [`heap`] and [`chunk_index`] parse
//!   the structures those messages point at.
//!
//! # Scope
//!
//! The reader targets the subset of HDF5 that netcdf-c writes.
//!
//! A feature outside that subset returns [`Error::Unsupported`]. Match on
//! [`Error::is_fallback_worthy`] to send one dataset to another library. Every
//! other error marks a damaged file or a defect here.
//!
//! The crate does not write files.

// docs.rs builds with `--cfg docsrs` on nightly, so every feature-gated item
// carries a badge that names the feature. A stable build ignores this.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]
#![deny(rust_2018_idioms)]

// ─── the container ─────────────────────────────────────────────────────────

pub mod btree1;
pub mod btree2;
pub mod chunk_index;
pub mod container;
pub mod context;
pub mod dense;
pub mod fractal;
pub mod heap;
pub mod message;
pub mod objheader;
pub mod superblock;
pub mod symbol_table;

// ─── the reader ────────────────────────────────────────────────────────────

pub mod cache;
pub mod checksum;
pub mod cursor;
pub mod dtype;
pub mod error;
pub mod filters;
pub mod index;
pub mod io;
pub mod read;
pub mod source;

// ─── the asynchronous engine ───────────────────────────────────────────────

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub mod async_hdf5;
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub mod async_source;
#[cfg(feature = "object-store")]
#[cfg_attr(docsrs, doc(cfg(feature = "object-store")))]
pub mod object_store_source;
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub mod replay;

// ─── the names most callers need ───────────────────────────────────────────

pub use container::{detect_container, Container, CDF_SIGNATURES, HDF5_SIGNATURE};
pub use context::Ctx;
pub use dtype::{DType, Element};
pub use error::{Error, Result};
pub use index::{DatasetIndex, GroupIndex, Hdf5File, OpenOptions};
pub use objheader::{HeaderMessage, MessageType, ObjectHeader};
pub use read::{Chunk, Hyperslab, RawData};
pub use source::{ByteSource, FileSource, MemorySource};
pub use superblock::{RootGroup, Superblock};
pub use symbol_table::{CachedInfo, SymbolTableEntry, SymbolTableNode};

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub use async_hdf5::{AsyncDataset, AsyncHdf5File};
#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub use async_source::{AsyncByteSource, SyncAsAsync};

/// The corpus directory, which sits at the workspace root.
///
/// Both crates read the same files, so the files are not copied into either.
#[cfg(test)]
pub(crate) fn test_files_dir() -> String {
    format!("{}/../../test_files", env!("CARGO_MANIFEST_DIR"))
}

#[cfg(test)]
pub(crate) mod test_corpus {
    /// Absolute paths to the netCDF-4 files checked into this repository.
    ///
    /// They cover the feature matrix this reader targets: superblock v0 and v2,
    /// chunked with shuffle+deflate, contiguous, big- and little-endian floats,
    /// and fixed-length strings.
    pub fn paths() -> Vec<String> {
        ["test_file.nc", "gridded-example.nc", "wod_ctd_1964.nc"]
            .iter()
            .map(|p| format!("{}/{p}", crate::test_files_dir()))
            .filter(|p| std::path::Path::new(p).exists())
            .collect()
    }
}
