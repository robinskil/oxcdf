//! The HDF5 writer. It builds a new HDF5 file.
//!
//! [`Hdf5Writer`] holds the tree of objects in memory. [`Hdf5Writer::close`]
//! writes the metadata and the superblock to the sink. Dataset values go to
//! the sink at each write.
//!
//! # Example
//!
//! ```no_run
//! use oxcdf_hdf5::message::ByteOrder;
//! use oxcdf_hdf5::write::{DatasetSpec, FileSink, Hdf5Writer, WriteOptions};
//! use oxcdf_hdf5::{DType, Hyperslab};
//!
//! let mut w = Hdf5Writer::create(FileSink::create("out.h5")?, WriteOptions::default())?;
//! let group = w.add_group(w.root(), "ocean")?;
//! w.put_text_attribute(group, "title", "sea surface")?;
//!
//! let spec = DatasetSpec::numeric(DType::Float(8), ByteOrder::Little, &[2, 3])?;
//! let temp = w.add_dataset(group, "temperature", spec)?;
//! w.write(temp, &Hyperslab::all(&[2, 3]), &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
//! w.put_text_attribute(temp, "units", "K")?;
//!
//! w.close()?;
//! # Ok::<(), oxcdf_hdf5::Error>(())
//! ```
//!
//! # Scope
//!
//! The writer creates a new file. It supports groups, attributes and
//! contiguous numeric datasets. Links and attributes are compact messages in
//! the object header.
//!
//! The writer does not write chunks, filters or strings in the global heap.
//! It does not apply the netCDF conventions. It does not change an existing
//! file.

pub(crate) mod alloc;
mod dataset;
pub(crate) mod encode;
pub(crate) mod header;
pub(crate) mod message;
mod sink;
mod writer;

pub use dataset::DatasetSpec;
pub use message::numeric_datatype;
pub use sink::{ByteSink, FileSink, MemorySink};
pub use writer::{DatasetId, GroupId, Hdf5Writer, ObjectId, WriteOptions};
