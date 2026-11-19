//! The HDF5 writer. It builds a new HDF5 file.
//!
//! [`Hdf5Writer`] holds the tree of objects in memory. [`Hdf5Writer::close`]
//! writes the metadata and the superblock to the sink. Dataset values go to
//! the sink at each write.
//!
//! The sink holds a valid file only after [`Hdf5Writer::close`] returns `Ok`.
//! A writer that is dropped leaves no superblock. [`FileSink`] writes to a
//! file and [`MemorySink`] writes to memory.
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
//! datasets of numbers, fixed strings and variable-length strings. A dataset
//! is contiguous or chunked. A chunked dataset can have unlimited dimensions.
//! A chunked dataset can have shuffle, deflate, Zstandard and Fletcher-32
//! filters. Links and attributes are compact messages in the object header.
//!
//! [`Hdf5Writer::attach_scale`] writes the two attributes of a dimension
//! scale. The `oxcdf` crate uses it for the netCDF conventions.
//!
//! [`Hdf5Writer::open`] opens a file that exists for edits. The values stay
//! in place. The close writes each object header again at the end of the
//! file, so each edit session adds the size of all headers to the file.
//!
//! The writer does not write datasets of variable-length sequences.
//! It does not apply the netCDF conventions. It does not remove or rename a
//! group, and it does not remove a dataset or an attribute.

pub(crate) mod alloc;
pub(crate) mod btree;
mod dataset;
pub(crate) mod encode;
pub(crate) mod filter;
pub(crate) mod gheap;
pub(crate) mod header;
pub(crate) mod message;
mod sink;
mod writer;

#[doc(inline)]
pub use dataset::DatasetSpec;
#[doc(inline)]
pub use filter::Filter;
#[doc(inline)]
pub use message::{numeric_datatype, vlen_string_datatype};
#[doc(inline)]
pub use sink::{ByteSink, FileSink, MemorySink};
#[doc(inline)]
pub use writer::{DatasetId, GroupId, Hdf5Writer, ObjectId, WriteOptions};

use crate::error::{Error, Result};

/// The error for a state that only a defect in the writer can cause.
pub(crate) fn defect(what: impl std::fmt::Display) -> Error {
    Error::malformed(format!("a defect in the writer: {what}"))
}

/// Convert a count or length to `u64`, or return an error when it does not fit.
fn to_u64(n: usize) -> Result<u64> {
    u64::try_from(n).map_err(|_| Error::unsupported(format!("{n} does not fit in 64 bits")))
}
