//! Parsers for the HDF5 container format.
//!
//! The modules here read the file as HDF5 and know nothing about netCDF. The
//! netCDF conventions that sit on top (dimension scales, hidden variables) live
//! in a separate layer so this half stays a plain HDF5 reader.

pub mod btree1;
pub mod btree2;
pub mod chunk_index;
pub mod context;
pub mod dense;
pub mod fractal;
pub mod heap;
pub mod message;
pub mod objheader;
pub mod superblock;
pub mod symbol_table;

pub use context::Ctx;
pub use objheader::{HeaderMessage, MessageType, ObjectHeader};
pub use superblock::{RootGroup, Superblock};
pub use symbol_table::{CachedInfo, SymbolTableEntry, SymbolTableNode};
