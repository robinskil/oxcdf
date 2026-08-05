//! Error type for the native reader.

use std::fmt;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Everything that can go wrong while reading a netCDF-4 / HDF5 file.
///
/// [`Error::Unsupported`] is deliberately distinct from the other variants. A
/// caller that keeps netcdf-c as a fallback matches on it to decide whether to
/// retry through the C library. Every other variant means the file is damaged
/// or this reader has a bug, and a fallback would only hide the problem.
#[derive(Debug)]
pub enum Error {
    /// The underlying storage failed.
    Io(std::io::Error),
    /// The file is not HDF5, or a structure is malformed.
    Malformed(String),
    /// A read ran past the end of a buffer or of the file.
    OutOfBounds {
        /// What the reader tried to access.
        what: &'static str,
        /// Byte offset of the attempted access.
        offset: u64,
        /// Number of bytes requested.
        len: u64,
        /// Size of the region that was available.
        available: u64,
    },
    /// A valid HDF5 feature that this reader does not implement yet.
    ///
    /// Callers may fall back to netcdf-c on this variant.
    Unsupported(String),
    /// A checksum stored in the file did not match the bytes read.
    ChecksumMismatch {
        /// Which structure failed the check.
        what: &'static str,
        /// Checksum recorded in the file.
        stored: u32,
        /// Checksum computed over the bytes read.
        computed: u32,
    },
    /// The caller asked for something the file does not contain.
    NotFound(String),
    /// The caller asked for a Rust type that the variable does not store.
    ///
    /// The reader never converts one numeric type to another. It returns the
    /// stored type or it fails. A conversion would hide a precision loss, and
    /// only the caller knows whether that loss is acceptable.
    ///
    /// Read [`Error::TypeMismatch::stored`] and ask for that type instead.
    TypeMismatch {
        /// The type the variable stores.
        stored: String,
        /// The Rust type the caller asked for.
        asked: &'static str,
        /// Which variable, when the reader knows.
        what: String,
    },
    /// The caller passed an invalid request, such as an out-of-range hyperslab.
    BadRequest(String),
    /// The byte source does not hold the requested bytes yet.
    ///
    /// The asynchronous engine produces this. It runs a synchronous walk over
    /// the bytes it holds. A walk that needs more bytes stops with this error.
    /// The engine fetches the missing bytes. The engine then runs the walk
    /// again. See the `replay` module, behind the `async` feature.
    ///
    /// A public asynchronous method never returns this. It surfaces only from a
    /// synchronous read on a file that an asynchronous open produced.
    Incomplete,
}

impl Error {
    /// Build an [`Error::Malformed`] from anything printable.
    pub fn malformed(msg: impl fmt::Display) -> Self {
        Error::Malformed(msg.to_string())
    }

    /// Build an [`Error::Unsupported`] from anything printable.
    pub fn unsupported(msg: impl fmt::Display) -> Self {
        Error::Unsupported(msg.to_string())
    }

    /// Build an [`Error::NotFound`] from anything printable.
    pub fn not_found(msg: impl fmt::Display) -> Self {
        Error::NotFound(msg.to_string())
    }

    /// Build an [`Error::BadRequest`] from anything printable.
    pub fn bad_request(msg: impl fmt::Display) -> Self {
        Error::BadRequest(msg.to_string())
    }

    /// Whether a caller should retry this read through netcdf-c.
    ///
    /// Only unsupported features qualify. A malformed file or a checksum
    /// failure is a real defect and must surface.
    pub fn is_fallback_worthy(&self) -> bool {
        matches!(self, Error::Unsupported(_))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Malformed(m) => write!(f, "malformed HDF5: {m}"),
            Error::OutOfBounds {
                what,
                offset,
                len,
                available,
            } => write!(
                f,
                "{what} read out of bounds: wanted {len} bytes at offset {offset}, \
                 but only {available} bytes are available"
            ),
            Error::Unsupported(m) => write!(f, "unsupported HDF5 feature: {m}"),
            Error::ChecksumMismatch {
                what,
                stored,
                computed,
            } => write!(
                f,
                "{what} checksum mismatch: file stores {stored:#010x}, computed {computed:#010x}"
            ),
            Error::NotFound(m) => write!(f, "not found: {m}"),
            Error::TypeMismatch {
                stored,
                asked,
                what,
            } => write!(
                f,
                "type mismatch{}: the variable stores {stored}, but {asked} was asked for; \
                 this reader does not convert between numeric types",
                if what.is_empty() {
                    String::new()
                } else {
                    format!(" for {what}")
                }
            ),
            Error::BadRequest(m) => write!(f, "bad request: {m}"),
            Error::Incomplete => write!(
                f,
                "the bytes are not in memory: an asynchronous open produced this file, \
                 so use the asynchronous read methods"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<std::convert::Infallible> for Error {
    /// A selection that converts without failing never reaches this.
    ///
    /// A read takes `E: TryInto<Extents>`. A type with a plain `From` gets
    /// `TryFrom` for free, and its error type is [`std::convert::Infallible`].
    /// This makes such a type usable.
    fn from(e: std::convert::Infallible) -> Self {
        match e {}
    }
}
