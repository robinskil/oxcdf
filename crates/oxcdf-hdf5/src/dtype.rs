//! Element types, and the Rust types that read them.
//!
//! This sits below the netCDF layer on purpose. [`crate::read::RawData`]
//! decodes through [`Element`], so the decoder cannot depend on the layer above
//! it. The `oxcdf` crate re-exports both names.
//!
//! The vocabulary is netCDF's, because that is what a caller asks for. The
//! mapping onto HDF5 is in [`DType::of`].

use crate::message::DatatypeClass;

/// A variable's netCDF type.
///
/// This mirrors `netcdf::types::NcVariableType`. The full HDF5 datatype is
/// still available through [`DatasetIndex::datatype`].
///
/// [`DatasetIndex::datatype`]: crate::index::DatasetIndex::datatype
///
/// Not `Copy`: a ragged array carries its element type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType {
    /// Signed integer of the given width in bytes. netCDF `byte`, `short`,
    /// `int` and `int64`.
    Int(u8),
    /// Unsigned integer of the given width in bytes. netCDF `ubyte`, `ushort`,
    /// `uint` and `uint64`.
    Uint(u8),
    /// IEEE float of the given width in bytes. netCDF `float` and `double`.
    Float(u8),
    /// netCDF `char`. One byte for each element.
    ///
    /// A `char` variable holds text across its last dimension. So
    /// `char name(casts, strnlen)` holds one string for each cast. The last
    /// dimension is the string length.
    ///
    /// This reader reports the elements as the file stores them. Join that axis
    /// yourself.
    Char,
    /// netCDF `string`. One variable-length string for each element, held in
    /// the global heap.
    String,
    /// A fixed-length string wider than one byte.
    ///
    /// netcdf-c never writes this. Another HDF5 writer may. One string sits in
    /// each element, and the dataspace already excludes the width.
    FixedString(u32),
    /// A ragged array of a base type, held in the global heap.
    Vlen(Box<DType>),
    /// Something this summary does not model.
    Other,
}

impl DType {
    /// The netCDF type of an HDF5 datatype.
    pub fn of(datatype: &crate::message::Datatype) -> Self {
        let size = datatype.size;
        match &datatype.class {
            DatatypeClass::FixedPoint { signed: true, .. } => DType::Int(size as u8),
            DatatypeClass::FixedPoint { signed: false, .. } => DType::Uint(size as u8),
            DatatypeClass::FloatingPoint { .. } => DType::Float(size as u8),
            // netcdf-c writes `char` as a one-byte HDF5 string and puts the
            // length in the dataspace. Any wider fixed string is not netCDF.
            DatatypeClass::String { .. } if size == 1 => DType::Char,
            DatatypeClass::String { .. } => DType::FixedString(size),
            DatatypeClass::VariableLength {
                kind: crate::message::VlenKind::String,
                ..
            } => DType::String,
            DatatypeClass::VariableLength {
                kind: crate::message::VlenKind::Sequence,
                base,
                ..
            } => DType::Vlen(Box::new(DType::of(base))),
            _ => DType::Other,
        }
    }

    /// Whether this is an integer type. `char` is not one.
    pub fn is_integer(&self) -> bool {
        matches!(self, DType::Int(_) | DType::Uint(_))
    }

    /// Whether this is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(self, DType::Float(_))
    }

    /// Whether this type holds text.
    pub fn is_text(&self) -> bool {
        matches!(self, DType::Char | DType::String | DType::FixedString(_))
    }

    /// Size of one element in bytes, where the type has a fixed one.
    ///
    /// A [`DType::String`] has no fixed size: the value lives in a heap.
    pub fn size(&self) -> Option<usize> {
        match self {
            DType::Int(n) | DType::Uint(n) | DType::Float(n) => Some(*n as usize),
            DType::Char => Some(1),
            DType::FixedString(n) => Some(*n as usize),
            DType::String | DType::Vlen(_) | DType::Other => None,
        }
    }

    /// The netCDF name of this type.
    ///
    /// A numeric type also names the Rust type that reads it without a
    /// conversion. Pass that to [`RawData::get`].
    ///
    /// [`RawData::get`]: crate::read::RawData::get
    pub fn name(&self) -> String {
        match self {
            DType::Int(1) => "i8".into(),
            DType::Int(2) => "i16".into(),
            DType::Int(4) => "i32".into(),
            DType::Int(8) => "i64".into(),
            DType::Uint(1) => "u8".into(),
            DType::Uint(2) => "u16".into(),
            DType::Uint(4) => "u32".into(),
            DType::Uint(8) => "u64".into(),
            DType::Float(4) => "f32".into(),
            DType::Float(8) => "f64".into(),
            DType::Int(n) => format!("a {n}-byte signed integer"),
            DType::Uint(n) => format!("a {n}-byte unsigned integer"),
            DType::Float(n) => format!("a {n}-byte float"),
            DType::Char => "char".into(),
            DType::String => "string".into(),
            DType::FixedString(n) => format!("a {n}-byte fixed string"),
            DType::Vlen(base) => format!("a ragged array of {}", base.name()),
            DType::Other => "a type this reader does not model".into(),
        }
    }
}

/// A Rust type a variable can be read as.
///
/// | Rust | netCDF | HDF5 |
/// |---|---|---|
/// | `i8` `i16` `i32` `i64` | `byte` `short` `int` `int64` | signed fixed point |
/// | `u8` `u16` `u32` `u64` | `ubyte` `ushort` `uint` `uint64` | unsigned fixed point |
/// | `f32` `f64` | `float` `double` | floating point |
///
/// # Conversion
///
/// A read converts between any two numeric types, which is what the `netcdf`
/// crate does. A read of a string or a compound as a number fails with
/// [`crate::Error::TypeMismatch`].
///
/// A conversion can lose information. `f64` to `f32` loses precision. `i64` to
/// `f64` loses integers above 2^53. A float to an integer truncates toward
/// zero, and saturates at the limits of the target.
///
/// Call [`DType::of`] to learn the stored type, then ask for that type. The
/// read then copies the values and changes nothing.
///
/// This trait is sealed. Only the ten types above implement it.
pub trait Element: Copy + Sized + sealed::Sealed {
    /// The stored type that needs no conversion.
    const DTYPE: DType;
    /// The name used in an error message.
    const NAME: &'static str;
    /// Decode one element from native-order bytes of exactly this type.
    ///
    /// The read path normalises byte order, so the bytes are always native by
    /// the time they arrive here.
    fn from_ne_bytes(bytes: &[u8]) -> Self;
    /// Convert from a stored signed integer.
    fn from_i64(value: i64) -> Self;
    /// Convert from a stored unsigned integer.
    fn from_u64(value: u64) -> Self;
    /// Convert from a stored float.
    fn from_f64(value: f64) -> Self;
}

mod sealed {
    /// Stops a type outside this crate from claiming a stored type mapping.
    pub trait Sealed {}
}

macro_rules! element {
    ($rust:ty, $dtype:expr, $name:literal) => {
        impl sealed::Sealed for $rust {}
        impl Element for $rust {
            const DTYPE: DType = $dtype;
            const NAME: &'static str = $name;
            fn from_ne_bytes(bytes: &[u8]) -> Self {
                // The caller checked the width against `size_of`, so this
                // cannot fail.
                <$rust>::from_ne_bytes(bytes.try_into().expect("width checked"))
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
            #[allow(clippy::cast_sign_loss, clippy::cast_precision_loss)]
            fn from_i64(value: i64) -> Self {
                value as $rust
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_lossless)]
            #[allow(clippy::cast_sign_loss, clippy::cast_precision_loss)]
            fn from_u64(value: u64) -> Self {
                value as $rust
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            #[allow(clippy::cast_precision_loss)]
            fn from_f64(value: f64) -> Self {
                value as $rust
            }
        }
    };
}

element!(i8, DType::Int(1), "i8");
element!(i16, DType::Int(2), "i16");
element!(i32, DType::Int(4), "i32");
element!(i64, DType::Int(8), "i64");
element!(u8, DType::Uint(1), "u8");
element!(u16, DType::Uint(2), "u16");
element!(u32, DType::Uint(4), "u32");
element!(u64, DType::Uint(8), "u64");
element!(f32, DType::Float(4), "f32");
element!(f64, DType::Float(8), "f64");
