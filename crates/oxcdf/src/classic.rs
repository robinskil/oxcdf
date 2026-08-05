//! The netCDF classic formats: CDF-1, CDF-2 and CDF-5.
//!
//! These formats have nothing to do with HDF5. A file holds a header of
//! dimensions, attributes and variable descriptors. The data follows at
//! computed offsets. There is no compression, no chunking and no index.
//!
//! A read is therefore parallel-safe. Once the reader parses the header, every
//! byte offset is arithmetic, so a read is one byte range.
//!
//! Two layout rules matter:
//!
//! * A fixed-size variable sits in one contiguous run, in header order.
//! * Record variables interleave. Every record variable contributes one record,
//!   in header order. The next record then follows. A record variable's
//!   elements are therefore strided. The stride is the sum of the per-record
//!   size of every record variable.
//!
//! All values are big-endian. Every field pads to a multiple of four bytes.

use std::sync::Arc;

use oxcdf_hdf5::cursor::Cursor;
use oxcdf_hdf5::error::{Error, Result};
use oxcdf_hdf5::read::Hyperslab;
use oxcdf_hdf5::source::{ByteSource, FileSource};
use oxcdf_hdf5::Container;

/// Tag marking the end of a list in the header.
const NC_DIMENSION: u32 = 0x0A;
const NC_VARIABLE: u32 = 0x0B;
const NC_ATTRIBUTE: u32 = 0x0C;

/// The element types the classic format defines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcType {
    /// 8-bit signed integer.
    Byte,
    /// 8-bit character.
    Char,
    /// 16-bit signed integer.
    Short,
    /// 32-bit signed integer.
    Int,
    /// 32-bit float.
    Float,
    /// 64-bit float.
    Double,
    /// 8-bit unsigned integer. CDF-5 only.
    UByte,
    /// 16-bit unsigned integer. CDF-5 only.
    UShort,
    /// 32-bit unsigned integer. CDF-5 only.
    UInt,
    /// 64-bit signed integer. CDF-5 only.
    Int64,
    /// 64-bit unsigned integer. CDF-5 only.
    UInt64,
}

impl NcType {
    fn from_code(code: u32) -> Result<Self> {
        Ok(match code {
            1 => NcType::Byte,
            2 => NcType::Char,
            3 => NcType::Short,
            4 => NcType::Int,
            5 => NcType::Float,
            6 => NcType::Double,
            7 => NcType::UByte,
            8 => NcType::UShort,
            9 => NcType::UInt,
            10 => NcType::Int64,
            11 => NcType::UInt64,
            other => return Err(Error::malformed(format!("classic type code {other}"))),
        })
    }

    /// Width of one element in bytes.
    pub fn size(&self) -> usize {
        match self {
            NcType::Byte | NcType::Char | NcType::UByte => 1,
            NcType::Short | NcType::UShort => 2,
            NcType::Int | NcType::UInt | NcType::Float => 4,
            NcType::Double | NcType::Int64 | NcType::UInt64 => 8,
        }
    }

    /// Whether values of this type are signed integers.
    pub fn is_signed_integer(&self) -> bool {
        matches!(
            self,
            NcType::Byte | NcType::Short | NcType::Int | NcType::Int64
        )
    }

    /// Whether values of this type are integers of any signedness.
    /// The crate's canonical element descriptor for this type.
    ///
    /// The netCDF layer decodes every value through
    /// [`oxcdf_hdf5::message::Datatype`], whatever container holds it. A
    /// classic file therefore describes its types the same way an HDF5 file
    /// does, and one decode path serves both.
    ///
    /// Classic files store values big-endian, and the order says so.
    pub fn to_datatype(&self) -> oxcdf_hdf5::message::Datatype {
        use oxcdf_hdf5::message::{ByteOrder, CharSet, DatatypeClass, StringPad};

        let size = self.size() as u32;
        let class = match self {
            // netCDF `char` is a one-byte string, exactly as netcdf-c writes it
            // into HDF5. The last dimension carries the string length.
            NcType::Char => DatatypeClass::String {
                pad: StringPad::NullTerminate,
                charset: CharSet::Ascii,
            },
            NcType::Float | NcType::Double => DatatypeClass::FloatingPoint {
                order: ByteOrder::Big,
                bit_offset: 0,
                bit_precision: (size * 8) as u16,
                exponent_location: if size == 4 { 23 } else { 52 },
                exponent_size: if size == 4 { 8 } else { 11 },
                mantissa_location: 0,
                mantissa_size: if size == 4 { 23 } else { 52 },
                exponent_bias: if size == 4 { 127 } else { 1023 },
                sign_location: (size * 8 - 1) as u8,
            },
            other => DatatypeClass::FixedPoint {
                order: ByteOrder::Big,
                signed: other.is_signed_integer(),
                bit_offset: 0,
                bit_precision: (size * 8) as u16,
            },
        };

        oxcdf_hdf5::message::Datatype {
            version: 1,
            size,
            class,
        }
    }

    /// Whether this is an integer type. `char` is not one.
    pub fn is_integer(&self) -> bool {
        !matches!(self, NcType::Float | NcType::Double | NcType::Char)
    }
}

/// One dimension.
#[derive(Debug, Clone)]
pub struct ClassicDimension {
    /// Dimension name.
    pub name: String,
    /// Length. Zero on disk marks the record dimension; this holds the actual
    /// record count.
    pub len: u64,
    /// Whether this is the unlimited (record) dimension.
    pub is_unlimited: bool,
}

/// One attribute, with its values decoded.
#[derive(Debug, Clone)]
pub struct ClassicAttribute {
    /// Attribute name.
    pub name: String,
    /// Element type.
    pub nc_type: NcType,
    /// Text value, for `char` attributes.
    pub text: Option<String>,
    /// Numeric values, widened to `f64`.
    ///
    /// The netCDF layer does not use this. It decodes [`ClassicAttribute::raw`]
    /// into a typed value instead.
    pub numbers: Vec<f64>,
    /// The values as stored, still big-endian.
    ///
    /// The netCDF layer decodes these through the same path an HDF5 attribute
    /// takes. The byte order lives in the datatype, so one decoder serves both.
    pub raw: Vec<u8>,
}

/// One variable.
#[derive(Debug, Clone)]
pub struct ClassicVariable {
    /// Variable name.
    pub name: String,
    /// Names of its dimensions, in order.
    pub dimensions: Vec<String>,
    /// Shape in elements.
    pub shape: Vec<u64>,
    /// Element type.
    pub nc_type: NcType,
    /// The same type as the crate's canonical descriptor.
    ///
    /// The netCDF layer reads through this, so one decode path serves a classic
    /// file and an HDF5 file alike.
    pub datatype: oxcdf_hdf5::message::Datatype,
    /// The variable's attributes.
    pub attributes: Vec<ClassicAttribute>,
    /// Byte offset of the variable's first element.
    pub begin: u64,
    /// Size on disk of one record, padded, for a record variable.
    pub record_size: u64,
    /// Whether this variable uses the record dimension.
    pub is_record: bool,
}

impl ClassicVariable {
    /// Total number of elements.
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }

    /// An attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&ClassicAttribute> {
        self.attributes.iter().find(|a| a.name == name)
    }
}

/// An open netCDF classic file.
///
/// Like the HDF5 side, the header is parsed once into an immutable index and
/// reads are pure functions over it, so this type is `Send + Sync`.
#[derive(Debug, Clone)]
pub struct ClassicFile {
    source: Arc<dyn ByteSource>,
    /// Which classic variant this is.
    pub container: Container,
    /// The file's dimensions.
    pub dimensions: Vec<ClassicDimension>,
    /// The file's global attributes.
    pub attributes: Vec<ClassicAttribute>,
    /// The file's variables.
    pub variables: Vec<ClassicVariable>,
    /// Number of records written.
    pub record_count: u64,
    /// Total size of one record across every record variable.
    pub record_stride: u64,
}

impl ClassicFile {
    /// Open a classic file from the filesystem.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_source(Arc::new(FileSource::open(path)?))
    }

    /// Open a classic file over any byte source.
    pub fn from_source(source: Arc<dyn ByteSource>) -> Result<Self> {
        let mut magic = [0u8; 4];
        source.read_exact_at(0, &mut magic)?;
        let container = match magic {
            [b'C', b'D', b'F', 1] => Container::Cdf1,
            [b'C', b'D', b'F', 2] => Container::Cdf2,
            [b'C', b'D', b'F', 5] => Container::Cdf5,
            _ => return Err(Error::malformed("not a netCDF classic file")),
        };

        // The header sits at the front and is small; read a generous window and
        // grow it if the header turns out to be longer.
        let mut window = (source.size().min(1 << 16)) as usize;
        loop {
            let buf = source.read_vec(0, window)?;
            match Header::parse(&buf, container) {
                Ok(header) => {
                    return Ok(Self {
                        source,
                        container,
                        dimensions: header.dimensions,
                        attributes: header.attributes,
                        variables: header.variables,
                        record_count: header.record_count,
                        record_stride: header.record_stride,
                    })
                }
                Err(Error::OutOfBounds { .. }) if (window as u64) < source.size() => {
                    window = (window * 4).min(source.size() as usize);
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// A variable by name.
    pub fn variable(&self, name: &str) -> Option<&ClassicVariable> {
        self.variables.iter().find(|v| v.name == name)
    }

    /// A dimension by name.
    pub fn dimension(&self, name: &str) -> Option<&ClassicDimension> {
        self.dimensions.iter().find(|d| d.name == name)
    }

    /// A global attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&ClassicAttribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Read a whole variable, returning native-order bytes in row-major order.
    pub fn read(&self, variable: &ClassicVariable) -> Result<Vec<u8>> {
        self.read_selection(variable, &Hyperslab::all(&variable.shape))
    }

    /// Read a hyperslab of a variable.
    pub fn read_selection(&self, variable: &ClassicVariable, slab: &Hyperslab) -> Result<Vec<u8>> {
        self.read_selection_with(self.source.as_ref(), variable, slab)
    }

    /// Read a selection through another byte source.
    ///
    /// The parsed header stays. Only the bytes come from somewhere else. The
    /// asynchronous engine uses this to serve a read from pages it holds.
    pub fn read_selection_with(
        &self,
        source: &dyn ByteSource,
        variable: &ClassicVariable,
        slab: &Hyperslab,
    ) -> Result<Vec<u8>> {
        slab.validate(&variable.shape)?;
        let element_size = variable.nc_type.size();
        let total = (slab.element_count() as usize)
            .checked_mul(element_size)
            .ok_or_else(|| Error::bad_request("selection is too large"))?;
        let mut out = vec![0u8; total];

        if variable.is_record {
            self.read_record_variable(source, variable, slab, &mut out, element_size)?;
        } else {
            self.read_fixed_variable(source, variable, slab, &mut out, element_size)?;
        }

        // Classic files are always big-endian.
        if element_size > 1 {
            for chunk in out.chunks_exact_mut(element_size) {
                chunk.reverse();
            }
        }
        Ok(out)
    }

    /// A fixed-size variable is one contiguous run.
    fn read_fixed_variable(
        &self,
        source: &dyn ByteSource,
        variable: &ClassicVariable,
        slab: &Hyperslab,
        out: &mut [u8],
        element_size: usize,
    ) -> Result<()> {
        let strides = row_major_strides(&variable.shape);
        let dst_strides = row_major_strides(&slab.count);
        let rank = variable.shape.len();

        if rank == 0 {
            let bytes = source.read_vec(variable.begin, element_size)?;
            out[..element_size].copy_from_slice(&bytes);
            return Ok(());
        }

        let run = slab.count[rank - 1];
        let outer: u64 = slab.count[..rank - 1].iter().product();
        let mut index = vec![0u64; rank - 1];

        for _ in 0..outer.max(if rank == 1 { 1 } else { 0 }) {
            let mut src = 0u64;
            let mut dst = 0u64;
            for axis in 0..rank - 1 {
                src += (slab.start[axis] + index[axis]) * strides[axis];
                dst += index[axis] * dst_strides[axis];
            }
            src += slab.start[rank - 1];

            let len = run as usize * element_size;
            let bytes = self
                .source
                .read_vec(variable.begin + src * element_size as u64, len)?;
            let db = dst as usize * element_size;
            out[db..db + len].copy_from_slice(&bytes);

            for axis in (0..rank.saturating_sub(1)).rev() {
                index[axis] += 1;
                if index[axis] < slab.count[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
        Ok(())
    }

    /// A record variable is strided: one record's worth, then a gap covering
    /// every other record variable, then the next record.
    fn read_record_variable(
        &self,
        source: &dyn ByteSource,
        variable: &ClassicVariable,
        slab: &Hyperslab,
        out: &mut [u8],
        element_size: usize,
    ) -> Result<()> {
        let rank = variable.shape.len();
        // Shape within one record, dropping the leading record axis.
        let inner_shape: Vec<u64> = variable.shape[1..].to_vec();
        let inner_strides = row_major_strides(&inner_shape);
        let dst_strides = row_major_strides(&slab.count);

        let inner_rank = inner_shape.len();
        let run = if rank == 1 { 1 } else { slab.count[rank - 1] };
        let inner_outer: u64 = if inner_rank <= 1 {
            1
        } else {
            slab.count[1..rank - 1].iter().product()
        };

        for record in 0..slab.count[0] {
            let record_index = slab.start[0] + record;
            let record_base = variable.begin + record_index * self.record_stride;

            let mut index = vec![0u64; inner_rank.saturating_sub(1)];
            for _ in 0..inner_outer {
                let mut src = 0u64;
                let mut dst = record * dst_strides[0];
                for axis in 0..inner_rank.saturating_sub(1) {
                    src += (slab.start[axis + 1] + index[axis]) * inner_strides[axis];
                    dst += index[axis] * dst_strides[axis + 1];
                }
                if inner_rank > 0 {
                    src += slab.start[rank - 1];
                    dst += 0;
                }

                let len = run as usize * element_size;
                let bytes = source.read_vec(record_base + src * element_size as u64, len)?;
                let db = dst as usize * element_size;
                out[db..db + len].copy_from_slice(&bytes);

                for axis in (0..inner_rank.saturating_sub(1)).rev() {
                    index[axis] += 1;
                    if index[axis] < slab.count[axis + 1] {
                        break;
                    }
                    index[axis] = 0;
                }
            }
        }
        Ok(())
    }

    /// Read a variable and widen it to `f64`.
    pub fn read_f64(&self, variable: &ClassicVariable) -> Result<Vec<f64>> {
        let bytes = self.read(variable)?;
        decode_numbers(&bytes, variable.nc_type)
    }

    /// Read a `char` variable as one string per row of the last axis.
    pub fn read_strings(&self, variable: &ClassicVariable) -> Result<Vec<String>> {
        if variable.nc_type != NcType::Char {
            return Err(Error::unsupported(format!(
                "variable {} is not a character variable",
                variable.name
            )));
        }
        let bytes = self.read(variable)?;
        let width = *variable.shape.last().unwrap_or(&1) as usize;
        if width == 0 {
            return Ok(Vec::new());
        }
        Ok(bytes
            .chunks(width)
            .map(|row| {
                let end = row.iter().position(|&b| b == 0).unwrap_or(row.len());
                let trimmed = row[..end]
                    .iter()
                    .rposition(|&b| b != b' ')
                    .map(|p| p + 1)
                    .unwrap_or(0);
                String::from_utf8_lossy(&row[..trimmed]).into_owned()
            })
            .collect())
    }
}

/// Widen big-endian-decoded native bytes to `f64`.
fn decode_numbers(bytes: &[u8], nc_type: NcType) -> Result<Vec<f64>> {
    let size = nc_type.size();
    let count = bytes.len() / size;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = &bytes[i * size..(i + 1) * size];
        out.push(match nc_type {
            NcType::Float => f32::from_ne_bytes(b.try_into().unwrap()) as f64,
            NcType::Double => f64::from_ne_bytes(b.try_into().unwrap()),
            NcType::Byte => i8::from_ne_bytes(b.try_into().unwrap()) as f64,
            NcType::UByte | NcType::Char => b[0] as f64,
            NcType::Short => i16::from_ne_bytes(b.try_into().unwrap()) as f64,
            NcType::UShort => u16::from_ne_bytes(b.try_into().unwrap()) as f64,
            NcType::Int => i32::from_ne_bytes(b.try_into().unwrap()) as f64,
            NcType::UInt => u32::from_ne_bytes(b.try_into().unwrap()) as f64,
            NcType::Int64 => i64::from_ne_bytes(b.try_into().unwrap()) as f64,
            NcType::UInt64 => u64::from_ne_bytes(b.try_into().unwrap()) as f64,
        });
    }
    Ok(out)
}

fn row_major_strides(shape: &[u64]) -> Vec<u64> {
    let mut out = vec![1u64; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        out[axis] = out[axis + 1] * shape[axis + 1];
    }
    out
}

/// The parsed header.
struct Header {
    dimensions: Vec<ClassicDimension>,
    attributes: Vec<ClassicAttribute>,
    variables: Vec<ClassicVariable>,
    record_count: u64,
    record_stride: u64,
}

impl Header {
    fn parse(buf: &[u8], container: Container) -> Result<Self> {
        let mut cur = Cursor::new(buf);
        cur.skip(4)?; // magic

        // Classic headers are big-endian throughout.
        let record_count = be_u32(&mut cur)? as u64;

        let is_cdf5 = container == Container::Cdf5;
        let read_size = |cur: &mut Cursor<'_>| -> Result<u64> {
            if is_cdf5 {
                be_u64(cur)
            } else {
                Ok(be_u32(cur)? as u64)
            }
        };
        let read_offset = |cur: &mut Cursor<'_>| -> Result<u64> {
            if container == Container::Cdf1 {
                Ok(be_u32(cur)? as u64)
            } else {
                be_u64(cur)
            }
        };

        // ── dimensions ────────────────────────────────────────────────────
        let mut dimensions = Vec::new();
        let tag = be_u32(&mut cur)?;
        let count = read_size(&mut cur)? as usize;
        if tag == NC_DIMENSION {
            for _ in 0..count {
                let name = read_name(&mut cur, is_cdf5)?;
                let len = read_size(&mut cur)?;
                dimensions.push(ClassicDimension {
                    name,
                    len,
                    is_unlimited: len == 0,
                });
            }
        } else if tag != 0 {
            return Err(Error::malformed(format!(
                "expected a dimension list, found tag {tag}"
            )));
        }

        // The record dimension's real length comes from the header's record
        // count, not from the zero stored in its entry.
        for d in dimensions.iter_mut() {
            if d.is_unlimited {
                d.len = record_count;
            }
        }

        // ── global attributes ─────────────────────────────────────────────
        let attributes = read_attributes(&mut cur, is_cdf5, &read_size)?;

        // ── variables ─────────────────────────────────────────────────────
        let mut variables = Vec::new();
        let tag = be_u32(&mut cur)?;
        let count = read_size(&mut cur)? as usize;
        if tag == NC_VARIABLE {
            for _ in 0..count {
                let name = read_name(&mut cur, is_cdf5)?;
                let rank = read_size(&mut cur)? as usize;
                let mut dim_ids = Vec::with_capacity(rank);
                for _ in 0..rank {
                    dim_ids.push(read_size(&mut cur)? as usize);
                }
                let attributes = read_attributes(&mut cur, is_cdf5, &read_size)?;
                let nc_type = NcType::from_code(be_u32(&mut cur)?)?;
                let _vsize = read_size(&mut cur)?;
                let begin = read_offset(&mut cur)?;

                let mut shape = Vec::with_capacity(rank);
                let mut dimension_names = Vec::with_capacity(rank);
                for &id in &dim_ids {
                    let d = dimensions.get(id).ok_or_else(|| {
                        Error::malformed(format!(
                            "variable {name} names dimension {id}, which does not exist"
                        ))
                    })?;
                    shape.push(d.len);
                    dimension_names.push(d.name.clone());
                }

                let is_record = dim_ids
                    .first()
                    .is_some_and(|&id| dimensions.get(id).is_some_and(|d| d.is_unlimited));

                // One record's worth of this variable, padded to four bytes.
                let per_record: u64 = if is_record {
                    let inner: u64 = shape[1..].iter().product();
                    pad4(inner * nc_type.size() as u64)
                } else {
                    0
                };

                variables.push(ClassicVariable {
                    name,
                    dimensions: dimension_names,
                    shape,
                    nc_type,
                    datatype: nc_type.to_datatype(),
                    attributes,
                    begin,
                    record_size: per_record,
                    is_record,
                });
            }
        } else if tag != 0 {
            return Err(Error::malformed(format!(
                "expected a variable list, found tag {tag}"
            )));
        }

        // A record variable's stride is the sum over every record variable.
        // The one exception: with exactly one record variable there is no
        // padding between records.
        let record_vars: Vec<&ClassicVariable> = variables.iter().filter(|v| v.is_record).collect();
        let record_stride = if record_vars.len() == 1 {
            let v = record_vars[0];
            let inner: u64 = v.shape[1..].iter().product();
            inner * v.nc_type.size() as u64
        } else {
            record_vars.iter().map(|v| v.record_size).sum()
        };

        Ok(Self {
            dimensions,
            attributes,
            variables,
            record_count,
            record_stride,
        })
    }
}

fn read_attributes(
    cur: &mut Cursor<'_>,
    is_cdf5: bool,
    read_size: &dyn Fn(&mut Cursor<'_>) -> Result<u64>,
) -> Result<Vec<ClassicAttribute>> {
    let tag = be_u32(cur)?;
    let count = read_size(cur)? as usize;
    if tag != NC_ATTRIBUTE {
        if tag != 0 {
            return Err(Error::malformed(format!(
                "expected an attribute list, found tag {tag}"
            )));
        }
        return Ok(Vec::new());
    }

    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let name = read_name(cur, is_cdf5)?;
        let nc_type = NcType::from_code(be_u32(cur)?)?;
        let n = read_size(cur)? as usize;
        let bytes = n * nc_type.size();
        let raw = cur.take(bytes)?.to_vec();
        cur.skip(pad4(bytes as u64) as usize - bytes)?;

        let (text, numbers) = if nc_type == NcType::Char {
            let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
            (
                Some(String::from_utf8_lossy(&raw[..end]).into_owned()),
                Vec::new(),
            )
        } else {
            // Values are big-endian on disk; flip into native before widening.
            let mut native = raw.clone();
            let size = nc_type.size();
            if size > 1 {
                for c in native.chunks_exact_mut(size) {
                    c.reverse();
                }
            }
            (None, decode_numbers(&native, nc_type)?)
        };

        out.push(ClassicAttribute {
            name,
            nc_type,
            text,
            numbers,
            raw,
        });
    }
    Ok(out)
}

/// Names are a length, the bytes, then padding to a four-byte boundary.
fn read_name(cur: &mut Cursor<'_>, is_cdf5: bool) -> Result<String> {
    let len = if is_cdf5 {
        be_u64(cur)? as usize
    } else {
        be_u32(cur)? as usize
    };
    let raw = cur.take(len)?;
    let name = String::from_utf8_lossy(raw).into_owned();
    cur.skip(pad4(len as u64) as usize - len)?;
    Ok(name)
}

fn pad4(n: u64) -> u64 {
    n.div_ceil(4) * 4
}

fn be_u32(cur: &mut Cursor<'_>) -> Result<u32> {
    let b = cur.take(4)?;
    Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

fn be_u64(cur: &mut Cursor<'_>) -> Result<u64> {
    let b = cur.take(8)?;
    Ok(u64::from_be_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLASSIC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files/classic.nc");
    const CLASSIC64: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_files/classic64.nc");

    #[test]
    fn nc_type_sizes_are_right() {
        assert_eq!(NcType::Byte.size(), 1);
        assert_eq!(NcType::Short.size(), 2);
        assert_eq!(NcType::Float.size(), 4);
        assert_eq!(NcType::Double.size(), 8);
        assert!(NcType::Int.is_signed_integer());
        assert!(!NcType::Float.is_integer());
    }

    #[test]
    fn pads_to_four_byte_boundaries() {
        assert_eq!(pad4(0), 0);
        assert_eq!(pad4(1), 4);
        assert_eq!(pad4(4), 4);
        assert_eq!(pad4(5), 8);
    }

    #[test]
    fn reads_the_cdf1_header() {
        let file = ClassicFile::open(CLASSIC).unwrap();
        assert_eq!(file.container, Container::Cdf1);
        assert_eq!(file.record_count, 4);

        let names: Vec<&str> = file.dimensions.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, vec!["time", "level", "name_strlen"]);
        assert!(file.dimension("time").unwrap().is_unlimited);
        assert_eq!(file.dimension("time").unwrap().len, 4);
        assert_eq!(file.dimension("level").unwrap().len, 3);

        let vars: Vec<&str> = file.variables.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(vars, vec!["time", "pressure", "count", "flag", "label"]);
    }

    #[test]
    fn reads_global_and_variable_attributes() {
        let file = ClassicFile::open(CLASSIC).unwrap();
        assert_eq!(
            file.attribute("title").unwrap().text.as_deref(),
            Some("classic format fixture")
        );
        assert_eq!(file.attribute("version").unwrap().numbers, vec![3.0]);

        let p = file.variable("pressure").unwrap();
        assert_eq!(p.attribute("units").unwrap().text.as_deref(), Some("dbar"));
        assert_eq!(p.attribute("_FillValue").unwrap().numbers, vec![-999.0]);
    }

    #[test]
    fn reads_a_fixed_size_variable() {
        let file = ClassicFile::open(CLASSIC).unwrap();
        let v = file.variable("count").unwrap();
        assert!(!v.is_record);
        assert_eq!(v.shape, vec![3]);
        assert_eq!(file.read_f64(v).unwrap(), vec![10.0, 20.0, 30.0]);
    }

    /// Record variables are interleaved, so this is the case a naive reader
    /// gets wrong: `pressure` and `flag` and `time` alternate on disk.
    #[test]
    fn reads_interleaved_record_variables() {
        let file = ClassicFile::open(CLASSIC).unwrap();

        let t = file.variable("time").unwrap();
        assert!(t.is_record);
        assert_eq!(file.read_f64(t).unwrap(), vec![0.0, 1.0, 2.0, 3.0]);

        let p = file.variable("pressure").unwrap();
        assert_eq!(p.shape, vec![4, 3]);
        let expected: Vec<f64> = (0..12).map(|i| i as f64 + 0.5).collect();
        assert_eq!(file.read_f64(p).unwrap(), expected);

        let f = file.variable("flag").unwrap();
        let expected: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        assert_eq!(file.read_f64(f).unwrap(), expected);
    }

    #[test]
    fn reads_character_variables_as_strings() {
        let file = ClassicFile::open(CLASSIC).unwrap();
        let v = file.variable("label").unwrap();
        assert_eq!(
            file.read_strings(v).unwrap(),
            vec!["surface", "middle", "bottom"]
        );
    }

    #[test]
    fn reads_a_hyperslab_of_a_record_variable() {
        let file = ClassicFile::open(CLASSIC).unwrap();
        let v = file.variable("pressure").unwrap();

        let slab = Hyperslab::new(vec![1, 1], vec![2, 2], &v.shape).unwrap();
        let bytes = file.read_selection(v, &slab).unwrap();
        let got = decode_numbers(&bytes, v.nc_type).unwrap();
        // Rows 1 and 2, columns 1 and 2 of the 4x3 grid of i+0.5.
        assert_eq!(got, vec![4.5, 5.5, 7.5, 8.5]);
    }

    #[test]
    fn reads_the_sixty_four_bit_offset_variant() {
        let file = ClassicFile::open(CLASSIC64).unwrap();
        assert_eq!(file.container, Container::Cdf2);
        let v = file.variable("count").unwrap();
        assert_eq!(file.read_f64(v).unwrap(), vec![10.0, 20.0, 30.0]);
        let p = file.variable("pressure").unwrap();
        let expected: Vec<f64> = (0..12).map(|i| i as f64 + 0.5).collect();
        assert_eq!(file.read_f64(p).unwrap(), expected);
    }

    #[test]
    fn the_index_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ClassicFile>();
    }

    #[test]
    fn rejects_a_file_that_is_not_classic() {
        let src = oxcdf_hdf5::source::MemorySource::new(oxcdf_hdf5::HDF5_SIGNATURE.to_vec());
        assert!(ClassicFile::from_source(Arc::new(src)).is_err());
    }
}
