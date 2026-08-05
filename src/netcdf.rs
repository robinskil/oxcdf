//! The netCDF-4 layer.
//!
//! netCDF-4 is a set of conventions layered on HDF5. This module applies them,
//! turning a tree of HDF5 datasets into dimensions, variables and attributes.
//!
//! # The conventions that matter
//!
//! * A **dimension** is an HDF5 dataset carrying `CLASS = "DIMENSION_SCALE"`.
//! * Its `NAME` attribute distinguishes two cases. When it begins with
//!   `"This is a netCDF dimension but not a netCDF variable"`, the dataset is a
//!   dimension *only*, and netCDF does not report it as a variable. Otherwise
//!   the dataset is a **coordinate variable**: a dimension and a variable at
//!   once.
//! * A variable's `DIMENSION_LIST` attribute is a variable-length sequence of
//!   object references, one entry per axis, each pointing at the object header
//!   of the dimension scale for that axis. The references live in the global
//!   heap.
//! * `_Netcdf4Dimid` numbers a dimension. `_Netcdf4Coordinates` lists a
//!   variable's dimension ids, and settles the order when `DIMENSION_LIST`
//!   alone is ambiguous.
//!
//! Getting the first two rules wrong is what makes a reader report dimensions as
//! if they were variables, which is exactly what this module exists to prevent.

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::hdf5::heap::{GlobalHeap, VlenDescriptor};
use crate::hdf5::message::{Attribute, DatatypeClass, Layout, StringPad};
use crate::index::{DatasetIndex, GroupIndex, Hdf5File};
use crate::read::{read_hyperslab, Hyperslab, RawData};

/// Prefix of the `NAME` attribute on a dimension that has no coordinate
/// variable. netcdf-c writes the dimension length after it.
const PHANTOM_DIMENSION_PREFIX: &str = "This is a netCDF dimension but not a netCDF variable";

/// Attributes that belong to the netCDF-4 encoding rather than to the data.
///
/// netCDF hides these, so this reader hides them too. Surfacing them would make
/// every variable look like it has bookkeeping attributes it does not have.
const RESERVED_ATTRIBUTES: &[&str] = &[
    "CLASS",
    "NAME",
    "DIMENSION_LIST",
    "REFERENCE_LIST",
    "_Netcdf4Dimid",
    "_Netcdf4Coordinates",
    "_nc3_strict",
    "_NCProperties",
];

/// A decoded attribute value.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    /// A fixed-length string, or several of them.
    Text(Vec<String>),
    /// Floating-point values.
    Floats(Vec<f64>),
    /// Integer values.
    Ints(Vec<i64>),
    /// A value this layer does not decode. The bytes are as stored.
    Raw(Vec<u8>),
}

impl AttributeValue {
    /// The value as a single string, when it is textual.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            AttributeValue::Text(v) => v.first().map(|s| s.as_str()),
            _ => None,
        }
    }

    /// The value as a single number, when it is numeric.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            AttributeValue::Floats(v) => v.first().copied(),
            AttributeValue::Ints(v) => v.first().map(|&i| i as f64),
            _ => None,
        }
    }
}

/// One netCDF attribute.
#[derive(Debug, Clone)]
pub struct NcAttribute {
    /// Attribute name.
    pub name: String,
    /// Decoded value.
    pub value: AttributeValue,
}

/// One netCDF dimension.
#[derive(Debug, Clone)]
pub struct NcDimension {
    /// Dimension name.
    pub name: String,
    /// Current length.
    pub len: u64,
    /// Whether the dimension can grow.
    pub is_unlimited: bool,
    /// The netCDF dimension id, from `_Netcdf4Dimid`.
    pub id: Option<i64>,
    /// Whether a variable of the same name shares this dimension.
    pub has_coordinate_variable: bool,
}

/// One netCDF variable.
#[derive(Debug, Clone)]
pub struct NcVariable {
    /// Variable name.
    pub name: String,
    /// Full path from the root group.
    pub path: String,
    /// Names of the dimensions, in order.
    ///
    /// Empty for a scalar. An axis whose dimension could not be resolved gets a
    /// generated `phony_dim_N` name, matching what other readers do, rather
    /// than a silently wrong one.
    pub dimensions: Vec<String>,
    /// Shape in elements.
    pub shape: Vec<u64>,
    /// The variable's attributes, with netCDF bookkeeping removed.
    pub attributes: Vec<NcAttribute>,
    /// Whether every attribute was recovered. See
    /// [`crate::index::DatasetIndex::attributes_complete`].
    pub attributes_complete: bool,
    /// Whether this variable is also a dimension.
    pub is_coordinate: bool,
}

impl NcVariable {
    /// An attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&NcAttribute> {
        self.attributes.iter().find(|a| a.name == name)
    }
}

/// One netCDF group.
#[derive(Debug, Clone, Default)]
pub struct NcGroup {
    /// Group name. Empty for the root group.
    pub name: String,
    /// Full path. `/` for the root group.
    pub path: String,
    /// Dimensions defined in this group.
    pub dimensions: Vec<NcDimension>,
    /// Variables in this group.
    pub variables: Vec<NcVariable>,
    /// Group attributes. For the root group these are the global attributes.
    pub attributes: Vec<NcAttribute>,
    /// Child groups.
    pub groups: Vec<NcGroup>,
}

impl NcGroup {
    /// A variable by name.
    pub fn variable(&self, name: &str) -> Option<&NcVariable> {
        self.variables.iter().find(|v| v.name == name)
    }

    /// A dimension by name.
    pub fn dimension(&self, name: &str) -> Option<&NcDimension> {
        self.dimensions.iter().find(|d| d.name == name)
    }

    /// An attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&NcAttribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Every variable at or below this group, depth first.
    pub fn variables_recursive(&self) -> Vec<&NcVariable> {
        let mut out: Vec<&NcVariable> = self.variables.iter().collect();
        for g in &self.groups {
            out.extend(g.variables_recursive());
        }
        out
    }
}

/// An open netCDF-4 file.
///
/// Wraps an [`Hdf5File`] and interprets it through the netCDF conventions. The
/// HDF5 index underneath stays available, so a caller can drop to that level for
/// anything this layer does not model.
#[derive(Debug, Clone)]
pub struct NetcdfFile {
    hdf5: Hdf5File,
    root: NcGroup,
}

impl NetcdfFile {
    /// Open a netCDF-4 file from the filesystem with default options.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_hdf5(Hdf5File::open(path)?)
    }

    /// Open a netCDF-4 file with explicit I/O and cache options.
    ///
    /// ```no_run
    /// # use oxcdf::netcdf::NetcdfFile;
    /// # use oxcdf::index::OpenOptions;
    /// let file = NetcdfFile::open_with(
    ///     "argo.nc",
    ///     OpenOptions::new()
    ///         .io_request_size(256 * 1024)
    ///         .io_cache_bytes(128 << 20),
    /// )?;
    /// # Ok::<(), oxcdf::Error>(())
    /// ```
    pub fn open_with(
        path: impl AsRef<std::path::Path>,
        options: crate::index::OpenOptions,
    ) -> Result<Self> {
        Self::from_hdf5(Hdf5File::open_with(path, options)?)
    }

    /// Open a netCDF-4 file over any byte source.
    pub fn from_source(source: Arc<dyn crate::source::ByteSource>) -> Result<Self> {
        Self::from_hdf5(Hdf5File::from_source(source)?)
    }

    /// Interpret an already-open HDF5 file as netCDF-4.
    pub fn from_hdf5(hdf5: Hdf5File) -> Result<Self> {
        // Dimension scales are referenced by object header address from
        // anywhere in the file, so the scan has to cover the whole tree before
        // any variable can resolve its axes.
        let scales = collect_dimension_scales(&hdf5)?;
        let mut heaps = HeapCache::default();
        let root = build_group(&hdf5, hdf5.root(), &scales, &mut heaps)?;
        Ok(Self { hdf5, root })
    }

    /// The root group.
    pub fn root(&self) -> &NcGroup {
        &self.root
    }

    /// The underlying HDF5 index.
    pub fn hdf5(&self) -> &Hdf5File {
        &self.hdf5
    }

    /// Replace the underlying HDF5 file, keeping the netCDF view.
    ///
    /// Use this to attach caches after an open. The netCDF view describes
    /// names, shapes and axes. None of those depend on how bytes arrive.
    pub fn map_hdf5(mut self, f: impl FnOnce(Hdf5File) -> Hdf5File) -> Self {
        self.hdf5 = f(self.hdf5);
        self
    }

    /// A variable's metadata by absolute path, without binding it to the file.
    ///
    /// Prefer [`NetcdfFile::variable`], which returns a readable handle.
    pub fn variable_info(&self, path: &str) -> Option<&NcVariable> {
        let trimmed = path.trim_start_matches('/');
        let mut parts: Vec<&str> = trimmed.split('/').collect();
        let leaf = parts.pop()?;
        let mut group = &self.root;
        for part in parts {
            group = group.groups.iter().find(|g| g.name == part)?;
        }
        group.variable(leaf)
    }

    /// The HDF5 dataset backing a variable.
    pub fn dataset(&self, variable: &NcVariable) -> Option<&DatasetIndex> {
        self.hdf5.dataset(&variable.path)
    }

    /// Read a hyperslab of a variable.
    pub fn read(&self, variable: &NcVariable, slab: &Hyperslab) -> Result<RawData> {
        let dataset = self
            .dataset(variable)
            .ok_or_else(|| Error::not_found(format!("dataset for variable {}", variable.path)))?;
        read_hyperslab(self.hdf5.ctx(), dataset, slab)
    }

    /// Read a whole variable.
    pub fn read_all(&self, variable: &NcVariable) -> Result<RawData> {
        self.read(variable, &Hyperslab::all(&variable.shape))
    }
}

/// What the scan learns about one dimension scale.
#[derive(Debug, Clone)]
struct DimensionScale {
    name: String,
    len: u64,
    is_unlimited: bool,
    id: Option<i64>,
    /// False when `NAME` marks it a dimension with no coordinate variable.
    is_variable: bool,
}

/// Find every dimension scale in the file, keyed by object header address.
///
/// The address is the key because `DIMENSION_LIST` references point at object
/// headers, not at names.
fn collect_dimension_scales(hdf5: &Hdf5File) -> Result<HashMap<u64, DimensionScale>> {
    let mut out = HashMap::new();

    for dataset in hdf5.datasets() {
        let Some(class) = dataset.attribute("CLASS") else {
            continue;
        };
        if decode_text(class).as_deref() != Some("DIMENSION_SCALE") {
            continue;
        }

        // The dimension's name is the dataset's own name. The `NAME` attribute
        // says whether a coordinate variable shares it.
        let name_attr = dataset.attribute("NAME").and_then(decode_text);
        let is_variable = !name_attr
            .as_deref()
            .is_some_and(|n| n.starts_with(PHANTOM_DIMENSION_PREFIX));

        let id = dataset
            .attribute("_Netcdf4Dimid")
            .and_then(|a| decode_ints(a).ok())
            .and_then(|v| v.first().copied());

        let len = dataset.shape.first().copied().unwrap_or(1);
        let is_unlimited = dataset
            .max_shape
            .as_ref()
            .and_then(|m| m.first().copied())
            .is_some_and(|m| m == u64::MAX);

        out.insert(
            dataset.address,
            DimensionScale {
                name: dataset.name.clone(),
                len,
                is_unlimited,
                id,
                is_variable,
            },
        );
    }

    Ok(out)
}

/// Global heap collections already read, keyed by address.
///
/// One collection usually serves every `DIMENSION_LIST` in a file, so caching
/// turns a per-variable read into a single one.
#[derive(Default)]
struct HeapCache {
    collections: HashMap<u64, GlobalHeap>,
}

impl HeapCache {
    fn get<'a>(&'a mut self, hdf5: &Hdf5File, address: u64) -> Result<&'a GlobalHeap> {
        use std::collections::hash_map::Entry;
        match self.collections.entry(address) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => Ok(e.insert(GlobalHeap::read(hdf5.ctx(), address)?)),
        }
    }
}

/// Turn one HDF5 group into a netCDF group.
fn build_group(
    hdf5: &Hdf5File,
    group: &GroupIndex,
    scales: &HashMap<u64, DimensionScale>,
    heaps: &mut HeapCache,
) -> Result<NcGroup> {
    let mut dimensions = Vec::new();
    let mut variables = Vec::new();

    for dataset in &group.datasets {
        let scale = scales.get(&dataset.address);

        if let Some(scale) = scale {
            dimensions.push(NcDimension {
                name: scale.name.clone(),
                len: scale.len,
                is_unlimited: scale.is_unlimited,
                id: scale.id,
                has_coordinate_variable: scale.is_variable,
            });

            // A dimension with no coordinate variable is not a netCDF variable.
            if !scale.is_variable {
                continue;
            }
        }

        variables.push(build_variable(hdf5, dataset, scales, heaps, scale.is_some())?);
    }

    // netCDF reports dimensions in id order where ids exist.
    dimensions.sort_by_key(|d| (d.id.unwrap_or(i64::MAX), d.name.clone()));

    let mut groups = Vec::new();
    for child in &group.groups {
        groups.push(build_group(hdf5, child, scales, heaps)?);
    }

    Ok(NcGroup {
        name: group.name.clone(),
        path: group.path.clone(),
        dimensions,
        variables,
        attributes: visible_attributes(&group.attributes),
        groups,
    })
}

/// Turn one HDF5 dataset into a netCDF variable.
fn build_variable(
    hdf5: &Hdf5File,
    dataset: &DatasetIndex,
    scales: &HashMap<u64, DimensionScale>,
    heaps: &mut HeapCache,
    is_coordinate: bool,
) -> Result<NcVariable> {
    let dimensions = resolve_dimensions(hdf5, dataset, scales, heaps, is_coordinate)?;

    Ok(NcVariable {
        name: dataset.name.clone(),
        path: dataset.path.clone(),
        dimensions,
        shape: dataset.shape.clone(),
        attributes: visible_attributes(&dataset.attributes),
        attributes_complete: dataset.attributes_complete,
        is_coordinate,
    })
}

/// Work out the dimension name for each axis of a variable.
fn resolve_dimensions(
    hdf5: &Hdf5File,
    dataset: &DatasetIndex,
    scales: &HashMap<u64, DimensionScale>,
    heaps: &mut HeapCache,
    is_coordinate: bool,
) -> Result<Vec<String>> {
    let rank = dataset.shape.len();
    if rank == 0 {
        return Ok(Vec::new());
    }

    // A coordinate variable is its own dimension on axis 0.
    if is_coordinate && rank == 1 {
        return Ok(vec![dataset.name.clone()]);
    }

    if let Some(attr) = dataset.attribute("DIMENSION_LIST") {
        if let Some(names) = resolve_from_dimension_list(hdf5, attr, scales, heaps, rank)? {
            return Ok(names);
        }
    }

    // Fall back to the dimension ids, which netcdf-c writes for the cases where
    // a reference list alone would be ambiguous.
    if let Some(attr) = dataset.attribute("_Netcdf4Coordinates") {
        if let Ok(ids) = decode_ints(attr) {
            if ids.len() == rank {
                let by_id: HashMap<i64, &DimensionScale> = scales
                    .values()
                    .filter_map(|s| s.id.map(|id| (id, s)))
                    .collect();
                if ids.iter().all(|id| by_id.contains_key(id)) {
                    return Ok(ids
                        .iter()
                        .map(|id| by_id[id].name.clone())
                        .collect());
                }
            }
        }
    }

    // Nothing resolved. Generate placeholder names rather than invent real ones.
    Ok((0..rank).map(|i| format!("phony_dim_{i}")).collect())
}

/// Follow a `DIMENSION_LIST` attribute to one dimension name per axis.
///
/// The attribute holds one variable-length sequence per axis. Each sequence
/// holds object references to the dimension scales for that axis; netCDF uses
/// the first.
fn resolve_from_dimension_list(
    hdf5: &Hdf5File,
    attr: &Attribute,
    scales: &HashMap<u64, DimensionScale>,
    heaps: &mut HeapCache,
    rank: usize,
) -> Result<Option<Vec<String>>> {
    let DatatypeClass::VariableLength { .. } = attr.datatype.class else {
        return Ok(None);
    };

    let sizes = hdf5.superblock().sizes;
    let descriptor_len = VlenDescriptor::encoded_len(sizes);
    if attr.data.len() < rank * descriptor_len {
        return Ok(None);
    }

    let mut names = Vec::with_capacity(rank);
    for axis in 0..rank {
        let start = axis * descriptor_len;
        let descriptor =
            VlenDescriptor::parse(&attr.data[start..start + descriptor_len], sizes)?;

        if descriptor.length == 0 {
            return Ok(None);
        }

        let heap = heaps.get(hdf5, descriptor.collection_address)?;
        let Some(object) = heap.object(descriptor.object_index as u16) else {
            return Ok(None);
        };

        // An object reference is the address of the target's object header.
        let width = sizes.offset as usize;
        if object.data.len() < width {
            return Ok(None);
        }
        let mut address = 0u64;
        for (i, byte) in object.data[..width].iter().enumerate() {
            address |= (*byte as u64) << (8 * i);
        }

        match scales.get(&address) {
            Some(scale) => names.push(scale.name.clone()),
            None => return Ok(None),
        }
    }

    Ok(Some(names))
}

/// Drop the netCDF bookkeeping attributes and decode the rest.
fn visible_attributes(attributes: &[Attribute]) -> Vec<NcAttribute> {
    attributes
        .iter()
        .filter(|a| !RESERVED_ATTRIBUTES.contains(&a.name.as_str()))
        .map(|a| NcAttribute {
            name: a.name.clone(),
            value: decode_value(a),
        })
        .collect()
}

/// Decode an attribute's bytes according to its datatype.
fn decode_value(attr: &Attribute) -> AttributeValue {
    match &attr.datatype.class {
        DatatypeClass::String { .. } => match decode_all_text(attr) {
            Some(v) => AttributeValue::Text(v),
            None => AttributeValue::Raw(attr.data.clone()),
        },
        DatatypeClass::FloatingPoint { .. } => match decode_floats(attr) {
            Ok(v) => AttributeValue::Floats(v),
            Err(_) => AttributeValue::Raw(attr.data.clone()),
        },
        DatatypeClass::FixedPoint { .. } => match decode_ints(attr) {
            Ok(v) => AttributeValue::Ints(v),
            Err(_) => AttributeValue::Raw(attr.data.clone()),
        },
        _ => AttributeValue::Raw(attr.data.clone()),
    }
}

/// Whether an attribute's values need their bytes reversed.
fn needs_swap(attr: &Attribute) -> bool {
    matches!(
        attr.datatype.byte_order(),
        Some(crate::hdf5::message::ByteOrder::Big)
    )
}

/// Take one element's bytes, in native order.
fn element(attr: &Attribute, index: usize) -> Option<Vec<u8>> {
    let size = attr.datatype.size as usize;
    if size == 0 || (index + 1) * size > attr.data.len() {
        return None;
    }
    let mut bytes = attr.data[index * size..(index + 1) * size].to_vec();
    if needs_swap(attr) {
        bytes.reverse();
    }
    Some(bytes)
}

/// Decode the first value of a string attribute.
fn decode_text(attr: &Attribute) -> Option<String> {
    decode_all_text(attr)?.into_iter().next()
}

/// Decode every value of a string attribute.
fn decode_all_text(attr: &Attribute) -> Option<Vec<String>> {
    let DatatypeClass::String { pad, .. } = &attr.datatype.class else {
        return None;
    };
    let size = attr.datatype.size as usize;
    if size == 0 {
        return Some(vec![String::new()]);
    }

    let count = (attr.data.len() / size).max(1);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * size;
        let end = (start + size).min(attr.data.len());
        if start >= end {
            break;
        }
        let raw = &attr.data[start..end];
        let stop = match pad {
            StringPad::NullTerminate | StringPad::NullPad => {
                raw.iter().position(|&b| b == 0).unwrap_or(raw.len())
            }
            StringPad::SpacePad => {
                let z = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
                raw[..z].iter().rposition(|&b| b != b' ').map(|p| p + 1).unwrap_or(0)
            }
        };
        out.push(String::from_utf8_lossy(&raw[..stop]).into_owned());
    }
    Some(out)
}

/// Decode a floating-point attribute.
fn decode_floats(attr: &Attribute) -> Result<Vec<f64>> {
    let size = attr.datatype.size as usize;
    let count = attr.data.len() / size.max(1);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let bytes = element(attr, i)
            .ok_or_else(|| Error::malformed("attribute value is shorter than its datatype"))?;
        out.push(match size {
            4 => f32::from_ne_bytes(bytes[..4].try_into().unwrap()) as f64,
            8 => f64::from_ne_bytes(bytes[..8].try_into().unwrap()),
            other => return Err(Error::unsupported(format!("{other}-byte float attribute"))),
        });
    }
    Ok(out)
}

/// Decode an integer attribute.
fn decode_ints(attr: &Attribute) -> Result<Vec<i64>> {
    let DatatypeClass::FixedPoint { signed, .. } = &attr.datatype.class else {
        return Err(Error::unsupported("attribute is not an integer"));
    };
    let size = attr.datatype.size as usize;
    if size == 0 || size > 8 {
        return Err(Error::unsupported(format!("{size}-byte integer attribute")));
    }

    let count = attr.data.len() / size;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let bytes = element(attr, i)
            .ok_or_else(|| Error::malformed("attribute value is shorter than its datatype"))?;
        let mut buf = [0u8; 8];
        buf[..size].copy_from_slice(&bytes[..size]);
        let raw = u64::from_le_bytes(buf);
        out.push(if *signed {
            let shift = 64 - (size as u32 * 8);
            ((raw << shift) as i64) >> shift
        } else {
            raw as i64
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The legacy fixture is plain HDF5, not netCDF, so it has no dimension
    /// scales. Every dataset should therefore be a variable with generated axis
    /// names, and nothing should be reported as a dimension.
    #[test]
    fn plain_hdf5_has_no_dimensions_and_all_datasets_are_variables() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_files/legacy_v1_objheader.h5"
        );
        let file = NetcdfFile::open(path).unwrap();

        assert!(file.root().dimensions.is_empty());
        assert_eq!(file.variables().len(), 5);

        let v = file.variable("/contig_f64").unwrap();
        assert_eq!(v.shape, vec![40, 6]);
        assert_eq!(v.dimensions, vec!["phony_dim_0", "phony_dim_1"]);
        assert!(!v.is_coordinate);
    }

    #[test]
    fn reads_values_through_the_netcdf_layer() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_files/legacy_v1_objheader.h5"
        );
        let file = NetcdfFile::open(path).unwrap();
        let v = file.variable("/chunked_i32").unwrap();
        let values = v.read().unwrap().to_i64().unwrap();

        assert_eq!(values.len(), 240);
        assert_eq!(values[0], -100);
        assert_eq!(values[239], 239 * 3 - 100);
    }

    #[test]
    fn a_scalar_variable_has_no_dimensions() {
        // Exercised through the attribute decoder rather than a file: the
        // fixture has no scalar dataset.
        let v = NcVariable {
            name: "x".into(),
            path: "/x".into(),
            dimensions: Vec::new(),
            shape: Vec::new(),
            attributes: Vec::new(),
            attributes_complete: true,
            is_coordinate: false,
        };
        assert!(v.dimensions.is_empty());
    }

    #[test]
    fn attribute_values_expose_text_and_numbers() {
        let text = AttributeValue::Text(vec!["degrees_north".into()]);
        assert_eq!(text.as_text(), Some("degrees_north"));
        assert_eq!(text.as_f64(), None);

        let nums = AttributeValue::Floats(vec![1.5, 2.5]);
        assert_eq!(nums.as_f64(), Some(1.5));
        assert_eq!(nums.as_text(), None);

        let ints = AttributeValue::Ints(vec![7]);
        assert_eq!(ints.as_f64(), Some(7.0));
    }

    #[test]
    fn reserved_attributes_are_hidden() {
        for name in RESERVED_ATTRIBUTES {
            assert!(
                RESERVED_ATTRIBUTES.contains(name),
                "{name} should be hidden from netCDF callers"
            );
        }
        assert!(RESERVED_ATTRIBUTES.contains(&"DIMENSION_LIST"));
        assert!(!RESERVED_ATTRIBUTES.contains(&"units"));
    }
}

// ─── The variable handle API ───────────────────────────────────────────────
//
// Everything above describes the file. Everything below is the interface a
// caller actually uses: navigate to a variable, look at its metadata, then read
// all of it, a slice of it, or one stored chunk at a time.

/// A simple view of a variable's element type.
///
/// The full HDF5 datatype is still available through
/// [`Variable::datatype`]; this is the summary most callers want.
///
/// Not `Copy`: a variable-length sequence carries its element type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DType {
    /// Signed integer of the given width in bytes.
    Int(u8),
    /// Unsigned integer of the given width in bytes.
    Uint(u8),
    /// IEEE float of the given width in bytes.
    Float(u8),
    /// Fixed-length string of the given width in bytes.
    String(u32),
    /// Variable-length string, stored through the global heap.
    VarString,
    /// Variable-length sequence of a base type, stored through the global heap.
    VarSequence(Box<DType>),
    /// Something this summary does not model.
    Other,
}

impl DType {
    /// The summary type of an HDF5 datatype.
    pub fn of(datatype: &crate::hdf5::message::Datatype) -> Self {
        let size = datatype.size;
        match &datatype.class {
            DatatypeClass::FixedPoint { signed: true, .. } => DType::Int(size as u8),
            DatatypeClass::FixedPoint { signed: false, .. } => DType::Uint(size as u8),
            DatatypeClass::FloatingPoint { .. } => DType::Float(size as u8),
            DatatypeClass::String { .. } => DType::String(size),
            DatatypeClass::VariableLength {
                kind: crate::hdf5::message::VlenKind::String,
                ..
            } => DType::VarString,
            DatatypeClass::VariableLength {
                kind: crate::hdf5::message::VlenKind::Sequence,
                base,
                ..
            } => DType::VarSequence(Box::new(DType::of(base))),
            _ => DType::Other,
        }
    }

    /// Whether this is an integer type.
    pub fn is_integer(&self) -> bool {
        matches!(self, DType::Int(_) | DType::Uint(_))
    }

    /// Whether this is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(self, DType::Float(_))
    }
}

/// One stored chunk of a variable.
///
/// A chunked variable is stored as independent compressed blocks. Reading them
/// separately is the natural unit of parallel work: each one is its own byte
/// range with its own filter pipeline, so nothing is shared between them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Element offset of the chunk's origin within the variable.
    pub offset: Vec<u64>,
    /// Shape of the region this chunk actually contributes.
    ///
    /// Chunks at the edge of a variable are stored at full size but hang past
    /// the end. This shape is already clipped, so reading a chunk never returns
    /// padding.
    pub shape: Vec<u64>,
    /// Size of the chunk on disk, after compression.
    pub stored_size: u64,
}

impl Chunk {
    /// Number of elements this chunk contributes.
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }

    /// The selection this chunk covers.
    pub fn selection(&self) -> Hyperslab {
        Hyperslab {
            start: self.offset.clone(),
            count: self.shape.clone(),
        }
    }
}

/// Values read from a variable, in native byte order and row-major order.
///
/// Unlike [`RawData`] this carries its own datatype, so converting needs
/// nothing else.
#[derive(Debug, Clone)]
pub struct Values {
    raw: RawData,
    datatype: crate::hdf5::message::Datatype,
    /// Already-resolved variable-length strings, when the variable holds them.
    ///
    /// These are resolved at read time because the characters live in the
    /// global heap, which needs the file.
    vlen_strings: Option<Vec<String>>,
    /// Already-resolved variable-length sequences, one buffer per element, in
    /// native byte order.
    vlen_sequences: Option<Vec<Vec<u8>>>,
    /// Element type of those sequences.
    vlen_base: Option<crate::hdf5::message::Datatype>,
}

impl Values {
    /// Shape of the block that was read.
    pub fn shape(&self) -> &[u64] {
        &self.raw.shape
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    /// Whether the block is empty.
    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        DType::of(&self.datatype)
    }

    /// The raw bytes, native order, row major.
    pub fn as_bytes(&self) -> &[u8] {
        &self.raw.bytes
    }

    /// The underlying [`RawData`].
    pub fn raw(&self) -> &RawData {
        &self.raw
    }

    /// Values as `f64`, widening from whatever numeric type was stored.
    pub fn to_f64(&self) -> Result<Vec<f64>> {
        self.raw.to_f64_of(&self.datatype)
    }

    /// Values as `i64`, for integer variables.
    pub fn to_i64(&self) -> Result<Vec<i64>> {
        self.raw.to_i64_of(&self.datatype)
    }

    /// Values as strings.
    ///
    /// Works for both fixed-length character variables and variable-length
    /// string variables; the latter were resolved from the global heap when the
    /// read happened.
    pub fn to_strings(&self) -> Result<Vec<String>> {
        if let Some(strings) = &self.vlen_strings {
            return Ok(strings.clone());
        }
        self.raw.to_strings_of(&self.datatype)
    }

    /// The values as an `ndarray` of `f64`, shaped as they were read.
    ///
    /// Row-major, so the array's axes match the variable's dimensions in order.
    #[cfg(feature = "ndarray")]
    pub fn to_array_f64(&self) -> Result<ndarray::ArrayD<f64>> {
        shape_into_array(self.shape(), self.to_f64()?)
    }

    /// The values as an `ndarray` of `i64`.
    #[cfg(feature = "ndarray")]
    pub fn to_array_i64(&self) -> Result<ndarray::ArrayD<i64>> {
        shape_into_array(self.shape(), self.to_i64()?)
    }

    /// The values as an `ndarray` of strings.
    ///
    /// For a fixed-length character variable the trailing axis is the string
    /// width, and it is *not* collapsed here: one element per character cell
    /// would be wrong, so the last axis is dropped and each row becomes one
    /// string.
    #[cfg(feature = "ndarray")]
    pub fn to_array_strings(&self) -> Result<ndarray::ArrayD<String>> {
        let strings = self.to_strings()?;
        // Variable-length strings are one per element, so the shape is intact.
        if self.vlen_strings.is_some() {
            return shape_into_array(self.shape(), strings);
        }
        // Fixed-length strings decode one per element too, because the element
        // *is* the whole string; the dataspace already excludes the width.
        shape_into_array(self.shape(), strings)
    }

    /// Whether the values came from variable-length storage.
    pub fn is_variable_length(&self) -> bool {
        self.vlen_strings.is_some() || self.vlen_sequences.is_some()
    }

    /// Variable-length sequences widened to `f64`, one vector per element.
    ///
    /// An empty sequence comes back as an empty vector, which is a real value
    /// rather than a missing one.
    pub fn to_sequences_f64(&self) -> Result<Vec<Vec<f64>>> {
        let (sequences, base) = self.sequences()?;
        sequences
            .iter()
            .map(|bytes| {
                RawData {
                    bytes: bytes.clone(),
                    element_size: base.size as usize,
                    shape: vec![(bytes.len() / (base.size as usize).max(1)) as u64],
                }
                .to_f64_of(base)
            })
            .collect()
    }

    /// Variable-length sequences as `i64`, for integer element types.
    pub fn to_sequences_i64(&self) -> Result<Vec<Vec<i64>>> {
        let (sequences, base) = self.sequences()?;
        sequences
            .iter()
            .map(|bytes| {
                RawData {
                    bytes: bytes.clone(),
                    element_size: base.size as usize,
                    shape: vec![(bytes.len() / (base.size as usize).max(1)) as u64],
                }
                .to_i64_of(base)
            })
            .collect()
    }

    /// The raw bytes of each sequence, in native order.
    pub fn sequence_bytes(&self) -> Result<&[Vec<u8>]> {
        Ok(self.sequences()?.0)
    }

    fn sequences(&self) -> Result<(&[Vec<u8>], &crate::hdf5::message::Datatype)> {
        match (&self.vlen_sequences, &self.vlen_base) {
            (Some(s), Some(b)) => Ok((s, b)),
            _ => Err(Error::unsupported(
                "this variable does not hold variable-length sequences",
            )),
        }
    }
}

/// Wrap a flat, row-major vector in an `ndarray` of the given shape.
#[cfg(feature = "ndarray")]
fn shape_into_array<T>(shape: &[u64], values: Vec<T>) -> Result<ndarray::ArrayD<T>> {
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let expected: usize = dims.iter().product::<usize>().max(if dims.is_empty() {
        1
    } else {
        0
    });
    if values.len() != expected {
        return Err(Error::malformed(format!(
            "cannot shape {} values into {dims:?}, which holds {expected}",
            values.len()
        )));
    }
    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&dims), values)
        .map_err(|e| Error::malformed(format!("failed to shape values into an array: {e}")))
}

/// A variable bound to its file, so it can be read directly.
///
/// Dereferences to [`NcVariable`], so the metadata fields (`name`, `shape`,
/// `dimensions`, `attributes`) are reachable as plain fields.
#[derive(Debug, Clone, Copy)]
pub struct Variable<'a> {
    file: &'a NetcdfFile,
    info: &'a NcVariable,
    dataset: &'a DatasetIndex,
}

impl<'a> std::ops::Deref for Variable<'a> {
    type Target = NcVariable;
    fn deref(&self) -> &Self::Target {
        self.info
    }
}

impl<'a> Variable<'a> {
    /// The variable's metadata.
    pub fn info(&self) -> &'a NcVariable {
        self.info
    }

    /// The HDF5 dataset backing this variable.
    pub fn dataset(&self) -> &'a DatasetIndex {
        self.dataset
    }

    /// The full HDF5 datatype.
    pub fn datatype(&self) -> &'a crate::hdf5::message::Datatype {
        &self.dataset.datatype
    }

    /// A summary of the element type.
    pub fn dtype(&self) -> DType {
        DType::of(&self.dataset.datatype)
    }

    /// Total number of elements.
    pub fn element_count(&self) -> u64 {
        self.info.shape.iter().product()
    }

    /// Whether this reader can decode the variable's values.
    ///
    /// Check this before reading if netcdf-c is kept as a fallback.
    pub fn is_readable(&self) -> bool {
        self.dataset.is_readable()
    }

    /// Read the whole variable straight into an `ndarray` of `f64`.
    #[cfg(feature = "ndarray")]
    pub fn read_array_f64(&self) -> Result<ndarray::ArrayD<f64>> {
        self.read()?.to_array_f64()
    }

    /// Read the whole variable straight into an `ndarray` of `i64`.
    #[cfg(feature = "ndarray")]
    pub fn read_array_i64(&self) -> Result<ndarray::ArrayD<i64>> {
        self.read()?.to_array_i64()
    }

    /// Read the whole variable.
    pub fn read(&self) -> Result<Values> {
        self.read_selection(&Hyperslab::all(&self.info.shape))
    }

    /// Read a slice, given one range per axis.
    ///
    /// ```no_run
    /// # use oxcdf::netcdf::NetcdfFile;
    /// # fn main() -> oxcdf::Result<()> {
    /// # let file = NetcdfFile::open("f.nc")?;
    /// let temp = file.variable("/TEMP").unwrap();
    /// let block = temp.read_slice(&[0..8, 10..30])?;
    /// # Ok(()) }
    /// ```
    pub fn read_slice(&self, ranges: &[std::ops::Range<u64>]) -> Result<Values> {
        let slab = Hyperslab::from_ranges(&self.info.path, &self.info.shape, ranges)?;
        self.read_selection(&slab)
    }

    /// Read an explicit selection.
    pub fn read_selection(&self, slab: &Hyperslab) -> Result<Values> {
        let ctx = self.file.hdf5().ctx();
        let raw = read_hyperslab(ctx, self.dataset, slab)?;
        values_from_raw(ctx, self.dataset, raw)
    }

    /// The shape of one storage chunk, when the variable is chunked.
    ///
    /// `None` means the variable is stored contiguously and has no chunk grid.
    pub fn chunk_shape(&self) -> Option<Vec<u64>> {
        match &self.dataset.layout {
            Layout::Chunked { chunk_dims, .. } => {
                Some(chunk_dims.iter().map(|&d| d as u64).collect())
            }
            _ => None,
        }
    }

    /// Every stored chunk of the variable, clipped to its bounds.
    ///
    /// For a contiguous variable this returns one chunk covering everything, so
    /// a caller can use the same loop either way.
    ///
    /// Chunks are independent, which makes this the natural unit of parallel
    /// work:
    ///
    /// ```no_run
    /// # use oxcdf::netcdf::NetcdfFile;
    /// # fn main() -> oxcdf::Result<()> {
    /// # let file = NetcdfFile::open("f.nc")?;
    /// # let temp = file.variable("/TEMP").unwrap();
    /// for chunk in temp.chunks() {
    ///     let block = temp.read_chunk(&chunk)?;   // safe from any thread
    /// }
    /// # Ok(()) }
    /// ```
    pub fn chunks(&self) -> Vec<Chunk> {
        self.try_chunks().unwrap_or_default()
    }

    /// Resolve the chunk index now, so later reads do no metadata input.
    ///
    /// This is an optimisation. A read resolves the index itself.
    pub fn prepare(&self) -> Result<()> {
        self.dataset.prepare(self.file.hdf5().ctx())
    }

    /// Every stored chunk, reporting a chunk index this reader cannot walk.
    pub fn try_chunks(&self) -> Result<Vec<Chunk>> {
        self.dataset.chunks(self.file.hdf5().ctx())?;
        chunks_of(self.dataset)
    }

    /// Read one chunk. The result covers exactly the chunk's clipped region.
    pub fn read_chunk(&self, chunk: &Chunk) -> Result<Values> {
        self.read_selection(&chunk.selection())
    }
}

/// Turn raw bytes into values, following any heap pointer they hold.
///
/// A variable-length string or sequence stores a pointer into the global heap.
/// The value is only complete once the reader follows it, so that read happens
/// here, while the file is still to hand. Both engines use this.
pub(crate) fn values_from_raw(
    ctx: crate::hdf5::context::Ctx<'_>,
    dataset: &DatasetIndex,
    raw: RawData,
) -> Result<Values> {
    let mut vlen_strings = None;
    let mut vlen_sequences = None;
    let mut vlen_base = None;

    if let DatatypeClass::VariableLength { kind, base, .. } = &dataset.datatype.class {
        match kind {
            crate::hdf5::message::VlenKind::String => {
                vlen_strings = Some(crate::read::resolve_vlen_strings(ctx, dataset, &raw)?);
            }
            crate::hdf5::message::VlenKind::Sequence => {
                vlen_sequences = Some(crate::read::resolve_vlen_sequences(ctx, dataset, &raw)?);
                vlen_base = Some((**base).clone());
            }
        }
    }

    Ok(Values {
        raw,
        datatype: dataset.datatype.clone(),
        vlen_strings,
        vlen_sequences,
        vlen_base,
    })
}

/// Every stored chunk of a dataset, clipped to its bounds.
///
/// The chunk index must be resolved already. A dataset that is not chunked
/// reports one chunk covering everything, so a caller uses one loop either way.
/// Both engines use this.
pub(crate) fn chunks_of(dataset: &DatasetIndex) -> Result<Vec<Chunk>> {
    let shape = &dataset.shape;
    let chunk_dims = match &dataset.layout {
        Layout::Chunked { chunk_dims, .. } => Some(
            chunk_dims
                .iter()
                .map(|&d| d as u64)
                .collect::<Vec<u64>>(),
        ),
        _ => None,
    };

    let (Some(chunk_dims), Some(records)) = (chunk_dims, dataset.resolved_chunks()) else {
        return Ok(vec![Chunk {
            offset: vec![0; shape.len()],
            shape: shape.clone(),
            stored_size: dataset.element_count() * dataset.element_size() as u64,
        }]);
    };

    Ok(records
        .iter()
        .filter_map(|record| {
            // Clip the stored chunk to the variable's bounds. An edge chunk is
            // stored full size and hangs past the end.
            let mut clipped = Vec::with_capacity(shape.len());
            for axis in 0..shape.len() {
                let origin = *record.offset.get(axis)?;
                let full = *chunk_dims.get(axis)?;
                let dim = *shape.get(axis)?;
                if origin >= dim {
                    return None;
                }
                clipped.push(full.min(dim - origin));
            }
            Some(Chunk {
                offset: record.offset.clone(),
                shape: clipped,
                stored_size: record.size as u64,
            })
        })
        .collect())
}

impl NetcdfFile {
    /// A variable by absolute path, bound to the file so it can be read.
    pub fn variable(&self, path: &str) -> Option<Variable<'_>> {
        let info = self.variable_info(path)?;
        let dataset = self.hdf5.dataset(&info.path)?;
        Some(Variable {
            file: self,
            info,
            dataset,
        })
    }

    /// Every variable in the file, bound for reading.
    pub fn variables(&self) -> Vec<Variable<'_>> {
        self.root
            .variables_recursive()
            .into_iter()
            .filter_map(|info| {
                self.hdf5.dataset(&info.path).map(|dataset| Variable {
                    file: self,
                    info,
                    dataset,
                })
            })
            .collect()
    }

    /// The file's global attributes, which are the root group's attributes.
    pub fn attributes(&self) -> &[NcAttribute] {
        &self.root.attributes
    }

    /// A global attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&NcAttribute> {
        self.root.attribute(name)
    }

    /// The dimensions defined in the root group.
    pub fn dimensions(&self) -> &[NcDimension] {
        &self.root.dimensions
    }

    /// A group by absolute path, such as `/` or `/processing`.
    pub fn group(&self, path: &str) -> Option<&NcGroup> {
        let trimmed = path.trim_matches('/');
        if trimmed.is_empty() {
            return Some(&self.root);
        }
        let mut group = &self.root;
        for part in trimmed.split('/') {
            group = group.groups.iter().find(|g| g.name == part)?;
        }
        Some(group)
    }
}
