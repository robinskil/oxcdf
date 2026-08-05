//! The netCDF-4 layer.
//!
//! netCDF-4 is a set of conventions over HDF5. This module applies them. It
//! turns a tree of HDF5 datasets into dimensions, variables and attributes.
//!
//! # The conventions that matter
//!
//! * A **dimension** is an HDF5 dataset with `CLASS = "DIMENSION_SCALE"`.
//! * Its `NAME` attribute separates two cases. A name that starts with
//!   `"This is a netCDF dimension but not a netCDF variable"` marks a dimension
//!   only. netCDF does not report it as a variable. Any other name marks a
//!   **coordinate variable**, which is a dimension and a variable at once.
//! * A variable's `DIMENSION_LIST` attribute holds one object reference for
//!   each axis. Each reference names the object header of that axis' dimension
//!   scale. The references live in the global heap.
//! * `_Netcdf4Dimid` numbers a dimension. `_Netcdf4Coordinates` lists a
//!   variable's dimension ids. It settles the order when `DIMENSION_LIST` alone
//!   is ambiguous.
//!
//! A reader that breaks the first two rules reports dimensions as variables.
//! This module exists to prevent that.

use std::collections::HashMap;
use std::sync::Arc;

pub use oxcdf_hdf5::dtype::{DType, Element};
use oxcdf_hdf5::error::{Error, Result};
use oxcdf_hdf5::heap::{GlobalHeap, VlenDescriptor};
use oxcdf_hdf5::index::{DatasetIndex, GroupIndex, Hdf5File};
use oxcdf_hdf5::message::{Attribute, Dataspace, DatatypeClass, Layout, StringPad};

use crate::extent::Extents;
// `Chunk` describes an HDF5 storage chunk, so the lower crate owns it. It is
// re-exported here because a caller reaches it through a netCDF variable.
pub use oxcdf_hdf5::read::Chunk;
use oxcdf_hdf5::read::{read_hyperslab, Hyperslab, RawData};

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
///
/// This mirrors `netcdf::AttributeValue`. The value keeps the type the file
/// stores. A single value gets the singular variant, and several values get the
/// plural one, exactly as that crate does.
///
/// ```no_run
/// # use oxcdf::AttributeValue;
/// # fn run(var: oxcdf::Variable<'_>) -> oxcdf::Result<()> {
/// match &var.attribute("_FillValue").unwrap().value {
///     AttributeValue::Float(v) => println!("f32 fill value {v}"),
///     AttributeValue::Double(v) => println!("f64 fill value {v}"),
///     other => println!("{other:?}"),
/// }
///
/// // Or take whatever number is there.
/// let scale = var.attribute("scale_factor").and_then(|a| a.value.as_f64());
/// # Ok(()) }
/// ```
#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)] // Each variant is its netCDF type; the names say it.
pub enum AttributeValue {
    Uchar(u8),
    Uchars(Vec<u8>),
    Schar(i8),
    Schars(Vec<i8>),
    Ushort(u16),
    Ushorts(Vec<u16>),
    Short(i16),
    Shorts(Vec<i16>),
    Uint(u32),
    Uints(Vec<u32>),
    Int(i32),
    Ints(Vec<i32>),
    Ulonglong(u64),
    Ulonglongs(Vec<u64>),
    Longlong(i64),
    Longlongs(Vec<i64>),
    Float(f32),
    Floats(Vec<f32>),
    Double(f64),
    Doubles(Vec<f64>),
    Str(String),
    Strs(Vec<String>),
    /// A value this reader does not decode. The bytes are as stored.
    ///
    /// The `netcdf` crate has no such variant, because netcdf-c refuses the
    /// file. This reader parses HDF5 directly, so it can meet a type netCDF
    /// never defines. The attribute stays visible rather than vanishing.
    Raw(Vec<u8>),
}

impl AttributeValue {
    /// The value as one string, when it is textual.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            AttributeValue::Str(s) => Some(s.as_str()),
            AttributeValue::Strs(v) => v.first().map(|s| s.as_str()),
            _ => None,
        }
    }

    /// Every string, when the value is textual.
    pub fn as_texts(&self) -> Option<Vec<&str>> {
        match self {
            AttributeValue::Str(s) => Some(vec![s.as_str()]),
            AttributeValue::Strs(v) => Some(v.iter().map(|s| s.as_str()).collect()),
            _ => None,
        }
    }

    /// The first value as `f64`, when it is numeric.
    ///
    /// This converts. Match on the variant to keep the stored type.
    pub fn as_f64(&self) -> Option<f64> {
        self.as_f64s().and_then(|v| v.first().copied())
    }

    /// Every value as `f64`, when they are numeric.
    ///
    /// This converts. Match on the variant to keep the stored type.
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64s(&self) -> Option<Vec<f64>> {
        use AttributeValue as A;
        Some(match self {
            A::Uchar(v) => vec![*v as f64],
            A::Uchars(v) => v.iter().map(|&x| x as f64).collect(),
            A::Schar(v) => vec![*v as f64],
            A::Schars(v) => v.iter().map(|&x| x as f64).collect(),
            A::Ushort(v) => vec![*v as f64],
            A::Ushorts(v) => v.iter().map(|&x| x as f64).collect(),
            A::Short(v) => vec![*v as f64],
            A::Shorts(v) => v.iter().map(|&x| x as f64).collect(),
            A::Uint(v) => vec![*v as f64],
            A::Uints(v) => v.iter().map(|&x| x as f64).collect(),
            A::Int(v) => vec![*v as f64],
            A::Ints(v) => v.iter().map(|&x| x as f64).collect(),
            A::Ulonglong(v) => vec![*v as f64],
            A::Ulonglongs(v) => v.iter().map(|&x| x as f64).collect(),
            A::Longlong(v) => vec![*v as f64],
            A::Longlongs(v) => v.iter().map(|&x| x as f64).collect(),
            A::Float(v) => vec![*v as f64],
            A::Floats(v) => v.iter().map(|&x| x as f64).collect(),
            A::Double(v) => vec![*v],
            A::Doubles(v) => v.clone(),
            A::Str(_) | A::Strs(_) | A::Raw(_) => return None,
        })
    }

    /// The netCDF type of the value.
    pub fn vartype(&self) -> DType {
        use AttributeValue as A;
        match self {
            A::Uchar(_) | A::Uchars(_) => DType::Uint(1),
            A::Schar(_) | A::Schars(_) => DType::Int(1),
            A::Ushort(_) | A::Ushorts(_) => DType::Uint(2),
            A::Short(_) | A::Shorts(_) => DType::Int(2),
            A::Uint(_) | A::Uints(_) => DType::Uint(4),
            A::Int(_) | A::Ints(_) => DType::Int(4),
            A::Ulonglong(_) | A::Ulonglongs(_) => DType::Uint(8),
            A::Longlong(_) | A::Longlongs(_) => DType::Int(8),
            A::Float(_) | A::Floats(_) => DType::Float(4),
            A::Double(_) | A::Doubles(_) => DType::Float(8),
            A::Str(_) | A::Strs(_) => DType::String,
            A::Raw(_) => DType::Other,
        }
    }

    /// Number of values.
    pub fn len(&self) -> usize {
        use AttributeValue as A;
        match self {
            A::Uchars(v) => v.len(),
            A::Schars(v) => v.len(),
            A::Ushorts(v) => v.len(),
            A::Shorts(v) => v.len(),
            A::Uints(v) => v.len(),
            A::Ints(v) => v.len(),
            A::Ulonglongs(v) => v.len(),
            A::Longlongs(v) => v.len(),
            A::Floats(v) => v.len(),
            A::Doubles(v) => v.len(),
            A::Strs(v) => v.len(),
            A::Raw(v) => v.len(),
            _ => 1,
        }
    }

    /// Whether the value holds nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    /// [`oxcdf_hdf5::index::DatasetIndex::attributes_complete`].
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

/// Which container holds the file.
///
/// netCDF-4 sits in an HDF5 container. netCDF classic has its own. The
/// interface above this is the same either way.
// An open file is large either way. Boxing would add an indirection to every
// read to save a field that is allocated once per file.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
enum Backend {
    Hdf5(Hdf5File),
    Classic(crate::classic::ClassicFile),
}

/// An open netCDF file.
///
/// The file may be netCDF-4 or netCDF classic. [`NetcdfFile::open`] reads the
/// magic bytes and picks the container. Every method below behaves the same
/// either way.
///
/// For a netCDF-4 file the HDF5 index stays available through
/// [`NetcdfFile::hdf5`], for anything this layer does not model.
#[derive(Debug, Clone)]
pub struct NetcdfFile {
    backend: Backend,
    root: NcGroup,
}

impl NetcdfFile {
    /// Open a netCDF file from the filesystem with default options.
    ///
    /// This reads the magic bytes. A netCDF-4 file goes through HDF5. A classic
    /// file goes through the classic parser. The result is the same type.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_byte_source(Arc::new(oxcdf_hdf5::source::FileSource::open(path)?), None)
    }

    /// Open a netCDF file with explicit I/O and cache options.
    ///
    /// The options size the caches of the HDF5 engine. A classic file reads its
    /// whole header at open and has no cache to size, so it ignores them.
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
        options: oxcdf_hdf5::index::OpenOptions,
    ) -> Result<Self> {
        Self::from_byte_source(
            Arc::new(oxcdf_hdf5::source::FileSource::open(path)?),
            Some(options),
        )
    }

    /// Open a netCDF file over any byte source.
    pub fn from_source(source: Arc<dyn oxcdf_hdf5::source::ByteSource>) -> Result<Self> {
        Self::from_byte_source(source, None)
    }

    fn from_byte_source(
        source: Arc<dyn oxcdf_hdf5::source::ByteSource>,
        options: Option<oxcdf_hdf5::index::OpenOptions>,
    ) -> Result<Self> {
        match oxcdf_hdf5::detect_container(source.as_ref())? {
            oxcdf_hdf5::Container::Hdf5 => {
                let hdf5 = match options {
                    Some(o) => Hdf5File::from_source_with(source, o)?,
                    None => Hdf5File::from_source(source)?,
                };
                Self::from_hdf5(hdf5)
            }
            // The classic parser reads the whole header at open. It has no
            // cache to size, so the options do not apply.
            _ => Self::from_classic(crate::classic::ClassicFile::from_source(source)?),
        }
    }

    /// Interpret an already-open HDF5 file as netCDF-4.
    pub fn from_hdf5(hdf5: Hdf5File) -> Result<Self> {
        // Dimension scales are referenced by object header address from
        // anywhere in the file, so the scan has to cover the whole tree before
        // any variable can resolve its axes.
        let scales = collect_dimension_scales(&hdf5)?;
        let mut heaps = HeapCache::default();
        let root = build_group(&hdf5, hdf5.root(), &scales, &mut heaps)?;
        Ok(Self {
            backend: Backend::Hdf5(hdf5),
            root,
        })
    }

    /// Interpret an already-open classic file through the netCDF conventions.
    ///
    /// A classic file is flat. It has one group and no nested groups.
    pub fn from_classic(classic: crate::classic::ClassicFile) -> Result<Self> {
        let dimensions = classic
            .dimensions
            .iter()
            .map(|d| NcDimension {
                name: d.name.clone(),
                len: d.len,
                is_unlimited: d.is_unlimited,
                // Classic files list dimensions in order and store no id.
                id: None,
                has_coordinate_variable: classic.variables.iter().any(|v| v.name == d.name),
            })
            .collect();

        let variables = classic
            .variables
            .iter()
            .map(|v| NcVariable {
                name: v.name.clone(),
                path: format!("/{}", v.name),
                dimensions: v.dimensions.clone(),
                shape: v.shape.clone(),
                attributes: classic_attributes(&v.attributes),
                // The classic header lists every attribute, so nothing is lost.
                attributes_complete: true,
                is_coordinate: v.dimensions.len() == 1 && v.dimensions[0] == v.name,
            })
            .collect();

        Ok(Self {
            root: NcGroup {
                name: String::new(),
                path: "/".to_string(),
                dimensions,
                variables,
                attributes: classic_attributes(&classic.attributes),
                groups: Vec::new(),
            },
            backend: Backend::Classic(classic),
        })
    }

    /// The classic file behind this one, when the container is classic.
    pub fn classic(&self) -> Option<&crate::classic::ClassicFile> {
        match &self.backend {
            Backend::Classic(c) => Some(c),
            Backend::Hdf5(_) => None,
        }
    }

    /// Which container holds the file.
    pub fn container(&self) -> oxcdf_hdf5::Container {
        match &self.backend {
            Backend::Hdf5(_) => oxcdf_hdf5::Container::Hdf5,
            Backend::Classic(c) => c.container,
        }
    }

    /// The root group.
    pub fn root(&self) -> &NcGroup {
        &self.root
    }

    /// The HDF5 index behind this file, when the container is HDF5.
    ///
    /// `None` for a classic file, which has no HDF5 layer.
    pub fn hdf5(&self) -> Option<&Hdf5File> {
        match &self.backend {
            Backend::Hdf5(h) => Some(h),
            Backend::Classic(_) => None,
        }
    }

    /// Replace the underlying HDF5 file, keeping the netCDF view.
    ///
    /// Use this to attach caches after an open. The netCDF view describes
    /// names, shapes and axes. None of those depend on how bytes arrive. A
    /// classic file passes through unchanged.
    pub fn map_hdf5(mut self, f: impl FnOnce(Hdf5File) -> Hdf5File) -> Self {
        if let Backend::Hdf5(h) = self.backend {
            self.backend = Backend::Hdf5(f(h));
        }
        self
    }

    /// The HDF5 dataset behind a variable, for the storage this layer does not
    /// model. `None` for a classic file.
    pub fn dataset(&self, variable: &NcVariable) -> Option<&DatasetIndex> {
        self.hdf5()?.dataset(&variable.path)
    }

    pub(crate) fn variable_info(&self, path: &str) -> Option<&NcVariable> {
        let trimmed = path.trim_start_matches('/');
        let mut parts: Vec<&str> = trimmed.split('/').collect();
        let leaf = parts.pop()?;
        let mut group = &self.root;
        for part in parts {
            group = group.groups.iter().find(|g| g.name == part)?;
        }
        group.variable(leaf)
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

        variables.push(build_variable(
            hdf5,
            dataset,
            scales,
            heaps,
            scale.is_some(),
        )?);
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
        attributes: visible_attributes(hdf5, &group.path, &group.attributes),
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
        attributes: visible_attributes(hdf5, &dataset.path, &dataset.attributes),
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
                    return Ok(ids.iter().map(|id| by_id[id].name.clone()).collect());
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
        let descriptor = VlenDescriptor::parse(&attr.data[start..start + descriptor_len], sizes)?;

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

/// Decode a classic file's attributes.
///
/// Each one becomes the same [`Attribute`] an HDF5 file would hold, so the
/// value goes through one decoder and keeps its type. A classic file has no
/// reserved attribute names and no variable-length values.
fn classic_attributes(attributes: &[crate::classic::ClassicAttribute]) -> Vec<NcAttribute> {
    attributes
        .iter()
        .map(|a| {
            let width = a.nc_type.size().max(1);
            let attr = Attribute {
                name: a.name.clone(),
                datatype: a.nc_type.to_datatype(),
                dataspace: Dataspace {
                    kind: oxcdf_hdf5::message::DataspaceKind::Simple,
                    dims: vec![(a.raw.len() / width) as u64],
                    max_dims: None,
                },
                data: a.raw.clone(),
            };
            NcAttribute {
                name: a.name.clone(),
                value: decode_value(None, &a.name, &attr),
            }
        })
        .collect()
}

/// Drop the netCDF bookkeeping attributes and decode the rest.
fn visible_attributes(hdf5: &Hdf5File, owner: &str, attributes: &[Attribute]) -> Vec<NcAttribute> {
    attributes
        .iter()
        .filter(|a| !RESERVED_ATTRIBUTES.contains(&a.name.as_str()))
        .map(|a| NcAttribute {
            name: a.name.clone(),
            value: decode_value(Some(hdf5), owner, a),
        })
        .collect()
}

/// Decode an attribute's bytes according to its datatype.
///
/// The value keeps the stored type. One value gets the singular variant.
/// Several values get the plural one. The `netcdf` crate does the same.
///
/// A type this reader cannot decode stays as [`AttributeValue::Raw`]. The
/// attribute is then still visible.
fn decode_value(hdf5: Option<&Hdf5File>, owner: &str, attr: &Attribute) -> AttributeValue {
    decode_typed(hdf5, owner, attr).unwrap_or_else(|| AttributeValue::Raw(attr.data.clone()))
}

fn decode_typed(hdf5: Option<&Hdf5File>, owner: &str, attr: &Attribute) -> Option<AttributeValue> {
    use AttributeValue as A;

    /// One value stays singular. Several become a list.
    fn fold<T, S, P>(mut values: Vec<T>, single: S, plural: P) -> AttributeValue
    where
        S: FnOnce(T) -> AttributeValue,
        P: FnOnce(Vec<T>) -> AttributeValue,
    {
        if values.len() == 1 {
            single(values.pop().expect("length checked"))
        } else {
            plural(values)
        }
    }

    let size = attr.datatype.size;
    match &attr.datatype.class {
        // A netCDF `char` attribute is always one string, however many
        // elements the dataspace holds. The `netcdf` crate reads the whole
        // buffer as one text, so join the parts rather than fold on count.
        DatatypeClass::String { .. } => Some(A::Str(decode_all_text(attr)?.concat())),
        DatatypeClass::VariableLength {
            kind: oxcdf_hdf5::message::VlenKind::String,
            ..
        } => {
            // The value is a descriptor into the global heap, so follow it.
            // Without this the attribute would surface as raw heap pointers.
            let raw = RawData {
                bytes: attr.data.clone(),
                element_size: size as usize,
                shape: vec![attr.element_count()],
            };
            // A netCDF `string` attribute is always plural, even with one
            // value. The `netcdf` crate does the same.
            let text = oxcdf_hdf5::read::resolve_vlen_strings_of(hdf5?.ctx(), owner, &raw).ok()?;
            Some(A::Strs(text))
        }
        DatatypeClass::FloatingPoint { .. } => {
            let values = decode_floats(attr).ok()?;
            match size {
                4 => Some(fold(
                    values.into_iter().map(|v| v as f32).collect(),
                    A::Float,
                    A::Floats,
                )),
                8 => Some(fold(values, A::Double, A::Doubles)),
                _ => None,
            }
        }
        DatatypeClass::FixedPoint { signed, .. } => {
            let values = decode_ints(attr).ok()?;
            Some(match (signed, size) {
                (true, 1) => fold(
                    values.into_iter().map(|v| v as i8).collect(),
                    A::Schar,
                    A::Schars,
                ),
                (true, 2) => fold(
                    values.into_iter().map(|v| v as i16).collect(),
                    A::Short,
                    A::Shorts,
                ),
                (true, 4) => fold(
                    values.into_iter().map(|v| v as i32).collect(),
                    A::Int,
                    A::Ints,
                ),
                (true, 8) => fold(values, A::Longlong, A::Longlongs),
                (false, 1) => fold(
                    values.into_iter().map(|v| v as u8).collect(),
                    A::Uchar,
                    A::Uchars,
                ),
                (false, 2) => fold(
                    values.into_iter().map(|v| v as u16).collect(),
                    A::Ushort,
                    A::Ushorts,
                ),
                (false, 4) => fold(
                    values.into_iter().map(|v| v as u32).collect(),
                    A::Uint,
                    A::Uints,
                ),
                (false, 8) => fold(
                    values.into_iter().map(|v| v as u64).collect(),
                    A::Ulonglong,
                    A::Ulonglongs,
                ),
                _ => return None,
            })
        }
        _ => None,
    }
}

/// Whether an attribute's values need their bytes reversed.
fn needs_swap(attr: &Attribute) -> bool {
    matches!(
        attr.datatype.byte_order(),
        Some(oxcdf_hdf5::message::ByteOrder::Big)
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
                raw[..z]
                    .iter()
                    .rposition(|&b| b != b' ')
                    .map(|p| p + 1)
                    .unwrap_or(0)
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
            "/../../test_files/legacy_v1_objheader.h5"
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
            "/../../test_files/legacy_v1_objheader.h5"
        );
        let file = NetcdfFile::open(path).unwrap();
        let v = file.variable("/chunked_i32").unwrap();
        let values = v.get_values::<i64, _>(..).unwrap();

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
        let text = AttributeValue::Str("degrees_north".into());
        assert_eq!(text.as_text(), Some("degrees_north"));
        assert_eq!(text.as_f64(), None);
        assert_eq!(text.vartype(), DType::String);

        let nums = AttributeValue::Floats(vec![1.5, 2.5]);
        assert_eq!(nums.as_f64(), Some(1.5));
        assert_eq!(nums.as_f64s(), Some(vec![1.5, 2.5]));
        assert_eq!(nums.as_text(), None);
        assert_eq!(nums.vartype(), DType::Float(4));
        assert_eq!(nums.len(), 2);

        let ints = AttributeValue::Int(7);
        assert_eq!(ints.as_f64(), Some(7.0));
        assert_eq!(ints.vartype(), DType::Int(4));
        assert_eq!(ints.len(), 1);

        // An unsigned 64-bit value above `i64::MAX` must survive decoding.
        let big = AttributeValue::Ulonglong(u64::MAX);
        assert_eq!(big.as_f64(), Some(u64::MAX as f64));
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

/// Values read from a variable, in native byte order and row-major order.
///
/// This is the internal carrier between the read path and the `get_*` methods.
/// It is not public: the interface mirrors the `netcdf` crate, which returns
/// plain vectors.
#[derive(Debug, Clone)]
pub(crate) struct Values {
    raw: RawData,
    datatype: oxcdf_hdf5::message::Datatype,
    /// Already-resolved variable-length strings, when the variable holds them.
    ///
    /// These are resolved at read time because the characters live in the
    /// global heap, which needs the file.
    vlen_strings: Option<Vec<String>>,
}

impl Values {
    /// Build values that hold no heap reference.
    ///
    /// A classic file has no variable-length type, so nothing needs to be
    /// followed into a heap.
    pub(crate) fn from_parts(raw: RawData, datatype: oxcdf_hdf5::message::Datatype) -> Self {
        Self {
            raw,
            datatype,
            vlen_strings: None,
        }
    }

    /// Shape of the block that was read.
    ///
    /// Only an `ndarray` result needs it; a flat read returns the values alone.
    #[cfg(feature = "ndarray")]
    pub fn shape(&self) -> &[u64] {
        &self.raw.shape
    }

    /// The raw block, giving up the datatype.
    pub fn into_raw(self) -> RawData {
        self.raw
    }

    /// Values as `T`.
    ///
    /// A `T` equal to the stored type copies the values. Any other numeric
    /// type converts, which is what the `netcdf` crate does. See [`Element`]
    /// for what a conversion can lose.
    pub fn get<T: Element>(&self) -> Result<Vec<T>> {
        self.raw.get_of(&self.datatype, "")
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

    /// The values as an `ndarray` of `T`, shaped as they were read.
    ///
    /// Row-major, so the array's axes match the variable's dimensions in order.
    #[cfg(feature = "ndarray")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
    pub fn to_array<T: Element>(&self) -> Result<ndarray::ArrayD<T>> {
        shape_into_array(self.shape(), self.get::<T>()?)
    }
}

/// Wrap a flat, row-major vector in an `ndarray` of the given shape.
#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
fn shape_into_array<T>(shape: &[u64], values: Vec<T>) -> Result<ndarray::ArrayD<T>> {
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let expected: usize = dims
        .iter()
        .product::<usize>()
        .max(if dims.is_empty() { 1 } else { 0 });
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
    store: VarStore<'a>,
}

/// Where a variable's values live.
#[derive(Clone, Copy, Debug)]
pub(crate) enum VarStore<'a> {
    Hdf5(&'a DatasetIndex),
    Classic(&'a crate::classic::ClassicVariable),
}

impl<'a> VarStore<'a> {
    /// The element type, whichever container holds it.
    pub(crate) fn datatype(&self) -> &'a oxcdf_hdf5::message::Datatype {
        match self {
            VarStore::Hdf5(d) => &d.datatype,
            VarStore::Classic(v) => &v.datatype,
        }
    }

    pub(crate) fn hdf5(&self) -> Option<&'a DatasetIndex> {
        match self {
            VarStore::Hdf5(d) => Some(d),
            VarStore::Classic(_) => None,
        }
    }
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

    /// The HDF5 dataset backing this variable. `None` for a classic file.
    pub fn dataset(&self) -> Option<&'a DatasetIndex> {
        self.store.hdf5()
    }

    /// The full element type, whichever container holds it.
    pub fn datatype(&self) -> &'a oxcdf_hdf5::message::Datatype {
        self.store.datatype()
    }

    /// The variable's netCDF type. This matches `netcdf::Variable::vartype`.
    pub fn vartype(&self) -> DType {
        DType::of(self.store.datatype())
    }

    /// The variable's attributes. This matches `netcdf::Variable::attributes`.
    pub fn attributes(&self) -> &'a [NcAttribute] {
        &self.info.attributes
    }

    /// One attribute by name. This matches `netcdf::Variable::attribute`.
    pub fn attribute(&self, name: &str) -> Option<&'a NcAttribute> {
        self.info.attributes.iter().find(|a| a.name == name)
    }

    /// Total number of elements. This matches `netcdf::Variable::len`.
    pub fn len(&self) -> u64 {
        self.info.shape.iter().product()
    }

    /// Whether the variable holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this reader can decode the variable's values.
    ///
    /// Check this before reading if netcdf-c is kept as a fallback.
    pub fn is_readable(&self) -> bool {
        // A classic file has no filters and no exotic types, so every variable
        // in one is readable.
        self.store.hdf5().is_none_or(|d| d.is_readable())
    }

    /// Read values as `T`, over any selection.
    ///
    /// This matches `netcdf::Variable::get_values`. `extents` accepts the same
    /// forms: [`Extents::All`], `..`, ranges, indices, or a start and count
    /// pair. See [`crate::extent`].
    ///
    /// ```no_run
    /// # use oxcdf::Extents;
    /// # fn run(var: oxcdf::Variable<'_>) -> oxcdf::Result<()> {
    /// let all = var.get_values::<f32, _>(Extents::All)?;
    /// let block = var.get_values::<f32, _>([0..8, 10..30])?;
    /// let row = var.get_values::<f32, _>([3, 0])?;
    /// # Ok(()) }
    /// ```
    ///
    /// A `T` equal to [`Variable::vartype`] copies the values. Any other numeric
    /// type converts. See [`Element`].
    pub fn get_values<T: Element, E>(&self, extents: E) -> Result<Vec<T>>
    where
        E: TryInto<Extents>,
        E::Error: Into<Error>,
    {
        let extents: Extents = extents.try_into().map_err(Into::into)?;
        let slab = extents.to_hyperslab(&self.info.path, &self.info.shape)?;
        self.read_selection(&slab)?.get()
    }

    /// Read one value as `T`.
    ///
    /// This matches `netcdf::Variable::get_value`. The selection must name one
    /// element.
    pub fn get_value<T: Element, E>(&self, extents: E) -> Result<T>
    where
        E: TryInto<Extents>,
        E::Error: Into<Error>,
    {
        let values = self.get_values::<T, E>(extents)?;
        one_value(&self.info.path, values)
    }

    /// Read strings, over any selection.
    ///
    /// This matches `netcdf::Variable::get_strings`. It takes the same
    /// selection forms as [`Variable::get_values`].
    ///
    /// A `string` variable holds one string in each element, so the result has
    /// one string for each element the selection names.
    ///
    /// A `char` variable holds one **byte** in each element. Its last dimension
    /// is the string length, so this returns one string for each character.
    /// Join them yourself, or read the bytes with [`Variable::get_raw_values`].
    /// The reader reports the elements as the file stores them.
    ///
    /// ```no_run
    /// # fn run(var: oxcdf::Variable<'_>) -> oxcdf::Result<()> {
    /// let names = var.get_strings(..)?;
    /// let some = var.get_strings([0..4])?;
    /// # Ok(()) }
    /// ```
    pub fn get_strings<E>(&self, extents: E) -> Result<Vec<String>>
    where
        E: TryInto<Extents>,
        E::Error: Into<Error>,
    {
        let extents: Extents = extents.try_into().map_err(Into::into)?;
        let slab = extents.to_hyperslab(&self.info.path, &self.info.shape)?;
        self.read_selection(&slab)?.to_strings()
    }

    /// Read one string.
    ///
    /// This matches `netcdf::Variable::get_string`. The selection must name one
    /// element.
    pub fn get_string<E>(&self, extents: E) -> Result<String>
    where
        E: TryInto<Extents>,
        E::Error: Into<Error>,
    {
        let strings = self.get_strings(extents)?;
        one_value(&self.info.path, strings)
    }

    /// Read values as an `ndarray` of `T`, over any selection.
    ///
    /// This matches `netcdf::Variable::get`. The array's shape is the
    /// selection's shape, and its axes follow the variable's dimensions.
    #[cfg(feature = "ndarray")]
    #[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
    pub fn get<T: Element, E>(&self, extents: E) -> Result<ndarray::ArrayD<T>>
    where
        E: TryInto<Extents>,
        E::Error: Into<Error>,
    {
        let extents: Extents = extents.try_into().map_err(Into::into)?;
        let slab = extents.to_hyperslab(&self.info.path, &self.info.shape)?;
        self.read_selection(&slab)?.to_array()
    }

    /// Read the raw bytes of a selection, in native order and row-major.
    ///
    /// This matches `netcdf::Variable::get_raw_values`. Use it for a `char`
    /// variable, or to build your own typed array without a copy through
    /// [`Vec`].
    pub fn get_raw_values<E>(&self, extents: E) -> Result<Vec<u8>>
    where
        E: TryInto<Extents>,
        E::Error: Into<Error>,
    {
        let extents: Extents = extents.try_into().map_err(Into::into)?;
        let slab = extents.to_hyperslab(&self.info.path, &self.info.shape)?;
        Ok(self.read_selection(&slab)?.into_raw().bytes)
    }

    /// Read an explicit selection.
    ///
    /// This is the engine behind every `get_*` method. It is private because
    /// the public interface mirrors the `netcdf` crate, which has no such
    /// call.
    pub(crate) fn read_selection(&self, slab: &Hyperslab) -> Result<Values> {
        match self.store {
            VarStore::Hdf5(dataset) => {
                let ctx = self
                    .file
                    .hdf5()
                    .ok_or_else(|| Error::malformed("an HDF5 variable without an HDF5 file"))?
                    .ctx();
                let raw = read_hyperslab(ctx, dataset, slab)?;
                values_from_raw(ctx, dataset, raw)
            }
            VarStore::Classic(variable) => {
                let classic = self
                    .file
                    .classic()
                    .ok_or_else(|| Error::malformed("a classic variable without a classic file"))?;
                slab.validate(&self.info.shape)?;
                // `read_selection` returns native order already, so the bytes
                // arrive exactly as the HDF5 path leaves them.
                let bytes = classic.read_selection(variable, slab)?;
                let width = variable.nc_type.size();

                Ok(Values::from_parts(
                    RawData {
                        bytes,
                        element_size: width,
                        shape: slab.count.clone(),
                    },
                    variable.datatype.clone(),
                ))
            }
        }
    }

    /// The shape of one storage chunk, when the variable is chunked.
    ///
    /// This matches `netcdf::Variable::chunking`. `None` means the variable is
    /// stored contiguously and has no chunk grid.
    pub fn chunking(&self) -> Result<Option<Vec<usize>>> {
        let Some(dataset) = self.store.hdf5() else {
            // A classic file has no chunking.
            return Ok(None);
        };
        Ok(match &dataset.layout {
            Layout::Chunked { chunk_dims, .. } => {
                Some(chunk_dims.iter().map(|&d| d as usize).collect())
            }
            _ => None,
        })
    }

    /// The variable's fill value, as `T`.
    ///
    /// This matches `netcdf::Variable::fill_value`. `None` means the variable
    /// declares none.
    pub fn fill_value<T: Element>(&self) -> Result<Option<T>> {
        fill_value_of(self.attribute("_FillValue"))
    }
}

/// Take the one value a selection named, or say how many it named instead.
pub(crate) fn one_value<T>(what: &str, mut values: Vec<T>) -> Result<T> {
    match values.len() {
        1 => Ok(values.pop().expect("length checked")),
        n => Err(Error::bad_request(format!(
            "the selection on variable {what} names {n} elements, but one was asked for"
        ))),
    }
}

/// A `_FillValue` attribute as `T`, for both variable types.
pub(crate) fn fill_value_of<T: Element>(attribute: Option<&NcAttribute>) -> Result<Option<T>> {
    Ok(attribute.and_then(|a| scalar_from_attribute(&a.value)))
}

/// One numeric attribute value, as `T`.
///
/// Each variant converts from the widest type that holds it exactly, so an
/// `i64` fill value does not detour through `f64`.
fn scalar_from_attribute<T: Element>(value: &AttributeValue) -> Option<T> {
    use AttributeValue as A;
    Some(match value {
        A::Uchar(v) => T::from_u64(*v as u64),
        A::Uchars(v) => T::from_u64(*v.first()? as u64),
        A::Schar(v) => T::from_i64(*v as i64),
        A::Schars(v) => T::from_i64(*v.first()? as i64),
        A::Ushort(v) => T::from_u64(*v as u64),
        A::Ushorts(v) => T::from_u64(*v.first()? as u64),
        A::Short(v) => T::from_i64(*v as i64),
        A::Shorts(v) => T::from_i64(*v.first()? as i64),
        A::Uint(v) => T::from_u64(*v as u64),
        A::Uints(v) => T::from_u64(*v.first()? as u64),
        A::Int(v) => T::from_i64(*v as i64),
        A::Ints(v) => T::from_i64(*v.first()? as i64),
        A::Ulonglong(v) => T::from_u64(*v),
        A::Ulonglongs(v) => T::from_u64(*v.first()?),
        A::Longlong(v) => T::from_i64(*v),
        A::Longlongs(v) => T::from_i64(*v.first()?),
        A::Float(v) => T::from_f64(*v as f64),
        A::Floats(v) => T::from_f64(*v.first()? as f64),
        A::Double(v) => T::from_f64(*v),
        A::Doubles(v) => T::from_f64(*v.first()?),
        A::Str(_) | A::Strs(_) | A::Raw(_) => return None,
    })
}

/// Turn raw bytes into values, following any heap pointer they hold.
///
/// A variable-length string stores a pointer into the global heap. The value
/// is only complete after the reader follows that pointer.
///
/// The read happens here, while the file is still to hand. Both engines use
/// this function.
pub(crate) fn values_from_raw(
    ctx: oxcdf_hdf5::context::Ctx<'_>,
    dataset: &DatasetIndex,
    raw: RawData,
) -> Result<Values> {
    let mut vlen_strings = None;

    if let DatatypeClass::VariableLength { kind, .. } = &dataset.datatype.class {
        // A ragged sequence has no netCDF-style reader, so it is not followed
        // here. `oxcdf_hdf5::read::resolve_vlen_sequences` still reads one for
        // a caller working at the HDF5 layer.
        if matches!(kind, oxcdf_hdf5::message::VlenKind::String) {
            vlen_strings = Some(oxcdf_hdf5::read::resolve_vlen_strings(ctx, dataset, &raw)?);
        }
    }

    Ok(Values {
        raw,
        datatype: dataset.datatype.clone(),
        vlen_strings,
    })
}

impl NetcdfFile {
    /// A variable by absolute path, bound to the file so it can be read.
    pub fn variable(&self, path: &str) -> Option<Variable<'_>> {
        let info = self.variable_info(path)?;
        Some(Variable {
            file: self,
            info,
            store: self.store_for(info)?,
        })
    }

    pub(crate) fn store_for(&self, info: &NcVariable) -> Option<VarStore<'_>> {
        match &self.backend {
            Backend::Hdf5(h) => h.dataset(&info.path).map(VarStore::Hdf5),
            Backend::Classic(c) => c
                .variables
                .iter()
                .find(|v| v.name == info.name)
                .map(VarStore::Classic),
        }
    }

    /// Every variable in the file, bound for reading.
    pub fn variables(&self) -> Vec<Variable<'_>> {
        self.root
            .variables_recursive()
            .into_iter()
            .filter_map(|info| {
                self.store_for(info).map(|store| Variable {
                    file: self,
                    info,
                    store,
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

    /// One dimension of the root group by name.
    pub fn dimension(&self, name: &str) -> Option<&NcDimension> {
        self.root.dimension(name)
    }

    /// The length of one dimension of the root group.
    pub fn dimension_len(&self, name: &str) -> Option<u64> {
        self.dimension(name).map(|d| d.len)
    }

    /// The groups directly inside the root group.
    pub fn groups(&self) -> &[NcGroup] {
        &self.root.groups
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
