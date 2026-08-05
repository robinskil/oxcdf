//! The immutable file index.
//!
//! An open walks the file metadata once. It builds the structures here. Nothing
//! here changes afterwards. Nothing here holds the file. Wrap an index in
//! [`std::sync::Arc`] and share it across threads.
//!
//! A read is then a pure function of the index and a request.
//!
//! # Lazy chunk indexes
//!
//! A chunk index resolves on first use. An open does not walk it. A walk costs
//! several dependent reads. A query reads a few of a file's variables.
//!
//! Call [`DatasetIndex::prepare`] to resolve an index early. A caller that
//! knows its projection prepares only the variables it reads.

use std::collections::HashSet;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::filters;
use crate::hdf5::btree1;
use crate::hdf5::context::Ctx;
use crate::hdf5::heap::LocalHeap;
use crate::hdf5::message::{
    Attribute, Datatype, Dataspace, FillValue, FilterPipeline, Layout, LinkTarget,
};
use crate::hdf5::objheader::ObjectHeader;
use crate::hdf5::superblock::Superblock;
use crate::hdf5::symbol_table::SymbolTableNode;
use crate::source::{ByteSource, FileSource};

/// How to open a file: I/O request size, cache sizes and range merging.
///
/// These matter most on object storage, where a request costs milliseconds
/// regardless of size. The defaults are tuned for local files.
///
/// ```no_run
/// # use oxcdf::index::{Hdf5File, OpenOptions};
/// let file = Hdf5File::open_with(
///     "argo.nc",
///     OpenOptions::new()
///         .io_request_size(256 * 1024)   // fetch 256 KiB per miss
///         .io_cache_bytes(128 << 20),    // keep 128 MiB of raw bytes
/// )?;
/// # Ok::<(), oxcdf::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct OpenOptions {
    io: crate::io::IoConfig,
    page_size: usize,
    io_cache_bytes: Option<usize>,
    chunk_cache_capacity: Option<u64>,
    readahead: usize,
    #[cfg(feature = "async")]
    open_prefetch: usize,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            io: crate::io::IoConfig::default(),
            page_size: crate::cache::DEFAULT_PAGE_SIZE,
            io_cache_bytes: Some(
                crate::cache::DEFAULT_PAGE_CAPACITY as usize * crate::cache::DEFAULT_PAGE_SIZE,
            ),
            chunk_cache_capacity: Some(crate::cache::DEFAULT_CAPACITY),
            readahead: crate::cache::DEFAULT_READAHEAD,
            #[cfg(feature = "async")]
            open_prefetch: crate::replay::DEFAULT_PREFETCH_BYTES,
        }
    }
}

impl OpenOptions {
    /// Default options: tuned for local files.
    pub fn new() -> Self {
        Self::default()
    }

    /// Options tuned for object storage: 256 KiB requests, 128 MiB of raw bytes
    /// cached, and aggressive range merging.
    pub fn remote() -> Self {
        Self::new()
            .io_request_size(256 * 1024)
            .io_cache_bytes(128 << 20)
            .io_config(crate::io::IoConfig::REMOTE)
    }

    /// Bytes fetched per cache miss. This is the I/O request size.
    pub fn io_request_size(mut self, bytes: usize) -> Self {
        self.page_size = bytes.max(1);
        self
    }

    /// Roughly how many bytes of raw file data to keep cached.
    pub fn io_cache_bytes(mut self, bytes: usize) -> Self {
        self.io_cache_bytes = Some(bytes);
        self
    }

    /// Serve every read straight from the source.
    pub fn without_io_cache(mut self) -> Self {
        self.io_cache_bytes = None;
        self
    }

    /// How many decoded chunks to keep.
    pub fn chunk_cache_capacity(mut self, chunks: u64) -> Self {
        self.chunk_cache_capacity = Some(chunks);
        self
    }

    /// Decode every chunk on every read.
    pub fn without_chunk_cache(mut self) -> Self {
        self.chunk_cache_capacity = None;
        self
    }

    /// How many chunks past a read's own to prefetch. Zero disables it.
    pub fn readahead(mut self, chunks: usize) -> Self {
        self.readahead = chunks;
        self
    }

    /// How aggressively to merge neighbouring byte-range requests.
    pub fn io_config(mut self, io: crate::io::IoConfig) -> Self {
        self.io = io;
        self
    }

    /// How many bytes an asynchronous open fetches before its first walk.
    ///
    /// netCDF writes its metadata near the front of the file. A window this
    /// size normally covers a whole open in one request. A window that is too
    /// small still works. It costs more round trips.
    #[cfg(feature = "async")]
    #[cfg_attr(docsrs, doc(cfg(feature = "async")))]
    pub fn open_prefetch_bytes(mut self, bytes: usize) -> Self {
        self.open_prefetch = bytes.max(1);
        self
    }

    /// The I/O request size, which is also the byte cache's page size.
    pub fn request_size(&self) -> usize {
        self.page_size
    }

    /// The window an asynchronous open fetches before its first walk.
    #[cfg(feature = "async")]
    #[cfg_attr(docsrs, doc(cfg(feature = "async")))]
    pub fn prefetch_bytes(&self) -> usize {
        self.open_prefetch
    }

    /// The byte-range merging policy in use.
    pub fn merge_policy(&self) -> crate::io::IoConfig {
        self.io
    }

    /// Build the byte cache these options describe.
    pub fn build_io_cache(&self) -> Option<crate::cache::IoCache> {
        self.io_cache_bytes
            .map(|bytes| crate::cache::IoCache::with_capacity_bytes(bytes, self.page_size))
    }

    /// Build the decoded-chunk cache these options describe.
    pub fn build_chunk_cache(&self) -> Option<crate::cache::ChunkCache> {
        self.chunk_cache_capacity
            .map(|n| crate::cache::ChunkCache::new(n).with_readahead(self.readahead))
    }
}

/// Guard against a link cycle in a damaged or hostile file.
const MAX_DEPTH: usize = 64;

/// One dataset, fully described and ready to read.
#[derive(Debug, Clone)]
pub struct DatasetIndex {
    /// Name within its parent group.
    pub name: String,
    /// Full path from the root group.
    pub path: String,
    /// Address of the dataset's object header.
    pub address: u64,
    /// Shape in elements.
    pub shape: Vec<u64>,
    /// Maximum shape, when the dataset records one. `u64::MAX` on an axis marks
    /// it unlimited, which is how netCDF stores an unlimited dimension.
    pub max_shape: Option<Vec<u64>>,
    /// Element type.
    pub datatype: Datatype,
    /// Storage layout.
    pub layout: Layout,
    /// Filters applied to each chunk.
    pub pipeline: FilterPipeline,
    /// What storage that was never written reads as.
    pub fill_value: FillValue,
    /// Every chunk, when the dataset is chunked. Sorted by position.
    ///
    /// Resolved lazily: walking a chunk index costs several dependent reads,
    /// and a query normally touches a handful of a file's variables. Opening a
    /// file with 55 chunked variables used to walk 55 B-trees regardless.
    chunks: std::sync::OnceLock<Option<Vec<btree1::ChunkRecord>>>,
    /// The dataset's attributes.
    pub attributes: Vec<Attribute>,
    /// Whether every attribute was found.
    ///
    /// False when the object's attribute heap has internal free space, which
    /// hides records from a sequential walk. The values present are correct;
    /// the list may be short. Fall back to netcdf-c for this object's
    /// attributes when that matters.
    pub attributes_complete: bool,
}

impl DatasetIndex {
    /// Number of elements in the whole dataset.
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Width of one element in bytes.
    pub fn element_size(&self) -> usize {
        self.datatype.size as usize
    }

    /// Rank of the dataset.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// An attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Whether this reader can decode the dataset's values.
    ///
    /// Cheap: it looks only at the datatype and the filter pipeline, so it can
    /// be called at plan time over every variable without touching the file. A
    /// chunk index this reader cannot walk surfaces later, from
    /// [`DatasetIndex::prepare`] or the first read.
    pub fn is_readable(&self) -> bool {
        self.datatype.is_decodable() && filters::pipeline_is_supported(&self.pipeline)
    }

    /// The chunk index, resolving it on first use.
    ///
    /// `None` means the dataset is not chunked. Two callers racing a cold index
    /// may both walk it; the walk is pure, so the duplicate is wasted work
    /// rather than a correctness problem, and it avoids a lock.
    pub fn chunks(&self, ctx: Ctx<'_>) -> Result<Option<&[btree1::ChunkRecord]>> {
        if let Some(resolved) = self.chunks.get() {
            return Ok(resolved.as_deref());
        }
        let resolved = self.resolve_chunks(ctx)?;
        let _ = self.chunks.set(resolved);
        Ok(self
            .chunks
            .get()
            .expect("just set")
            .as_deref())
    }

    /// The chunk index if it has already been resolved, without touching the file.
    pub fn resolved_chunks(&self) -> Option<&[btree1::ChunkRecord]> {
        self.chunks.get().and_then(|c| c.as_deref())
    }

    /// Resolve the chunk index now, so later reads do no metadata input.
    ///
    /// This is an optimisation. A read resolves the index itself. A caller that
    /// knows its projection prepares exactly the variables it reads, which
    /// moves that cost off the read path.
    pub fn prepare(&self, ctx: Ctx<'_>) -> Result<()> {
        self.chunks(ctx).map(|_| ())
    }

    /// Whether the chunk index has been resolved.
    pub fn is_prepared(&self) -> bool {
        self.chunks.get().is_some()
    }

    fn resolve_chunks(&self, ctx: Ctx<'_>) -> Result<Option<Vec<btree1::ChunkRecord>>> {
        match &self.layout {
            Layout::Chunked {
                address,
                chunk_dims,
                index,
                element_size,
            } => match index {
                crate::hdf5::message::ChunkIndex::BtreeV1 => match address {
                    None => Ok(Some(Vec::new())),
                    Some(a) => Ok(Some(btree1::read_chunk_index(ctx, *a, chunk_dims.len())?)),
                },
                other => {
                    // A version 4 layout does not repeat the element size, so it
                    // comes from the datatype instead.
                    let width = if *element_size == 0 {
                        self.datatype.size as usize
                    } else {
                        *element_size as usize
                    };
                    let dims: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();
                    Ok(Some(crate::hdf5::chunk_index::read(
                        ctx,
                        other,
                        *address,
                        &self.shape,
                        &dims,
                        width,
                        !self.pipeline.is_empty(),
                    )?))
                }
            },
            _ => Ok(None),
        }
    }
}

/// One group, with its children already resolved.
#[derive(Debug, Clone, Default)]
pub struct GroupIndex {
    /// Name within its parent. Empty for the root group.
    pub name: String,
    /// Full path from the root group. `/` for the root.
    pub path: String,
    /// Address of the group's object header.
    pub address: u64,
    /// The group's attributes.
    pub attributes: Vec<Attribute>,
    /// Whether every attribute was found. See [`DatasetIndex::attributes_complete`].
    pub attributes_complete: bool,
    /// Child groups.
    pub groups: Vec<GroupIndex>,
    /// Child datasets.
    pub datasets: Vec<DatasetIndex>,
}

impl GroupIndex {
    /// A child group by name.
    pub fn group(&self, name: &str) -> Option<&GroupIndex> {
        self.groups.iter().find(|g| g.name == name)
    }

    /// A child dataset by name.
    pub fn dataset(&self, name: &str) -> Option<&DatasetIndex> {
        self.datasets.iter().find(|d| d.name == name)
    }

    /// An attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Every dataset at or below this group, depth first.
    pub fn datasets_recursive(&self) -> Vec<&DatasetIndex> {
        let mut out: Vec<&DatasetIndex> = self.datasets.iter().collect();
        for g in &self.groups {
            out.extend(g.datasets_recursive());
        }
        out
    }
}

/// An open HDF5 file: the bytes plus the index built from them.
///
/// This type is `Send + Sync`. Clone it freely, or share one behind an
/// [`std::sync::Arc`]; concurrent reads need no coordination.
#[derive(Debug, Clone)]
pub struct Hdf5File {
    source: Arc<dyn ByteSource>,
    superblock: Superblock,
    root: GroupIndex,
    /// Decoded chunks, shared by every read of this file.
    cache: Option<crate::cache::ChunkCache>,
    /// How aggressively to merge byte-range requests.
    io: crate::io::IoConfig,
    /// Raw byte cache, shared by every read of this file.
    io_cache: Option<crate::cache::IoCache>,
}

impl Hdf5File {
    /// Open a file from the filesystem with default options.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_source(Arc::new(FileSource::open(path)?))
    }

    /// Open a file from the filesystem with explicit options.
    ///
    /// The options apply to the open itself, not just to later reads: indexing
    /// is many small clustered reads, which is exactly what the page cache and
    /// the request size govern.
    pub fn open_with(path: impl AsRef<std::path::Path>, options: OpenOptions) -> Result<Self> {
        Self::from_source_with(Arc::new(FileSource::open(path)?), options)
    }

    /// Open a file over any byte source with default options.
    pub fn from_source(source: Arc<dyn ByteSource>) -> Result<Self> {
        Self::from_source_with(source, OpenOptions::default())
    }

    /// Open a file over any byte source with explicit options.
    pub fn from_source_with(source: Arc<dyn ByteSource>, options: OpenOptions) -> Result<Self> {
        let io_cache = options.build_io_cache();
        Self::from_source_reusing(source, &options, io_cache)
    }

    /// Open a file over any byte source, reusing a byte cache that already
    /// holds some of its pages.
    ///
    /// The asynchronous open uses this. It fetches the metadata pages itself,
    /// then hands them straight to the file.
    pub fn from_source_reusing(
        source: Arc<dyn ByteSource>,
        options: &OpenOptions,
        io_cache: Option<crate::cache::IoCache>,
    ) -> Result<Self> {
        let superblock = Superblock::read(source.as_ref())?;
        let root = {
            // Indexing is many small reads over a small region: exactly what the
            // page cache is for. The decoded-chunk cache stays out of it.
            let ctx = Ctx::new(source.as_ref(), &superblock).with_io_cache(io_cache.as_ref());
            let address = superblock.root_object_header_address()?;
            let mut visited = HashSet::new();
            build_group(ctx, address, String::new(), "/".to_string(), &mut visited, 0)?
        };
        Ok(Self {
            source,
            superblock,
            root,
            cache: options.build_chunk_cache(),
            io: options.io,
            io_cache,
        })
    }

    /// Replace the raw byte cache, or drop it with `None`.
    pub fn with_io_cache(mut self, cache: Option<crate::cache::IoCache>) -> Self {
        self.io_cache = cache;
        self
    }

    /// The raw byte cache in use, if any.
    pub fn io_cache(&self) -> Option<&crate::cache::IoCache> {
        self.io_cache.as_ref()
    }

    /// Set how aggressively byte-range requests are merged.
    ///
    /// Use [`crate::io::IoConfig::REMOTE`] for object storage, where a request
    /// costs milliseconds and bandwidth is cheap.
    pub fn with_io(mut self, io: crate::io::IoConfig) -> Self {
        self.io = io;
        self
    }

    /// The byte-range merging policy in use.
    pub fn io(&self) -> crate::io::IoConfig {
        self.io
    }

    /// Replace the decoded-chunk cache, or drop it with `None`.
    ///
    /// A scan that touches every chunk exactly once gains nothing from caching
    /// and can turn it off to save the memory.
    pub fn with_cache(mut self, cache: Option<crate::cache::ChunkCache>) -> Self {
        self.cache = cache;
        self
    }

    /// The decoded-chunk cache in use, if any.
    pub fn cache(&self) -> Option<&crate::cache::ChunkCache> {
        self.cache.as_ref()
    }

    /// The root group.
    pub fn root(&self) -> &GroupIndex {
        &self.root
    }

    /// The parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// The underlying byte source.
    pub fn source(&self) -> &Arc<dyn ByteSource> {
        &self.source
    }

    /// A read context over this file.
    pub fn ctx(&self) -> Ctx<'_> {
        Ctx::new(self.source.as_ref(), &self.superblock)
            .with_cache(self.cache.as_ref())
            .with_io(self.io)
            .with_io_cache(self.io_cache.as_ref())
    }

    /// Look up a dataset by an absolute path such as `/subgroup/nested_i16`.
    pub fn dataset(&self, path: &str) -> Option<&DatasetIndex> {
        let trimmed = path.trim_start_matches('/');
        if trimmed.is_empty() {
            return None;
        }
        let mut parts: Vec<&str> = trimmed.split('/').collect();
        let leaf = parts.pop()?;

        let mut group = &self.root;
        for part in parts {
            group = group.group(part)?;
        }
        group.dataset(leaf)
    }

    /// Every dataset in the file, depth first.
    pub fn datasets(&self) -> Vec<&DatasetIndex> {
        self.root.datasets_recursive()
    }

    /// Resolve the chunk index of every dataset.
    ///
    /// Restores the old eager behaviour for callers that really do read
    /// everything. A caller with a projection should prepare only what it needs.
    pub fn prepare_all(&self) -> Result<()> {
        let ctx = self.ctx();
        for d in self.datasets() {
            // An index this reader cannot walk makes one variable unreadable,
            // not the whole file.
            if let Err(e) = d.prepare(ctx) {
                if !e.is_fallback_worthy() {
                    return Err(e);
                }
            }
        }
        Ok(())
    }
}

/// One resolved child of a group.
struct Child {
    name: String,
    address: u64,
}

/// Build the index for the group whose object header is at `address`.
fn build_group(
    ctx: Ctx<'_>,
    address: u64,
    name: String,
    path: String,
    visited: &mut HashSet<u64>,
    depth: usize,
) -> Result<GroupIndex> {
    if depth > MAX_DEPTH {
        return Err(Error::malformed(
            "group nesting exceeded the depth limit; the file may be cyclic",
        ));
    }
    if !visited.insert(address) {
        return Err(Error::malformed(format!(
            "group link cycle at object header {address}"
        )));
    }

    let header = ObjectHeader::read(ctx, address)?;
    let (attributes, attributes_complete) = read_attributes(ctx, &header)?;
    let children = resolve_children(ctx, &header)?;

    let mut groups = Vec::new();
    let mut datasets = Vec::new();

    for child in children {
        let child_header = ObjectHeader::read(ctx, child.address)?;
        let child_path = if path == "/" {
            format!("/{}", child.name)
        } else {
            format!("{path}/{}", child.name)
        };

        if child_header.is_dataset() {
            datasets.push(build_dataset(
                ctx,
                &child_header,
                child.name,
                child_path,
                child.address,
            )?);
        } else {
            groups.push(build_group(
                ctx,
                child.address,
                child.name,
                child_path,
                visited,
                depth + 1,
            )?);
        }
    }

    Ok(GroupIndex {
        name,
        path,
        address,
        attributes,
        attributes_complete,
        groups,
        datasets,
    })
}

/// Every attribute of an object, from wherever it is stored.
///
/// Attributes start out inline in the object header and move to a fractal heap
/// once there are enough of them. Both groups and datasets can cross that
/// threshold, so both go through here.
fn read_attributes(ctx: Ctx<'_>, header: &ObjectHeader) -> Result<(Vec<Attribute>, bool)> {
    let sizes = ctx.sizes();
    let mut attributes = header.compact_attributes(sizes).0;
    let mut complete = true;

    if let Some(info) = header.attribute_info(sizes)? {
        if let Some(heap_address) = info.fractal_heap_address {
            let (dense, all_found) = crate::hdf5::dense::read_dense_attributes(
                ctx,
                heap_address,
                info.name_btree_address,
                sizes,
            )?;
            attributes.extend(dense);
            complete = all_found;
        }
    }

    Ok((attributes, complete))
}

/// List a group's children, whichever storage style it uses.
fn resolve_children(ctx: Ctx<'_>, header: &ObjectHeader) -> Result<Vec<Child>> {
    let sizes = ctx.sizes();

    // Old style: a version 1 B-tree over symbol table nodes, plus a local heap
    // of names.
    if let Some(st) = header.symbol_table(sizes)? {
        let heap = LocalHeap::read(ctx, st.local_heap_address)?;
        let node_addresses = btree1::read_group_node_addresses(ctx, st.btree_address)?;

        let mut out = Vec::new();
        for node_address in node_addresses {
            // Read the node header first so the body read is exactly sized.
            let head = ctx.read(node_address, 8)?;
            let count = u16::from_le_bytes([head[6], head[7]]) as usize;
            let raw = ctx.read(
                node_address,
                SymbolTableNode::encoded_len(count, sizes),
            )?;
            let node = SymbolTableNode::parse(&raw, sizes)?;

            for entry in node.entries {
                if let Some(addr) = entry.object_header_address {
                    out.push(Child {
                        name: heap.name_at(entry.link_name_offset)?,
                        address: addr,
                    });
                }
            }
        }
        return Ok(out);
    }

    // New style: link messages inline, or a fractal heap when there are many.
    let mut out = Vec::new();
    for link in header.compact_links(sizes)? {
        if let LinkTarget::Hard { address } = link.target {
            out.push(Child {
                name: link.name,
                address,
            });
        }
        // Soft and external links are not followed. netCDF does not create
        // them, and resolving them would change what "this file" means.
    }

    if let Some(info) = header.link_info(sizes)? {
        if let Some(heap_address) = info.fractal_heap_address {
            let links =
                crate::hdf5::dense::read_dense_links(ctx, heap_address, info.name_btree_address)?;
            for link in links {
                if let LinkTarget::Hard { address } = link.target {
                    out.push(Child {
                        name: link.name,
                        address,
                    });
                }
            }
        }
    }

    Ok(out)
}

/// Build the index entry for one dataset.
fn build_dataset(
    ctx: Ctx<'_>,
    header: &ObjectHeader,
    name: String,
    path: String,
    address: u64,
) -> Result<DatasetIndex> {
    let sizes = ctx.sizes();

    let dataspace = header
        .dataspace(sizes)?
        .unwrap_or(Dataspace {
            kind: crate::hdf5::message::DataspaceKind::Scalar,
            dims: Vec::new(),
            max_dims: None,
        });
    let datatype = header
        .datatype()?
        .ok_or_else(|| Error::malformed(format!("dataset {path} has no datatype message")))?;
    let layout = header
        .layout(sizes)?
        .ok_or_else(|| Error::malformed(format!("dataset {path} has no data layout message")))?;
    let pipeline = header.filter_pipeline()?;
    let fill_value = header.fill_value()?;

    let (attributes, attributes_complete) = read_attributes(ctx, header)?;

    Ok(DatasetIndex {
        name,
        path,
        address,
        shape: dataspace.dims,
        max_shape: dataspace.max_dims,
        datatype,
        layout,
        pipeline,
        fill_value,
        chunks: std::sync::OnceLock::new(),
        attributes,
        attributes_complete,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEGACY_FILE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_files/legacy_v1_objheader.h5"
    );

    #[test]
    fn indexes_every_dataset_of_the_legacy_fixture() {
        let file = Hdf5File::open(LEGACY_FILE).unwrap();
        let mut names: Vec<&str> = file.datasets().iter().map(|d| d.path.as_str()).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "/chunked_i32",
                "/contig_f32be",
                "/contig_f64",
                "/fixed_strings",
                "/subgroup/nested_i16",
            ]
        );
    }

    #[test]
    fn records_shapes_and_element_sizes() {
        let file = Hdf5File::open(LEGACY_FILE).unwrap();

        let d = file.dataset("/contig_f64").unwrap();
        assert_eq!(d.shape, vec![40, 6]);
        assert_eq!(d.element_size(), 8);
        assert_eq!(d.element_count(), 240);

        let s = file.dataset("/fixed_strings").unwrap();
        assert_eq!(s.shape, vec![5]);
        assert_eq!(s.element_size(), 8, "the fixture uses 8-byte strings");
    }

    #[test]
    fn resolves_a_nested_dataset_by_path() {
        let file = Hdf5File::open(LEGACY_FILE).unwrap();
        let d = file.dataset("/subgroup/nested_i16").unwrap();
        assert_eq!(d.shape, vec![6]);
        assert_eq!(d.element_size(), 2);
        assert!(file.dataset("/subgroup/missing").is_none());
        assert!(file.dataset("/no_such_group/x").is_none());
    }

    #[test]
    fn resolves_the_chunk_index_at_open_time() {
        let file = Hdf5File::open(LEGACY_FILE).unwrap();
        let d = file.dataset("/chunked_i32").unwrap();
        let chunks = d.chunks(file.ctx()).unwrap().expect("a chunked dataset has an index");
        assert_eq!(chunks.len(), 12);
        assert!(
            !d.pipeline.is_empty(),
            "the fixture applies shuffle and deflate"
        );
        assert!(d.is_readable());
    }

    #[test]
    fn a_contiguous_dataset_has_no_chunk_index() {
        let file = Hdf5File::open(LEGACY_FILE).unwrap();
        let d = file.dataset("/contig_f64").unwrap();
        assert!(d.chunks(file.ctx()).unwrap().is_none());
        assert!(matches!(d.layout, Layout::Contiguous { .. }));
    }

    #[test]
    fn reads_group_and_dataset_attributes() {
        let file = Hdf5File::open(LEGACY_FILE).unwrap();
        assert!(
            file.root().attribute("title").is_some(),
            "the fixture puts `title` on the root group"
        );
        let d = file.dataset("/contig_f64").unwrap();
        let a = d.attribute("valid_range").expect("valid_range attribute");
        assert_eq!(a.element_count(), 3);
    }

    /// The index must be safe to move between threads and share.
    #[test]
    fn the_index_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Hdf5File>();
        assert_send_sync::<DatasetIndex>();
        assert_send_sync::<GroupIndex>();
    }
}
