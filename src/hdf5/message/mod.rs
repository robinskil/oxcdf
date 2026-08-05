//! Typed views over object header message bodies.
//!
//! [`super::objheader`] frames messages and keeps their bytes. These modules
//! turn those bytes into values. Splitting the two means the framing rules
//! (which differ between object header versions) stay in one place, and the
//! message parsers do not care which version framed them.

pub mod attribute;
pub mod dataspace;
pub mod datatype;
pub mod fillvalue;
pub mod filter;
pub mod layout;
pub mod link;

pub use attribute::Attribute;
pub use dataspace::{Dataspace, DataspaceKind};
pub use datatype::{ByteOrder, CharSet, Datatype, DatatypeClass, StringPad, VlenKind};
pub use fillvalue::FillValue;
pub use filter::{Filter, FilterPipeline};
pub use layout::{ChunkIndex, Layout};
pub use link::{AttributeInfo, Link, LinkInfo, LinkTarget, SymbolTable};

use crate::cursor::Sizes;
use crate::error::{Error, Result};
use crate::hdf5::objheader::{HeaderMessage, MessageType, ObjectHeader};

/// Reject a message whose body is a pointer into the shared message table.
///
/// netcdf-c never enables shared messages, so rather than decode the pointer as
/// if it were the message itself, say plainly that it is not supported.
fn reject_shared(msg: &HeaderMessage, what: &str) -> Result<()> {
    if msg.is_shared() {
        return Err(Error::unsupported(format!("shared {what} message")));
    }
    Ok(())
}

/// Convenience accessors that pull typed messages out of an object header.
impl ObjectHeader {
    /// The object's dataspace, when it has one.
    pub fn dataspace(&self, sizes: Sizes) -> Result<Option<Dataspace>> {
        match self.message_of(MessageType::Dataspace) {
            Some(m) => {
                reject_shared(m, "dataspace")?;
                Ok(Some(Dataspace::parse(&m.data, sizes)?))
            }
            None => Ok(None),
        }
    }

    /// The object's datatype, when it has one.
    pub fn datatype(&self) -> Result<Option<Datatype>> {
        match self.message_of(MessageType::Datatype) {
            Some(m) => {
                reject_shared(m, "datatype")?;
                Ok(Some(Datatype::parse(&m.data)?))
            }
            None => Ok(None),
        }
    }

    /// The object's data layout, when it has one.
    pub fn layout(&self, sizes: Sizes) -> Result<Option<Layout>> {
        match self.message_of(MessageType::DataLayout) {
            Some(m) => {
                reject_shared(m, "data layout")?;
                Ok(Some(Layout::parse(&m.data, sizes)?))
            }
            None => Ok(None),
        }
    }

    /// The object's filter pipeline. An object without the message has none.
    pub fn filter_pipeline(&self) -> Result<FilterPipeline> {
        match self.message_of(MessageType::FilterPipeline) {
            Some(m) => {
                reject_shared(m, "filter pipeline")?;
                FilterPipeline::parse(&m.data)
            }
            None => Ok(FilterPipeline::default()),
        }
    }

    /// The dataset's fill value, which is what unwritten storage reads as.
    ///
    /// An absent message means zeros, which is the format's defined default.
    pub fn fill_value(&self) -> Result<FillValue> {
        if let Some(m) = self.message_of(MessageType::FillValue) {
            return FillValue::parse(&m.data);
        }
        if let Some(m) = self.message_of(MessageType::FillValueOld) {
            return FillValue::parse_old(&m.data);
        }
        Ok(FillValue::default())
    }

    /// The link info message, when this object is a new-style group.
    pub fn link_info(&self, sizes: Sizes) -> Result<Option<LinkInfo>> {
        match self.message_of(MessageType::LinkInfo) {
            Some(m) => Ok(Some(LinkInfo::parse(&m.data, sizes)?)),
            None => Ok(None),
        }
    }

    /// The symbol table message, when this object is an old-style group.
    pub fn symbol_table(&self, sizes: Sizes) -> Result<Option<SymbolTable>> {
        match self.message_of(MessageType::SymbolTable) {
            Some(m) => Ok(Some(SymbolTable::parse(&m.data, sizes)?)),
            None => Ok(None),
        }
    }

    /// The attribute info message, when attributes are stored densely.
    pub fn attribute_info(&self, sizes: Sizes) -> Result<Option<AttributeInfo>> {
        match self.message_of(MessageType::AttributeInfo) {
            Some(m) => Ok(Some(AttributeInfo::parse(&m.data, sizes)?)),
            None => Ok(None),
        }
    }

    /// Every link stored inline in this group's header (compact storage).
    pub fn compact_links(&self, sizes: Sizes) -> Result<Vec<Link>> {
        self.messages_of(MessageType::Link)
            .map(|m| Link::parse(&m.data, sizes))
            .collect()
    }

    /// Every attribute stored inline in this object's header.
    ///
    /// An attribute that this reader cannot parse is skipped rather than
    /// failing the object: one exotic attribute should not make a variable
    /// unreadable. The count of skipped attributes is returned alongside.
    pub fn compact_attributes(&self, sizes: Sizes) -> (Vec<Attribute>, Vec<Error>) {
        let mut out = Vec::new();
        let mut errors = Vec::new();
        for m in self.messages_of(MessageType::Attribute) {
            if m.is_shared() {
                errors.push(Error::unsupported("shared attribute message"));
                continue;
            }
            match Attribute::parse(&m.data, sizes) {
                Ok(a) => out.push(a),
                Err(e) => errors.push(e),
            }
        }
        (out, errors)
    }

    /// Whether this object looks like a group rather than a dataset.
    ///
    /// A dataset always carries a data layout message; a group never does.
    pub fn is_group(&self) -> bool {
        self.message_of(MessageType::DataLayout).is_none()
            && (self.message_of(MessageType::LinkInfo).is_some()
                || self.message_of(MessageType::SymbolTable).is_some()
                || self.message_of(MessageType::GroupInfo).is_some())
    }

    /// Whether this object is a dataset.
    pub fn is_dataset(&self) -> bool {
        self.message_of(MessageType::DataLayout).is_some()
    }
}
