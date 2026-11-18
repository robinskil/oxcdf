//! Datasets with contiguous storage for the writer.

use super::alloc::Allocator;
use super::message::{
    encode_contiguous_layout, encode_dataspace, encode_datatype, encode_fill_value,
    numeric_datatype,
};
use super::sink::ByteSink;
use super::writer::message;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::message::{ByteOrder, Dataspace, DataspaceKind, Datatype, FillValue};
use crate::objheader::{HeaderMessage, MessageType};
use crate::read::Hyperslab;

/// Largest block of fill bytes in one sink write.
const FILL_BLOCK_BYTES: usize = 1 << 20;

/// The datatype, shape and fill value of a new dataset.
#[derive(Debug, Clone)]
pub struct DatasetSpec {
    /// Type of one element.
    pub datatype: Datatype,
    /// Size of each axis. An empty shape gives a scalar dataset.
    pub shape: Vec<u64>,
    /// One element in the dataset byte order. `None` reads as zeros.
    pub fill_value: Option<Vec<u8>>,
}

impl DatasetSpec {
    /// A numeric dataset with no fill value.
    ///
    /// A type with no numeric HDF5 datatype is [`Error::Unsupported`].
    pub fn numeric(dtype: DType, order: ByteOrder, shape: &[u64]) -> Result<Self> {
        Ok(Self {
            datatype: numeric_datatype(dtype, order)?,
            shape: shape.to_vec(),
            fill_value: None,
        })
    }

    /// Set the fill value: one element in the dataset byte order.
    pub fn fill_value(mut self, bytes: Vec<u8>) -> Self {
        self.fill_value = Some(bytes);
        self
    }
}

/// A dataset in the writer, with its fixed messages encoded at creation.
#[derive(Debug)]
pub(super) struct Dataset {
    datatype: Datatype,
    shape: Vec<u64>,
    fill_value: Option<Vec<u8>>,
    /// Size of one element in bytes, above zero.
    element_size: u64,
    /// Size of the full extent in bytes.
    extent_bytes: u64,
    /// Address of the extent, set by the first write that has bytes.
    address: Option<u64>,
    dataspace_body: Vec<u8>,
    datatype_body: Vec<u8>,
    fill_body: Vec<u8>,
}

impl Dataset {
    /// Check `spec` and encode its fixed messages, so that close cannot fail on it.
    pub(super) fn new(spec: DatasetSpec) -> Result<Self> {
        let datatype_body = encode_datatype(&spec.datatype)?;
        let kind = if spec.shape.is_empty() {
            DataspaceKind::Scalar
        } else {
            DataspaceKind::Simple
        };
        let dataspace_body = encode_dataspace(&Dataspace {
            kind,
            dims: spec.shape.clone(),
            max_dims: None,
        })?;
        let element_size = u64::from(spec.datatype.size);
        if element_size == 0 {
            return Err(Error::bad_request("a dataset element has 0 bytes"));
        }
        if let Some(fill) = &spec.fill_value {
            if u64::try_from(fill.len()).ok() != Some(element_size) {
                return Err(Error::bad_request(format!(
                    "fill value has {} bytes, but one element has {element_size}",
                    fill.len()
                )));
            }
        }
        let fill = FillValue {
            bytes: spec.fill_value,
        };
        let fill_body = encode_fill_value(&fill)?;
        // A zero axis empties the dataset, whatever the size of the other axes.
        let elements = if spec.shape.contains(&0) {
            Some(0)
        } else {
            spec.shape
                .iter()
                .try_fold(1u64, |n, &dim| n.checked_mul(dim))
        };
        let extent_bytes = elements
            .and_then(|n| n.checked_mul(element_size))
            .ok_or_else(|| {
                Error::bad_request(format!(
                    "shape {:?} with {element_size}-byte elements needs more than u64::MAX bytes",
                    spec.shape
                ))
            })?;
        Ok(Self {
            datatype: spec.datatype,
            shape: spec.shape,
            fill_value: fill.bytes,
            element_size,
            extent_bytes,
            address: None,
            dataspace_body,
            datatype_body,
            fill_body,
        })
    }

    /// The type of one element.
    pub(super) fn datatype(&self) -> &Datatype {
        &self.datatype
    }

    /// The dataspace, datatype, fill value and layout messages, in that order.
    pub(super) fn messages(&self) -> [HeaderMessage; 4] {
        [
            message(MessageType::Dataspace, self.dataspace_body.clone()),
            message(MessageType::Datatype, self.datatype_body.clone()),
            message(MessageType::FillValue, self.fill_body.clone()),
            message(
                MessageType::DataLayout,
                encode_contiguous_layout(self.address, self.extent_bytes),
            ),
        ]
    }

    /// Write `bytes` in the dataset byte order to the elements of `slab`.
    pub(super) fn write_raw<S: ByteSink>(
        &mut self,
        sink: &mut S,
        alloc: &mut Allocator,
        slab: &Hyperslab,
        bytes: &[u8],
    ) -> Result<()> {
        slab.validate(&self.shape)?;
        let selected = slab.element_count().checked_mul(self.element_size);
        if selected != u64::try_from(bytes.len()).ok() {
            return Err(Error::bad_request(format!(
                "{} bytes for a selection of {} elements of {} bytes",
                bytes.len(),
                slab.element_count(),
                self.element_size
            )));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        let address = match self.address {
            Some(address) => address,
            None => {
                let address = alloc.alloc_data(self.extent_bytes)?;
                // A selection inside the shape with all its bytes covers the full extent.
                if selected != Some(self.extent_bytes) {
                    if let Err(e) = self.write_fill(sink, address) {
                        alloc.free(address, self.extent_bytes);
                        return Err(e);
                    }
                }
                self.address = Some(address);
                address
            }
        };
        self.write_runs(sink, address, slab, bytes)
    }

    /// Write the fill value, or zeros, to the full extent at `address`.
    fn write_fill<S: ByteSink>(&self, sink: &mut S, address: u64) -> Result<()> {
        let block = match &self.fill_value {
            Some(fill) => {
                // A fill value fits in one message, so one element fits in a block.
                let per_block = (FILL_BLOCK_BYTES / fill.len().max(1)).max(1);
                let elements = (self.extent_bytes / self.element_size).min(to_u64(per_block)?);
                fill.repeat(to_usize(elements)?)
            }
            None => vec![0; to_usize(self.extent_bytes.min(to_u64(FILL_BLOCK_BYTES)?))?],
        };
        let mut done = 0u64;
        while done < self.extent_bytes {
            let len = (self.extent_bytes - done).min(to_u64(block.len())?);
            let block = block
                .get(..to_usize(len)?)
                .filter(|b| !b.is_empty())
                .ok_or_else(|| Error::bad_request("a fill block is empty or too short"))?;
            sink.write_all_at(offset(address, done)?, block)?;
            done += len;
        }
        Ok(())
    }

    /// Write each contiguous run of `slab` from `bytes` to the extent at `address`.
    fn write_runs<S: ByteSink>(
        &self,
        sink: &mut S,
        address: u64,
        slab: &Hyperslab,
        bytes: &[u8],
    ) -> Result<()> {
        let rank = self.shape.len();
        if rank == 0 {
            return sink.write_all_at(address, bytes);
        }
        // Element strides of each axis in row-major order.
        let mut strides = vec![1u64; rank];
        for axis in (0..rank - 1).rev() {
            strides[axis] = strides[axis + 1]
                .checked_mul(self.shape[axis + 1])
                .ok_or_else(overflow)?;
        }
        // One run spans `axis` and each later axis that the slab covers in full.
        let mut axis = rank - 1;
        while axis > 0 && slab.start[axis] == 0 && slab.count[axis] == self.shape[axis] {
            axis -= 1;
        }
        let run_bytes = slab.count[axis]
            .checked_mul(strides[axis])
            .and_then(|n| n.checked_mul(self.element_size))
            .filter(|&n| n > 0)
            .ok_or_else(overflow)?;
        let first = slab.start[axis]
            .checked_mul(strides[axis])
            .ok_or_else(overflow)?;
        // Position of the current run in the slab, on the axes before `axis`.
        let mut index = vec![0u64; axis];
        for run in bytes.chunks_exact(to_usize(run_bytes)?) {
            let mut element = first;
            for a in 0..axis {
                element = slab.start[a]
                    .checked_add(index[a])
                    .and_then(|i| i.checked_mul(strides[a]))
                    .and_then(|n| n.checked_add(element))
                    .ok_or_else(overflow)?;
            }
            let at = element
                .checked_mul(self.element_size)
                .ok_or_else(overflow)?;
            sink.write_all_at(offset(address, at)?, run)?;
            for a in (0..axis).rev() {
                index[a] += 1;
                if index[a] < slab.count[a] {
                    break;
                }
                index[a] = 0;
            }
        }
        Ok(())
    }
}

/// Swap each element of `size` bytes in place when `order` is not the host order.
pub(super) fn to_order(bytes: &mut [u8], size: usize, order: Option<ByteOrder>) {
    let host = if cfg!(target_endian = "big") {
        ByteOrder::Big
    } else {
        ByteOrder::Little
    };
    if size > 1 && matches!(order, Some(o) if o != host) {
        for element in bytes.chunks_exact_mut(size) {
            element.reverse();
        }
    }
}

/// The address `at` bytes past `address`.
fn offset(address: u64, at: u64) -> Result<u64> {
    address.checked_add(at).ok_or_else(overflow)
}

/// The error for a file offset past `u64::MAX`.
fn overflow() -> Error {
    Error::bad_request("a dataset offset passes u64::MAX")
}

/// Convert a length to `usize`, or return an error when it does not fit.
fn to_usize(n: u64) -> Result<usize> {
    usize::try_from(n).map_err(|_| Error::unsupported(format!("{n} does not fit in usize")))
}

/// Convert a length to `u64`, or return an error when it does not fit.
fn to_u64(n: usize) -> Result<u64> {
    u64::try_from(n).map_err(|_| Error::unsupported(format!("{n} does not fit in 64 bits")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::MemorySink;

    /// A memory sink that records the offset and length of each write.
    #[derive(Debug, Default)]
    struct Recorder {
        inner: MemorySink,
        writes: Vec<(u64, usize)>,
    }

    impl ByteSink for Recorder {
        fn write_all_at(&mut self, offset: u64, data: &[u8]) -> Result<()> {
            self.writes.push((offset, data.len()));
            self.inner.write_all_at(offset, data)
        }

        fn set_len(&mut self, len: u64) -> Result<()> {
            self.inner.set_len(len)
        }

        fn sync(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn dataset(shape: &[u64]) -> Dataset {
        Dataset::new(DatasetSpec::numeric(DType::Uint(1), ByteOrder::Little, shape).unwrap())
            .unwrap()
    }

    #[test]
    fn full_trailing_axes_merge_into_one_run() {
        let mut d = dataset(&[4, 3, 5]);
        let mut sink = Recorder::default();
        let mut alloc = Allocator::new(0, 0);
        let slab = Hyperslab::new(vec![1, 0, 0], vec![2, 3, 5], &d.shape).unwrap();
        d.write_raw(&mut sink, &mut alloc, &slab, &[1; 30]).unwrap();
        // The fill of 60 bytes, then one run of two full planes.
        assert_eq!(sink.writes, [(0, 60), (15, 30)]);
    }

    #[test]
    fn a_partial_last_axis_writes_one_run_per_row() {
        let mut d = dataset(&[2, 3, 5]);
        let mut sink = Recorder::default();
        let mut alloc = Allocator::new(0, 0);
        let slab = Hyperslab::new(vec![0, 1, 1], vec![2, 2, 3], &d.shape).unwrap();
        let bytes: Vec<u8> = (1..=12).collect();
        d.write_raw(&mut sink, &mut alloc, &slab, &bytes).unwrap();
        assert_eq!(sink.writes[1..], [(6, 3), (11, 3), (21, 3), (26, 3)]);
        let data = sink.inner.into_inner();
        assert_eq!(data[6..9], [1, 2, 3]);
        assert_eq!(data[26..29], [10, 11, 12]);
    }

    #[test]
    fn the_fill_goes_in_blocks_of_one_mebibyte_at_most() {
        let spec = DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, &[600_000])
            .unwrap()
            .fill_value(vec![1, 2, 3, 4]);
        let mut d = Dataset::new(spec).unwrap();
        let mut sink = Recorder::default();
        let mut alloc = Allocator::new(0, 0);
        let slab = Hyperslab::new(vec![0], vec![1], &d.shape).unwrap();
        d.write_raw(&mut sink, &mut alloc, &slab, &[9; 4]).unwrap();
        let lens: Vec<usize> = sink.writes.iter().map(|w| w.1).collect();
        assert_eq!(lens, [1 << 20, 1 << 20, 2_400_000 - (2 << 20), 4]);
        let data = sink.inner.into_inner();
        assert_eq!(data[..8], [9, 9, 9, 9, 1, 2, 3, 4]);
        assert_eq!(data[2_399_996..], [1, 2, 3, 4]);
    }

    #[test]
    fn a_full_first_write_writes_no_fill() {
        let mut d = dataset(&[2, 2]);
        let mut sink = Recorder::default();
        let mut alloc = Allocator::new(0, 0);
        let slab = Hyperslab::all(&d.shape);
        d.write_raw(&mut sink, &mut alloc, &slab, &[1, 2, 3, 4])
            .unwrap();
        assert_eq!(sink.writes, [(0, 4)]);
    }

    #[test]
    fn wrong_specs_are_rejected() {
        let int4 = |shape: &[u64]| DatasetSpec::numeric(DType::Int(4), ByteOrder::Little, shape);
        assert!(matches!(
            Dataset::new(int4(&[u64::MAX, 2]).unwrap()),
            Err(Error::BadRequest(_))
        ));
        assert!(matches!(
            Dataset::new(int4(&[1; 33]).unwrap()),
            Err(Error::BadRequest(_))
        ));
        assert!(matches!(
            DatasetSpec::numeric(DType::Char, ByteOrder::Little, &[1]),
            Err(Error::Unsupported(_))
        ));
        let compound = DatasetSpec {
            datatype: Datatype {
                version: 1,
                size: 8,
                class: crate::message::DatatypeClass::Compound { members: vec![] },
            },
            shape: vec![1],
            fill_value: None,
        };
        assert!(matches!(Dataset::new(compound), Err(Error::Unsupported(_))));
        // A zero axis gives an empty extent before the product overflows.
        assert_eq!(
            Dataset::new(int4(&[u64::MAX, u64::MAX, 0]).unwrap())
                .unwrap()
                .extent_bytes,
            0
        );
    }
}
