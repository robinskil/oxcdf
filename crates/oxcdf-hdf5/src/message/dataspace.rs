//! The dataspace message: the shape of a dataset or attribute.

use crate::cursor::{Cursor, Sizes};
use crate::error::{Error, Result};

/// What kind of dataspace this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataspaceKind {
    /// A single element, rank 0.
    Scalar,
    /// A rectangular array.
    Simple,
    /// No elements at all.
    Null,
}

/// The shape of a dataset or attribute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dataspace {
    /// Whether this is scalar, simple or null.
    pub kind: DataspaceKind,
    /// Size along each dimension. Empty for scalar and null.
    pub dims: Vec<u64>,
    /// Maximum size along each dimension, when recorded.
    ///
    /// A value of `u64::MAX` marks an unlimited dimension.
    pub max_dims: Option<Vec<u64>>,
}

impl Dataspace {
    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements.
    ///
    /// A scalar holds one element; a null dataspace holds none.
    pub fn element_count(&self) -> u64 {
        match self.kind {
            DataspaceKind::Null => 0,
            DataspaceKind::Scalar => 1,
            DataspaceKind::Simple => self.dims.iter().product(),
        }
    }

    /// Whether any dimension is unlimited.
    pub fn has_unlimited(&self) -> bool {
        self.max_dims
            .as_ref()
            .is_some_and(|m| m.contains(&u64::MAX))
    }

    /// Parse a dataspace message body.
    pub fn parse(data: &[u8], sizes: Sizes) -> Result<Self> {
        let mut cur = Cursor::new(data);
        let version = cur.u8()?;
        match version {
            1 => Self::parse_v1(&mut cur, sizes),
            2 => Self::parse_v2(&mut cur, sizes),
            other => Err(Error::unsupported(format!(
                "dataspace message version {other}"
            ))),
        }
    }

    fn parse_v1(cur: &mut Cursor<'_>, sizes: Sizes) -> Result<Self> {
        let rank = cur.u8()? as usize;
        let flags = cur.u8()?;
        cur.skip(1)?; // reserved
        cur.skip(4)?; // reserved

        let dims = read_dims(cur, rank, sizes)?;
        let max_dims = if flags & 0x01 != 0 {
            Some(read_dims(cur, rank, sizes)?)
        } else {
            None
        };
        // Permutation indices were never implemented by the library. If the
        // flag is set the message still carries the field, but the values are
        // meaningless, so skipping is correct.
        if flags & 0x02 != 0 {
            cur.skip(rank * sizes.length as usize)?;
        }

        // Version 1 has no explicit type field. Rank zero means scalar.
        let kind = if rank == 0 {
            DataspaceKind::Scalar
        } else {
            DataspaceKind::Simple
        };

        Ok(Self {
            kind,
            dims,
            max_dims,
        })
    }

    fn parse_v2(cur: &mut Cursor<'_>, sizes: Sizes) -> Result<Self> {
        let rank = cur.u8()? as usize;
        let flags = cur.u8()?;
        let kind = match cur.u8()? {
            0 => DataspaceKind::Scalar,
            1 => DataspaceKind::Simple,
            2 => DataspaceKind::Null,
            other => {
                return Err(Error::malformed(format!("dataspace type {other}")));
            }
        };

        let dims = read_dims(cur, rank, sizes)?;
        let max_dims = if flags & 0x01 != 0 {
            Some(read_dims(cur, rank, sizes)?)
        } else {
            None
        };

        Ok(Self {
            kind,
            dims,
            max_dims,
        })
    }
}

fn read_dims(cur: &mut Cursor<'_>, rank: usize, sizes: Sizes) -> Result<Vec<u64>> {
    let mut out = Vec::with_capacity(rank);
    for _ in 0..rank {
        out.push(cur.length(sizes)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v2_simple(dims: &[u64], max: Option<&[u64]>) -> Vec<u8> {
        let mut v = vec![2u8, dims.len() as u8, if max.is_some() { 1 } else { 0 }, 1];
        for d in dims {
            v.extend_from_slice(&d.to_le_bytes());
        }
        if let Some(m) = max {
            for d in m {
                v.extend_from_slice(&d.to_le_bytes());
            }
        }
        v
    }

    #[test]
    fn parses_a_simple_two_dimensional_space() {
        let data = v2_simple(&[4, 5], None);
        let ds = Dataspace::parse(&data, Sizes::EIGHT).unwrap();
        assert_eq!(ds.kind, DataspaceKind::Simple);
        assert_eq!(ds.dims, vec![4, 5]);
        assert_eq!(ds.rank(), 2);
        assert_eq!(ds.element_count(), 20);
        assert!(!ds.has_unlimited());
    }

    #[test]
    fn parses_maximum_dimensions_and_detects_unlimited() {
        let data = v2_simple(&[3], Some(&[u64::MAX]));
        let ds = Dataspace::parse(&data, Sizes::EIGHT).unwrap();
        assert_eq!(ds.max_dims, Some(vec![u64::MAX]));
        assert!(ds.has_unlimited(), "all-bits-set marks an unlimited axis");
    }

    #[test]
    fn a_scalar_space_holds_one_element() {
        let data = vec![2u8, 0, 0, 0];
        let ds = Dataspace::parse(&data, Sizes::EIGHT).unwrap();
        assert_eq!(ds.kind, DataspaceKind::Scalar);
        assert_eq!(ds.rank(), 0);
        assert_eq!(ds.element_count(), 1);
    }

    #[test]
    fn a_null_space_holds_no_elements() {
        let data = vec![2u8, 0, 0, 2];
        let ds = Dataspace::parse(&data, Sizes::EIGHT).unwrap();
        assert_eq!(ds.kind, DataspaceKind::Null);
        assert_eq!(ds.element_count(), 0);
    }

    #[test]
    fn parses_a_version_one_space() {
        let mut data = vec![1u8, 2, 0, 0, 0, 0, 0, 0];
        data.extend_from_slice(&7u64.to_le_bytes());
        data.extend_from_slice(&8u64.to_le_bytes());
        let ds = Dataspace::parse(&data, Sizes::EIGHT).unwrap();
        assert_eq!(ds.dims, vec![7, 8]);
        assert_eq!(ds.kind, DataspaceKind::Simple);
    }

    #[test]
    fn a_version_one_space_of_rank_zero_is_scalar() {
        let data = vec![1u8, 0, 0, 0, 0, 0, 0, 0];
        let ds = Dataspace::parse(&data, Sizes::EIGHT).unwrap();
        assert_eq!(ds.kind, DataspaceKind::Scalar);
        assert_eq!(ds.element_count(), 1);
    }

    #[test]
    fn version_one_skips_permutation_indices() {
        // Flags bit 1 set: the message carries a permutation block that must be
        // consumed even though its contents are meaningless.
        let mut data = vec![1u8, 1, 0x02, 0, 0, 0, 0, 0];
        data.extend_from_slice(&5u64.to_le_bytes()); // dim
        data.extend_from_slice(&0u64.to_le_bytes()); // permutation
        let ds = Dataspace::parse(&data, Sizes::EIGHT).unwrap();
        assert_eq!(ds.dims, vec![5]);
    }

    #[test]
    fn rejects_an_unknown_version() {
        let err = Dataspace::parse(&[9u8, 0, 0, 0], Sizes::EIGHT).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }
}
