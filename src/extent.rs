//! Selections along the axes of a variable.
//!
//! This mirrors the `netcdf` crate. A read takes anything that converts into
//! [`Extents`]. The common forms are ranges and indices.
//!
//! ```no_run
//! # use oxcdf::{Extent, Extents};
//! # fn run(var: oxcdf::Variable<'_>) -> oxcdf::Result<()> {
//! var.get_values::<f32, _>(Extents::All)?;      // the whole variable
//! var.get_values::<f32, _>(..)?;                // the same
//! var.get_values::<f32, _>([0..8, 10..30])?;    // one range for each axis
//! var.get_values::<f32, _>([0, 3])?;            // one element
//! var.get_values::<f32, _>([2.., 5..])?;        // to the end of each axis
//! var.get_values::<f32, _>([..8, ..30])?;       // from the start of each axis
//!
//! // A Rust array holds one type, so mix an index and a range through `Extent`.
//! var.get_values::<f32, _>([Extent::Index(3), (0..6).into()])?;
//!
//! // A start and a count for each axis.
//! var.get_values::<f32, _>(([0usize, 10].as_slice(), [8usize, 20].as_slice()))?;
//! # Ok(()) }
//! ```
//!
//! # Stride
//!
//! [`Extent`] carries a stride, so the type matches the `netcdf` crate. This
//! reader does not read strided selections yet. A stride other than 1 returns
//! [`crate::Error::Unsupported`]. It never reads the wrong elements silently.

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::error::{Error, Result};
use crate::read::Hyperslab;

/// A selection along one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extent {
    /// From `start` to the end of the axis.
    Slice {
        /// First element.
        start: usize,
        /// Step between elements.
        stride: isize,
    },
    /// From `start` up to but not including `end`.
    SliceEnd {
        /// First element.
        start: usize,
        /// One past the last element.
        end: usize,
        /// Step between elements.
        stride: isize,
    },
    /// `count` elements from `start`.
    SliceCount {
        /// First element.
        start: usize,
        /// Number of elements.
        count: usize,
        /// Step between elements.
        stride: isize,
    },
    /// One element.
    Index(usize),
}

impl Extent {
    /// The start and count this extent selects on an axis of length `len`.
    fn resolve(&self, axis: usize, len: u64) -> Result<(u64, u64)> {
        let stride = match self {
            Extent::Slice { stride, .. }
            | Extent::SliceEnd { stride, .. }
            | Extent::SliceCount { stride, .. } => *stride,
            Extent::Index(_) => 1,
        };
        if stride != 1 {
            return Err(Error::unsupported(format!(
                "a stride of {stride} on axis {axis}; this reader reads contiguous \
                 selections only"
            )));
        }

        let (start, count) = match *self {
            Extent::Slice { start, .. } => {
                let start = start as u64;
                (start, len.saturating_sub(start))
            }
            Extent::SliceEnd { start, end, .. } => {
                if end < start {
                    return Err(Error::bad_request(format!(
                        "the selection on axis {axis} ends at {end} but starts at {start}"
                    )));
                }
                (start as u64, (end - start) as u64)
            }
            Extent::SliceCount { start, count, .. } => (start as u64, count as u64),
            Extent::Index(at) => (at as u64, 1),
        };
        Ok((start, count))
    }
}

/// A selection over a whole variable.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Extents {
    /// Every element.
    #[default]
    All,
    /// One [`Extent`] for each axis.
    Extent(Vec<Extent>),
}

impl Extents {
    /// Turn this into a [`Hyperslab`] over a variable of the given shape.
    ///
    /// `what` names the variable in an error message.
    pub fn to_hyperslab(&self, what: &str, shape: &[u64]) -> Result<Hyperslab> {
        let extents = match self {
            Extents::All => return Ok(Hyperslab::all(shape)),
            Extents::Extent(e) => e,
        };
        if extents.len() != shape.len() {
            return Err(Error::bad_request(format!(
                "variable {what} has rank {} but the selection has rank {}",
                shape.len(),
                extents.len()
            )));
        }

        let mut start = Vec::with_capacity(extents.len());
        let mut count = Vec::with_capacity(extents.len());
        for (axis, (extent, &len)) in extents.iter().zip(shape).enumerate() {
            let (s, c) = extent.resolve(axis, len)?;
            start.push(s);
            count.push(c);
        }
        let slab = Hyperslab { start, count };
        slab.validate(shape)?;
        Ok(slab)
    }
}

// ─── one axis ──────────────────────────────────────────────────────────────

impl From<usize> for Extent {
    fn from(at: usize) -> Self {
        Extent::Index(at)
    }
}

impl From<Range<usize>> for Extent {
    fn from(r: Range<usize>) -> Self {
        Extent::SliceEnd {
            start: r.start,
            end: r.end,
            stride: 1,
        }
    }
}

impl From<RangeFrom<usize>> for Extent {
    fn from(r: RangeFrom<usize>) -> Self {
        Extent::Slice {
            start: r.start,
            stride: 1,
        }
    }
}

impl From<RangeTo<usize>> for Extent {
    fn from(r: RangeTo<usize>) -> Self {
        Extent::SliceEnd {
            start: 0,
            end: r.end,
            stride: 1,
        }
    }
}

impl From<RangeToInclusive<usize>> for Extent {
    fn from(r: RangeToInclusive<usize>) -> Self {
        Extent::SliceEnd {
            start: 0,
            end: r.end + 1,
            stride: 1,
        }
    }
}

impl From<RangeInclusive<usize>> for Extent {
    fn from(r: RangeInclusive<usize>) -> Self {
        Extent::SliceEnd {
            start: *r.start(),
            end: *r.end() + 1,
            stride: 1,
        }
    }
}

impl From<RangeFull> for Extent {
    fn from(_: RangeFull) -> Self {
        Extent::Slice {
            start: 0,
            stride: 1,
        }
    }
}

impl From<&Extent> for Extent {
    fn from(e: &Extent) -> Self {
        *e
    }
}

// ─── every axis ────────────────────────────────────────────────────────────

impl From<RangeFull> for Extents {
    fn from(_: RangeFull) -> Self {
        Extents::All
    }
}

impl From<()> for Extents {
    fn from(_: ()) -> Self {
        Extents::All
    }
}

impl From<&Extents> for Extents {
    fn from(e: &Extents) -> Self {
        e.clone()
    }
}

/// Accept an array, a slice or a vector of anything that becomes an [`Extent`].
macro_rules! extents_from {
    ($from:ty) => {
        impl<const N: usize> From<[$from; N]> for Extents {
            fn from(v: [$from; N]) -> Self {
                Extents::Extent(v.into_iter().map(Extent::from).collect())
            }
        }
        impl<const N: usize> From<&[$from; N]> for Extents {
            fn from(v: &[$from; N]) -> Self {
                Extents::Extent(v.iter().cloned().map(Extent::from).collect())
            }
        }
        impl From<&[$from]> for Extents {
            fn from(v: &[$from]) -> Self {
                Extents::Extent(v.iter().cloned().map(Extent::from).collect())
            }
        }
        impl From<Vec<$from>> for Extents {
            fn from(v: Vec<$from>) -> Self {
                Extents::Extent(v.into_iter().map(Extent::from).collect())
            }
        }
        impl From<&Vec<$from>> for Extents {
            fn from(v: &Vec<$from>) -> Self {
                Extents::Extent(v.iter().cloned().map(Extent::from).collect())
            }
        }
    };
}

extents_from!(Extent);
extents_from!(usize);
extents_from!(Range<usize>);
extents_from!(RangeFrom<usize>);
extents_from!(RangeTo<usize>);
extents_from!(RangeToInclusive<usize>);
extents_from!(RangeInclusive<usize>);

// Only `usize` indices are accepted, exactly as the `netcdf` crate does. A
// second integer width would make `[0..8, 10..30]` ambiguous, and the literals
// would fall back to `i32` and stop compiling.

/// A start and a count for each axis.
impl TryFrom<(&[usize], &[usize])> for Extents {
    type Error = Error;
    fn try_from((start, count): (&[usize], &[usize])) -> Result<Self> {
        if start.len() != count.len() {
            return Err(Error::bad_request(format!(
                "the selection gives {} starts and {} counts",
                start.len(),
                count.len()
            )));
        }
        Ok(Extents::Extent(
            start
                .iter()
                .zip(count)
                .map(|(&start, &count)| Extent::SliceCount {
                    start,
                    count,
                    stride: 1,
                })
                .collect(),
        ))
    }
}

impl TryFrom<(Vec<usize>, Vec<usize>)> for Extents {
    type Error = Error;
    fn try_from((start, count): (Vec<usize>, Vec<usize>)) -> Result<Self> {
        Extents::try_from((start.as_slice(), count.as_slice()))
    }
}

impl From<Hyperslab> for Extents {
    fn from(slab: Hyperslab) -> Self {
        Extents::Extent(
            slab.start
                .iter()
                .zip(&slab.count)
                .map(|(&start, &count)| Extent::SliceCount {
                    start: start as usize,
                    count: count as usize,
                    stride: 1,
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPE: [u64; 2] = [10, 20];

    fn slab(e: impl Into<Extents>) -> Hyperslab {
        e.into().to_hyperslab("x", &SHAPE).unwrap()
    }

    #[test]
    fn everything_selects_the_whole_variable() {
        assert_eq!(slab(Extents::All), Hyperslab::all(&SHAPE));
        assert_eq!(slab(..), Hyperslab::all(&SHAPE));
        assert_eq!(slab(()), Hyperslab::all(&SHAPE));
    }

    #[test]
    fn ranges_select_one_block() {
        let s = slab([2..5, 3..8]);
        assert_eq!(s.start, vec![2, 3]);
        assert_eq!(s.count, vec![3, 5]);
    }

    #[test]
    fn an_index_selects_one_element_on_that_axis() {
        let s = slab([1usize, 2]);
        assert_eq!(s.start, vec![1, 2]);
        assert_eq!(s.count, vec![1, 1]);
        assert_eq!(s.element_count(), 1);
    }

    #[test]
    fn an_open_range_runs_to_the_end_of_the_axis() {
        let s = slab([3.., 5..]);
        assert_eq!(s.start, vec![3, 5]);
        assert_eq!(s.count, vec![7, 15]);
    }

    #[test]
    fn a_range_to_starts_at_zero() {
        let s = slab([..4, ..6]);
        assert_eq!(s.start, vec![0, 0]);
        assert_eq!(s.count, vec![4, 6]);
    }

    #[test]
    fn an_inclusive_range_includes_its_end() {
        let s = slab([2..=4, 0..=1]);
        assert_eq!(s.count, vec![3, 2]);
    }

    #[test]
    fn a_start_and_count_pair_works() {
        let s = Extents::try_from(([1usize, 2].as_slice(), [3usize, 4].as_slice()))
            .unwrap()
            .to_hyperslab("x", &SHAPE)
            .unwrap();
        assert_eq!(s.start, vec![1, 2]);
        assert_eq!(s.count, vec![3, 4]);
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn a_selection_of_the_wrong_rank_is_reported() {
        let err = Extents::from([0..2]).to_hyperslab("x", &SHAPE).unwrap_err();
        assert!(matches!(err, Error::BadRequest(_)), "got {err:?}");
    }

    #[test]
    fn a_selection_past_the_end_is_reported() {
        let err = Extents::from([0..2, 0..99])
            .to_hyperslab("x", &SHAPE)
            .unwrap_err();
        assert!(matches!(err, Error::BadRequest(_)), "got {err:?}");
    }

    #[test]
    fn a_stride_is_refused_rather_than_ignored() {
        let strided = Extents::Extent(vec![
            Extent::Index(0),
            Extent::SliceEnd {
                start: 0,
                end: 10,
                stride: 2,
            },
        ]);
        let err = strided.to_hyperslab("x", &SHAPE).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
    }
}
