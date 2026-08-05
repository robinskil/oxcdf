//! End-to-end value checks against a fixture whose contents are known exactly.
//!
//! `test_files/generate_legacy.c` writes each dataset from a closed-form
//! expression, so these tests compare decoded values against arithmetic rather
//! than against a golden file. Between them they cover every part of the read
//! path: contiguous and chunked storage, the shuffle and deflate filters,
//! big-endian floats, fixed-length strings, nested groups, and partial
//! hyperslabs that straddle chunk boundaries.

use std::sync::Arc;

use oxcdf_hdf5::index::Hdf5File;
use oxcdf_hdf5::read::{read_hyperslab, Hyperslab};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_files/legacy_v1_objheader.h5"
);

const NX: usize = 40;
const NY: usize = 6;

fn open() -> Hdf5File {
    Hdf5File::open(FIXTURE).expect("the fixture should open")
}

#[test]
fn reads_a_whole_contiguous_f64_dataset() {
    let file = open();
    let d = file.dataset("/contig_f64").unwrap();
    let raw = read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape)).unwrap();
    let values = raw.get::<f64>(d).unwrap();

    assert_eq!(values.len(), NX * NY);
    for (i, v) in values.iter().enumerate() {
        assert_eq!(*v, i as f64 * 0.5, "element {i}");
    }
}

#[test]
fn reads_a_whole_chunked_and_compressed_dataset() {
    let file = open();
    let d = file.dataset("/chunked_i32").unwrap();
    assert!(
        !d.pipeline.is_empty(),
        "this dataset is shuffled and deflated"
    );

    let raw = read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape)).unwrap();
    let values = raw.get::<i64>(d).unwrap();

    assert_eq!(values.len(), NX * NY);
    for (i, v) in values.iter().enumerate() {
        assert_eq!(*v, i as i64 * 3 - 100, "element {i}");
    }
}

/// The chunk shape is 7x4 over a 40x6 dataset, so this selection starts inside
/// one chunk and ends inside another along both axes.
#[test]
fn reads_a_hyperslab_that_straddles_chunk_boundaries() {
    let file = open();
    let d = file.dataset("/chunked_i32").unwrap();

    let slab = Hyperslab::new(vec![5, 2], vec![10, 3], &d.shape).unwrap();
    let raw = read_hyperslab(file.ctx(), d, &slab).unwrap();
    let values = raw.get::<i64>(d).unwrap();

    assert_eq!(values.len(), 30);
    for row in 0..10usize {
        for col in 0..3usize {
            let global = (5 + row) * NY + (2 + col);
            assert_eq!(
                values[row * 3 + col],
                global as i64 * 3 - 100,
                "row {row} col {col}"
            );
        }
    }
}

#[test]
fn reads_a_hyperslab_of_a_contiguous_dataset() {
    let file = open();
    let d = file.dataset("/contig_f64").unwrap();

    let slab = Hyperslab::new(vec![37, 1], vec![3, 4], &d.shape).unwrap();
    let values = read_hyperslab(file.ctx(), d, &slab)
        .unwrap()
        .get::<f64>(d)
        .unwrap();

    assert_eq!(values.len(), 12);
    for row in 0..3usize {
        for col in 0..4usize {
            let global = (37 + row) * NY + (1 + col);
            assert_eq!(values[row * 4 + col], global as f64 * 0.5);
        }
    }
}

/// Big-endian storage is the case a reader is most likely to get wrong, and the
/// values would look plausible-but-wrong rather than failing loudly.
#[test]
fn reads_big_endian_floats_in_the_right_byte_order() {
    let file = open();
    let d = file.dataset("/contig_f32be").unwrap();
    assert_eq!(
        d.datatype.byte_order(),
        Some(oxcdf_hdf5::message::ByteOrder::Big)
    );

    let values = read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
        .unwrap()
        .get::<f64>(d)
        .unwrap();

    assert_eq!(values.len(), NX);
    for (i, v) in values.iter().enumerate() {
        assert_eq!(*v, i as f64 * -1.25, "element {i}");
    }
}

#[test]
fn reads_fixed_length_strings() {
    let file = open();
    let d = file.dataset("/fixed_strings").unwrap();
    let strings = read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
        .unwrap()
        .to_strings(d)
        .unwrap();

    assert_eq!(strings, vec!["alpha", "beta", "gamma", "delta", "epsilon"]);
}

#[test]
fn reads_a_dataset_inside_a_subgroup() {
    let file = open();
    let d = file.dataset("/subgroup/nested_i16").unwrap();
    let values = read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
        .unwrap()
        .get::<i64>(d)
        .unwrap();
    assert_eq!(values, (0..NY as i64).map(|i| 1000 + i).collect::<Vec<_>>());
}

#[test]
fn reads_attribute_metadata() {
    let file = open();
    assert!(file.root().attribute("title").is_some());
    let d = file.dataset("/contig_f64").unwrap();
    assert_eq!(d.attribute("valid_range").unwrap().element_count(), 3);
}

#[test]
fn rejects_a_selection_outside_the_dataset() {
    let file = open();
    let d = file.dataset("/contig_f64").unwrap();
    let slab = Hyperslab {
        start: vec![39, 0],
        count: vec![5, 6],
    };
    assert!(read_hyperslab(file.ctx(), d, &slab).is_err());
}

/// The point of the crate: many threads reading the same file at once, with no
/// lock between them, producing the same answers as a sequential read.
#[test]
fn many_threads_read_the_same_file_concurrently() {
    let file = Arc::new(open());

    let expected: Vec<i64> = {
        let d = file.dataset("/chunked_i32").unwrap();
        read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
            .unwrap()
            .get::<i64>(d)
            .unwrap()
    };

    let mut handles = Vec::new();
    for thread in 0..8 {
        let file = Arc::clone(&file);
        let expected = expected.clone();
        handles.push(std::thread::spawn(move || {
            // Each thread reads a different row range, then the whole dataset,
            // so the reads overlap heavily.
            let d = file.dataset("/chunked_i32").unwrap();
            let start = (thread * 5) as u64;

            let slab = Hyperslab::new(vec![start, 0], vec![5, NY as u64], &d.shape).unwrap();
            let part = read_hyperslab(file.ctx(), d, &slab)
                .unwrap()
                .get::<i64>(d)
                .unwrap();
            for (i, v) in part.iter().enumerate() {
                let global = start as usize * NY + i;
                assert_eq!(*v, expected[global], "thread {thread} element {i}");
            }

            let whole = read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
                .unwrap()
                .get::<i64>(d)
                .unwrap();
            assert_eq!(whole, expected, "thread {thread} full read");
        }));
    }

    for h in handles {
        h.join().expect("no thread should panic");
    }
}
