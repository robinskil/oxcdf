//! Variable-length strings, the newer filters, and the classic formats.

use oxcdf::classic::ClassicFile;
use oxcdf::filters::{decode_chunk, pipeline_is_supported, unzstd};
use oxcdf::hdf5::message::filter::{id, Filter, FilterPipeline};
use oxcdf::netcdf::{DType, NetcdfFile};
use oxcdf::read::Hyperslab;
use oxcdf::{detect_container, Container, FileSource};

const VLEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_files/vlen_strings.nc");
const CLASSIC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_files/classic.nc");

// ── variable-length strings ───────────────────────────────────────────────

#[test]
fn reads_variable_length_strings() {
    let file = NetcdfFile::open(VLEN).unwrap();
    let v = file.variable("/station_name").expect("station_name");

    assert_eq!(v.vartype(), DType::String);
    assert_eq!(v.shape, vec![4]);

    let values = v.read().unwrap();
    assert!(values.is_variable_length());
    assert_eq!(
        values.to_strings().unwrap(),
        vec![
            "Ålesund",
            "Bergen",
            "",
            "a much longer station name than the others",
        ],
        "covers non-ASCII, an empty string and a string too long to inline"
    );
}

#[test]
fn reads_a_slice_of_a_variable_length_string_variable() {
    let file = NetcdfFile::open(VLEN).unwrap();
    let v = file.variable("/comment").unwrap();
    let ranges: Vec<std::ops::Range<usize>> = (0..1).map(|_| 1usize..3).collect();
    let got = v.get_strings(ranges).unwrap();
    assert_eq!(got, vec!["beta", "gamma"]);
}

#[test]
fn a_variable_length_variable_is_reported_as_readable() {
    let file = NetcdfFile::open(VLEN).unwrap();
    assert!(file.variable("/station_name").unwrap().is_readable());
}

/// The numeric variable in the same file must still read correctly, so the
/// vlen work did not disturb the ordinary path.
#[test]
fn numeric_variables_alongside_strings_still_read() {
    let file = NetcdfFile::open(VLEN).unwrap();
    let t = file.variable("/temperature").unwrap();
    assert_eq!(t.shape, vec![4, 6]);
    let values = t.read().unwrap().get::<f64>().unwrap();
    let expected: Vec<f64> = (0..24).map(|i| i as f64 + 1.5).collect();
    assert_eq!(values, expected);
}

// ── filters ───────────────────────────────────────────────────────────────

fn filter(id: u16, client: Vec<u32>) -> Filter {
    Filter {
        id,
        name: String::new(),
        flags: 0,
        client_data: client,
    }
}

#[test]
fn zstd_round_trips() {
    let original: Vec<u8> = (0..2000).map(|i| (i % 251) as u8).collect();
    let compressed = zstd::stream::encode_all(&original[..], 3).unwrap();
    assert_eq!(unzstd(&compressed, original.len()).unwrap(), original);
}

#[test]
fn a_zstd_pipeline_decodes_a_chunk() {
    let original: Vec<u8> = (0..512u32).flat_map(|v| v.to_le_bytes()).collect();
    let compressed = zstd::stream::encode_all(&original[..], 3).unwrap();

    let pipeline = FilterPipeline {
        filters: vec![filter(id::ZSTD, vec![3])],
    };
    assert!(pipeline_is_supported(&pipeline));
    assert_eq!(
        decode_chunk(&pipeline, 0, compressed, original.len()).unwrap(),
        original
    );
}

#[test]
fn zstd_after_shuffle_decodes_in_the_right_order() {
    use oxcdf::filters::shuffle;
    let original: Vec<u8> = (0..256u32).flat_map(|v| v.to_le_bytes()).collect();
    let stored = zstd::stream::encode_all(&shuffle(&original, 4)[..], 3).unwrap();

    let pipeline = FilterPipeline {
        filters: vec![filter(id::SHUFFLE, vec![4]), filter(id::ZSTD, vec![3])],
    };
    assert_eq!(
        decode_chunk(&pipeline, 0, stored, original.len()).unwrap(),
        original
    );
}

/// A hand-built Blosc container wrapping a zstd block, to check the header
/// parse and block dispatch without needing the Blosc HDF5 plugin.
#[test]
fn blosc_container_with_a_zstd_block_decodes() {
    use oxcdf::filters::unblosc;

    let payload: Vec<u8> = (0..400).map(|i| (i % 97) as u8).collect();
    let block = zstd::stream::encode_all(&payload[..], 3).unwrap();

    // version, blosclz version, flags (codec 5 = zstd, no shuffle), type size
    let mut chunk = vec![2u8, 1u8, 5u8 << 5, 1u8];
    chunk.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // nbytes
    chunk.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // blocksize
    let cbytes = 16 + 4 + 4 + block.len();
    chunk.extend_from_slice(&(cbytes as u32).to_le_bytes()); // cbytes
    // The 16-byte header plus a single 4-byte offset entry ends at 20, which is
    // where the block's own length prefix begins.
    chunk.extend_from_slice(&20u32.to_le_bytes()); // one block offset
    chunk.extend_from_slice(&(block.len() as u32).to_le_bytes()); // block length
    chunk.extend_from_slice(&block);

    assert_eq!(unblosc(&chunk).unwrap(), payload);
}

#[test]
fn blosc_reports_the_codecs_it_cannot_decode() {
    use oxcdf::filters::unblosc;
    // Codec 3 is snappy, which this reader still does not decode.
    let mut chunk = vec![2u8, 1, 3u8 << 5, 1];
    chunk.extend_from_slice(&8u32.to_le_bytes());
    chunk.extend_from_slice(&8u32.to_le_bytes());
    chunk.extend_from_slice(&32u32.to_le_bytes());
    chunk.extend_from_slice(&20u32.to_le_bytes());
    chunk.extend_from_slice(&4u32.to_le_bytes());
    chunk.extend_from_slice(&[0u8; 8]);

    let err = unblosc(&chunk).unwrap_err();
    assert!(
        err.is_fallback_worthy(),
        "an undecodable sub-codec must be reported as unsupported, got {err:?}"
    );
}

#[test]
fn blosc_rejects_a_truncated_header() {
    use oxcdf::filters::unblosc;
    assert!(unblosc(&[2u8, 1, 0, 1]).is_err());
}

// ── classic formats ───────────────────────────────────────────────────────

#[test]
fn detects_and_reads_both_classic_variants() {
    for (path, want) in [
        (CLASSIC, Container::Cdf1),
        (
            concat!(env!("CARGO_MANIFEST_DIR"), "/test_files/classic64.nc"),
            Container::Cdf2,
        ),
    ] {
        let src = FileSource::open(path).unwrap();
        assert_eq!(detect_container(&src).unwrap(), want);

        let file = ClassicFile::open(path).unwrap();
        assert_eq!(file.container, want);
        assert_eq!(file.variables.len(), 5);
    }
}

/// The classic reader must agree with netcdf-c, same as the HDF5 side.
#[cfg(feature = "diff-tests")]
#[test]
fn classic_values_match_netcdf_c() {
    use netcdf::types::{FloatType, IntType, NcVariableType};

    for path in [
        CLASSIC,
        concat!(env!("CARGO_MANIFEST_DIR"), "/test_files/classic64.nc"),
    ] {
        let reference = netcdf::open(path).unwrap();
        let native = ClassicFile::open(path).unwrap();
        let mut compared = 0;

        for ref_var in reference.variables() {
            let name = ref_var.name();
            let Some(mine) = native.variable(&name) else {
                panic!("{name}: the classic reader did not find this variable");
            };

            let expected: Vec<f64> = match ref_var.vartype() {
                NcVariableType::Float(FloatType::F32) => ref_var
                    .get_values::<f32, _>(netcdf::Extents::All)
                    .unwrap()
                    .into_iter()
                    .map(f64::from)
                    .collect(),
                NcVariableType::Float(FloatType::F64) => {
                    ref_var.get_values::<f64, _>(netcdf::Extents::All).unwrap()
                }
                NcVariableType::Int(IntType::I32) => ref_var
                    .get_values::<i32, _>(netcdf::Extents::All)
                    .unwrap()
                    .into_iter()
                    .map(f64::from)
                    .collect(),
                NcVariableType::Int(IntType::I16) => ref_var
                    .get_values::<i16, _>(netcdf::Extents::All)
                    .unwrap()
                    .into_iter()
                    .map(f64::from)
                    .collect(),
                _ => continue, // char handled separately below
            };

            let got = native.read_f64(mine).unwrap();
            assert_eq!(got.len(), expected.len(), "{name}: element count differs");
            for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name}: element {i} differs from netcdf-c"
                );
            }
            compared += 1;
        }

        assert!(compared >= 4, "expected several variables to be compared");
    }
}

#[test]
fn classic_hyperslabs_are_bounds_checked() {
    let file = ClassicFile::open(CLASSIC).unwrap();
    let v = file.variable("pressure").unwrap();
    let slab = Hyperslab {
        start: vec![3, 0],
        count: vec![5, 3],
    };
    assert!(file.read_selection(v, &slab).is_err());
}

// ── version 2 B-tree recovery of gapped attribute heaps ───────────────────

/// Some attribute heaps develop internal free space when netcdf-c rewrites a
/// record in place. A sequential walk cannot see past the gap; the version 2
/// B-tree name index lists every live record, so it can.
///
/// `gridded-example.nc` is the file that exercises this: four of its variables
/// have gapped attribute heaps.
#[test]
fn gapped_attribute_heaps_are_now_complete() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_files/gridded-example.nc"
    );
    let file = NetcdfFile::open(path).unwrap();

    for v in file.variables() {
        assert!(
            v.attributes_complete,
            "{}: attribute list is still short; the B-tree index did not recover it",
            v.path
        );
    }
}

/// The recovered attributes must be the right ones, not just the right count.
#[cfg(feature = "diff-tests")]
#[test]
fn recovered_attribute_names_match_netcdf_c() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_files/gridded-example.nc"
    );
    let reference = netcdf::open(path).unwrap();
    let native = NetcdfFile::open(path).unwrap();

    let mut checked = 0;
    for ref_var in reference.variables() {
        let name = ref_var.name();
        let mine = native.variable(&format!("/{name}")).unwrap();

        for ref_attr in ref_var.attributes() {
            let attr_name = ref_attr.name().to_string();
            assert!(
                mine.attribute(&attr_name).is_some(),
                "{name}: attribute {attr_name} is missing from the native reader"
            );
            checked += 1;
        }
    }
    assert!(checked > 20, "expected many attributes to be checked");
}

// ── blosclz and the bit shuffle ───────────────────────────────────────────

/// Hand-built blosclz streams, checked against the format definition rather
/// than against a compressor, because no blosclz encoder is available here.
#[test]
fn blosclz_decodes_literals_and_back_references() {
    use oxcdf::filters::unblosclz;

    // A single literal run of five bytes: token 0x04 means "4 + 1 literals".
    let stream = [0x04u8, b'a', b'b', b'c', b'd', b'e'];
    assert_eq!(unblosclz(&stream, 5).unwrap(), b"abcde");

    // Literal run, then a back-reference of length 3 at distance 1, which
    // repeats the last byte three times.
    // Token 0x20 = (length-2) 1 in the top three bits, distance high bits 0.
    let stream = [0x00u8, b'x', 0x20, 0x00];
    assert_eq!(unblosclz(&stream, 4).unwrap(), b"xxxx");

    // An overlapping match must copy byte by byte, producing a repeating run
    // longer than the distance it points back.
    let stream = [0x01u8, b'a', b'b', 0x80, 0x01];
    // Literals "ab", then length (4>>0)+2 = 6 at distance 2 → "ababab".
    assert_eq!(unblosclz(&stream, 8).unwrap(), b"ababababab".to_vec()[..8].to_vec());
}

#[test]
fn blosclz_rejects_a_match_pointing_before_the_block() {
    use oxcdf::filters::unblosclz;
    // A back-reference as the second token, pointing further back than exists.
    let stream = [0x00u8, b'x', 0x20, 0xff];
    assert!(unblosclz(&stream, 4).is_err());
}

#[test]
fn blosclz_rejects_a_truncated_literal_run() {
    use oxcdf::filters::unblosclz;
    assert!(unblosclz(&[0x0fu8, b'a'], 16).is_err());
}

#[test]
fn the_bit_shuffle_round_trips() {
    use oxcdf::filters::{bitshuffle, unbitshuffle};
    for typesize in [1usize, 2, 4, 8] {
        let data: Vec<u8> = (0..(typesize * 16) as u32).map(|i| (i * 37 % 251) as u8).collect();
        let shuffled = bitshuffle(&data, typesize);
        assert_eq!(
            unbitshuffle(&shuffled, typesize),
            data,
            "type size {typesize} must round trip"
        );
    }
}

#[test]
fn blosclz_is_now_an_accepted_sub_codec() {
    use oxcdf::filters::unblosc;
    // Codec 0 with a valid one-byte literal payload.
    let payload = b"hello!!!";
    let block = [0x07u8, b'h', b'e', b'l', b'l', b'o', b'!', b'!', b'!'];

    let mut chunk = vec![2u8, 1, 0, 1]; // codec 0 = blosclz, no shuffle
    chunk.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    chunk.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    chunk.extend_from_slice(&((16 + 4 + 4 + block.len()) as u32).to_le_bytes());
    chunk.extend_from_slice(&20u32.to_le_bytes());
    chunk.extend_from_slice(&(block.len() as u32).to_le_bytes());
    chunk.extend_from_slice(&block);

    assert_eq!(unblosc(&chunk).unwrap(), payload);
}

// ── variable-length sequences ─────────────────────────────────────────────

const VLEN_SEQ: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_files/vlen_seq.nc");

#[test]
fn reads_ragged_float_sequences() {
    let file = NetcdfFile::open(VLEN_SEQ).unwrap();
    let v = file.variable("/rows").expect("rows");

    assert_eq!(v.vartype(), DType::Vlen(Box::new(DType::Float(4))));
    assert_eq!(v.shape, vec![4]);

    let values = v.read().unwrap();
    assert!(values.is_variable_length());
    assert_eq!(
        values.to_sequences::<f64>().unwrap(),
        vec![
            vec![1.5, 2.5],
            vec![],
            vec![3.5, 4.5, 5.5],
            vec![-0.25],
        ],
        "an empty sequence is a real value, not a missing one"
    );
}

#[test]
fn reads_ragged_integer_sequences() {
    let file = NetcdfFile::open(VLEN_SEQ).unwrap();
    let v = file.variable("/indices").unwrap();

    assert_eq!(v.vartype(), DType::Vlen(Box::new(DType::Int(4))));
    assert_eq!(
        v.read().unwrap().to_sequences::<i64>().unwrap(),
        vec![vec![1], vec![2, 3], vec![], vec![4, 5, 6, 7]]
    );
}

#[test]
fn reads_a_slice_of_a_sequence_variable() {
    let file = NetcdfFile::open(VLEN_SEQ).unwrap();
    let v = file.variable("/rows").unwrap();
    // A ragged array has no netCDF read of its own, so go through `Values`.
    let slab = oxcdf::Hyperslab::new(vec![2], vec![2], &v.shape).unwrap();
    let got = v.read_selection(&slab).unwrap().to_sequences::<f64>().unwrap();
    assert_eq!(got, vec![vec![3.5, 4.5, 5.5], vec![-0.25]]);
}

#[test]
fn sequence_variables_keep_their_attributes() {
    let file = NetcdfFile::open(VLEN_SEQ).unwrap();
    let v = file.variable("/rows").unwrap();
    assert_eq!(
        v.attribute("long_name").unwrap().value.as_text(),
        Some("ragged rows")
    );
}

/// Asking a sequence variable for strings, or the reverse, must fail rather
/// than reinterpret the descriptors.
#[test]
fn sequence_and_string_accessors_do_not_mix() {
    let seq_file = NetcdfFile::open(VLEN_SEQ).unwrap();
    let seq = seq_file.variable("/rows").unwrap().read().unwrap();
    assert!(seq.to_strings().is_err());

    let str_file = NetcdfFile::open(VLEN).unwrap();
    let text = str_file.variable("/station_name").unwrap().read().unwrap();
    assert!(text.to_sequences::<f64>().is_err());
}

// ── the fallback contract, on a real szip file ────────────────────────────

const SZIP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_files/szip_test.h5");

/// szip is not decoded. What matters is that an undecodable filter behaves
/// predictably: the dataset is still described, it reports itself unreadable
/// before any I/O happens, and a read fails loudly instead of returning bytes
/// that only look plausible.
///
/// This is the contract a caller relies on to route one variable to netcdf-c
/// while reading the rest natively.
#[test]
fn an_szip_dataset_is_described_but_refuses_to_read() {
    use oxcdf::index::Hdf5File;
    use oxcdf::read::{read_hyperslab, Hyperslab};

    let file = Hdf5File::open(SZIP).expect("the file itself must still open");
    let d = file
        .dataset("/szipped")
        .expect("an szip dataset must still be listed");

    // Structure is fully available even though the values are not.
    assert_eq!(d.shape, vec![64, 8]);
    assert_eq!(d.element_size(), 4);
    assert_eq!(d.chunks(file.ctx()).unwrap().unwrap().len(), 4, "16x8 chunks over 64x8");

    // The filter is recognised, just not decoded.
    assert!(
        d.pipeline.filters.iter().any(|f| f.id == id::SZIP),
        "the pipeline should name szip"
    );

    // Refusal happens at plan time, before any read.
    assert!(
        !d.is_readable(),
        "callers must be able to route around it without reading first"
    );

    let err = read_hyperslab(file.ctx(), d, &Hyperslab::all(&d.shape))
        .expect_err("reading szip data must fail rather than return wrong bytes");
    assert!(
        err.is_fallback_worthy(),
        "the failure must be classed as unsupported, not as corruption: {err:?}"
    );
}
