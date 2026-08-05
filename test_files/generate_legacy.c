/* Generates `legacy_v1_objheader.h5`, a fixture for the oldest on-disk layout.
 *
 * The netCDF corpus in this repository is written by a modern netcdf-c, which
 * always tracks attribute creation order. That forces version 2 object headers
 * and new-style group links, so the corpus never exercises:
 *
 *   - version 1 object headers (no OHDR signature, 8-byte message alignment)
 *   - symbol table groups (SNOD nodes in a version 1 B-tree)
 *   - local heaps (HEAP), which hold symbol table link names
 *
 * Plenty of netCDF-4 files in the wild still use that layout, so the reader
 * supports it and needs a real HDF5-written file to test against.
 *
 * Pinning the library bounds to EARLIEST is what selects the old layout.
 *
 * Rebuild with:
 *   h5cc -o generate_legacy generate_legacy.c && ./generate_legacy
 */

#include "hdf5.h"
#include <string.h>

#define NX 40
#define NY 6
#define NSTR 5
#define STRLEN 8

int main(void) {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    /* This is the whole point of the fixture. HDF5 1.14 rejects an EARLIEST
     * upper bound, so pin the pair it does accept; the low bound is what
     * selects the old layout. */
    H5Pset_libver_bounds(fapl, H5F_LIBVER_EARLIEST, H5F_LIBVER_V18);

    hid_t file = H5Fcreate("legacy_v1_objheader.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    /* ── contiguous little-endian f64, 2-D ───────────────────────────────── */
    {
        hsize_t dims[2] = {NX, NY};
        hid_t space = H5Screate_simple(2, dims, NULL);
        hid_t dset = H5Dcreate2(file, "contig_f64", H5T_IEEE_F64LE, space,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        double buf[NX * NY];
        for (int i = 0; i < NX * NY; i++) buf[i] = i * 0.5;
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
        H5Dclose(dset);
        H5Sclose(space);
    }

    /* ── chunked + shuffle + deflate i32, 2-D ────────────────────────────── */
    {
        hsize_t dims[2] = {NX, NY};
        hsize_t chunk[2] = {7, 4}; /* deliberately not a divisor of dims */
        hid_t space = H5Screate_simple(2, dims, NULL);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 2, chunk);
        H5Pset_shuffle(dcpl);
        H5Pset_deflate(dcpl, 6);
        hid_t dset = H5Dcreate2(file, "chunked_i32", H5T_STD_I32LE, space,
                                H5P_DEFAULT, dcpl, H5P_DEFAULT);
        int buf[NX * NY];
        for (int i = 0; i < NX * NY; i++) buf[i] = i * 3 - 100;
        H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
        H5Dclose(dset);
        H5Pclose(dcpl);
        H5Sclose(space);
    }

    /* ── contiguous big-endian f32, 1-D ──────────────────────────────────── */
    {
        hsize_t dims[1] = {NX};
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dset = H5Dcreate2(file, "contig_f32be", H5T_IEEE_F32BE, space,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        float buf[NX];
        for (int i = 0; i < NX; i++) buf[i] = (float)i * -1.25f;
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
        H5Dclose(dset);
        H5Sclose(space);
    }

    /* ── fixed-length strings, 1-D ───────────────────────────────────────── */
    {
        hsize_t dims[1] = {NSTR};
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t stype = H5Tcopy(H5T_C_S1);
        H5Tset_size(stype, STRLEN);
        H5Tset_strpad(stype, H5T_STR_NULLPAD);
        hid_t dset = H5Dcreate2(file, "fixed_strings", stype, space,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        char buf[NSTR * STRLEN];
        memset(buf, 0, sizeof(buf));
        const char *words[NSTR] = {"alpha", "beta", "gamma", "delta", "epsilon"};
        for (int i = 0; i < NSTR; i++)
            strncpy(buf + i * STRLEN, words[i], STRLEN);
        H5Dwrite(dset, stype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
        H5Dclose(dset);
        H5Tclose(stype);
        H5Sclose(space);
    }

    /* ── a subgroup holding one dataset, so group traversal must recurse ─── */
    {
        hid_t grp = H5Gcreate2(file, "subgroup", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims[1] = {NY};
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dset = H5Dcreate2(grp, "nested_i16", H5T_STD_I16LE, space,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        short buf[NY];
        for (int i = 0; i < NY; i++) buf[i] = (short)(1000 + i);
        H5Dwrite(dset, H5T_NATIVE_SHORT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
        H5Dclose(dset);
        H5Sclose(space);
        H5Gclose(grp);
    }

    /* ── attributes: a scalar string on the root, a float array on a dataset ── */
    {
        hid_t stype = H5Tcopy(H5T_C_S1);
        /* "legacy fixture" is 14 characters plus the terminator. */
        H5Tset_size(stype, 15);
        hid_t sspace = H5Screate(H5S_SCALAR);
        hid_t attr = H5Acreate2(file, "title", stype, sspace, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, stype, "legacy fixture");
        H5Aclose(attr);
        H5Sclose(sspace);
        H5Tclose(stype);

        hid_t dset = H5Dopen2(file, "contig_f64", H5P_DEFAULT);
        hsize_t adims[1] = {3};
        hid_t aspace = H5Screate_simple(1, adims, NULL);
        hid_t a2 = H5Acreate2(dset, "valid_range", H5T_IEEE_F64LE, aspace,
                              H5P_DEFAULT, H5P_DEFAULT);
        double range[3] = {-1.0, 0.0, 1.0};
        H5Awrite(a2, H5T_NATIVE_DOUBLE, range);
        H5Aclose(a2);
        H5Sclose(aspace);
        H5Dclose(dset);
    }

    H5Fclose(file);
    H5Pclose(fapl);
    return 0;
}
