/* Generates `latest_v4_layout.h5`, exercising version 4 data layout messages.
 *
 * Pinning the library to its latest format makes HDF5 write version 4 layouts,
 * which select a chunk index by dataset shape:
 *   - one chunk covering everything  -> "single chunk"
 *   - unfiltered, fixed maximum dims -> "implicit"
 *   - filtered, fixed maximum dims   -> "fixed array"
 *   - one unlimited dimension        -> "extensible array"
 *   - several unlimited dimensions   -> "version 2 B-tree"
 *
 * Rebuild with:
 *   h5cc -o generate_latest generate_latest.c && ./generate_latest
 */
#include "hdf5.h"

#define NX 40
#define NY 6

static void write_i32(hid_t file, const char *name, hid_t dcpl, hsize_t *maxdims) {
    hsize_t dims[2] = {NX, NY};
    hid_t space = H5Screate_simple(2, dims, maxdims);
    hid_t dset = H5Dcreate2(file, name, H5T_STD_I32LE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    int buf[NX * NY];
    for (int i = 0; i < NX * NY; i++) buf[i] = i * 3 - 100;
    H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
    H5Dclose(dset);
    H5Sclose(space);
}

int main(void) {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
    hid_t file = H5Fcreate("latest_v4_layout.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

    hsize_t whole[2] = {NX, NY};
    hsize_t small[2] = {7, 4};
    hsize_t one_unlim[2] = {H5S_UNLIMITED, NY};
    hsize_t two_unlim[2] = {H5S_UNLIMITED, H5S_UNLIMITED};

    /* single chunk: one chunk covers the whole dataset */
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 2, whole);
    write_i32(file, "single_chunk", dcpl, NULL);
    H5Pclose(dcpl);

    /* implicit: unfiltered, early allocation, fixed dims */
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 2, small);
    H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY);
    H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
    write_i32(file, "implicit", dcpl, NULL);
    H5Pclose(dcpl);

    /* fixed array: filtered, fixed maximum dims */
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 2, small);
    H5Pset_shuffle(dcpl);
    H5Pset_deflate(dcpl, 6);
    write_i32(file, "fixed_array", dcpl, NULL);
    H5Pclose(dcpl);

    /* extensible array: exactly one unlimited dimension */
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 2, small);
    write_i32(file, "extensible_array", dcpl, one_unlim);
    H5Pclose(dcpl);

    /* version 2 B-tree: more than one unlimited dimension */
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 2, small);
    write_i32(file, "btree2_index", dcpl, two_unlim);
    H5Pclose(dcpl);

    H5Fclose(file);
    H5Pclose(fapl);
    return 0;
}
