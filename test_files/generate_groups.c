/* Generates `nested_groups.h5`, a fixture for a plain HDF5 file with groups.
 *
 * The netCDF corpus in this repository is written by netcdf-c, so every axis
 * has a dimension scale and every file is one group deep. Files written by an
 * instrument are neither. They nest groups several levels, and they name no
 * dimension at all, so netCDF has to invent one for every axis.
 *
 * That invention has rules, and this fixture pins each one:
 *
 *   - the number is a counter over the whole file, and a group's children are
 *     numbered before the group's own variables (`inner` takes 0)
 *   - one dimension serves every axis of that length in its own group
 *   - a dimension does not serve two axes of one variable (`square`)
 *   - a group does not reach its parent's dimensions (`inner/six`)
 *   - an axis of length zero gets a dimension of its own each time, because
 *     netCDF has no fixed dimension of length zero (`empty_a`, `empty_b`)
 *   - a growable axis matches only another growable one (`growable`)
 *   - an axis with no scale of its own still lands on a named dimension of the
 *     right length (`uses_scale`)
 *
 * It also carries attributes on a group and on a variable at each depth. The
 * netCDF corpus is one group deep, so nothing else here reads an attribute out
 * of a nested group.
 *
 * `ncdump -h nested_groups.h5` is the reference. The reader must agree with it.
 *
 * Rebuild with:
 *   h5cc -lhdf5_hl -o generate_groups generate_groups.c && ./generate_groups
 */

#include "hdf5.h"
#include "hdf5_hl.h"
#include <stdlib.h>
#include <string.h>

/* One integer dataset, filled with its own index. */
static void mkint(hid_t loc, const char *name, int rank, hsize_t *dims,
                  hsize_t *max) {
    hid_t dcpl = H5P_DEFAULT;
    hsize_t chunk[4];
    hsize_t count = 1;
    int i;

    for (i = 0; i < rank; i++) {
        count *= dims[i];
        chunk[i] = dims[i] ? dims[i] : 1;
    }

    /* HDF5 stores a growable dataset chunked, never contiguous. */
    if (max) {
        for (i = 0; i < rank; i++) {
            if (max[i] == H5S_UNLIMITED) {
                dcpl = H5Pcreate(H5P_DATASET_CREATE);
                H5Pset_chunk(dcpl, rank, chunk);
                break;
            }
        }
    }

    hid_t space = H5Screate_simple(rank, dims, max);
    hid_t dset = H5Dcreate2(loc, name, H5T_STD_I32LE, space, H5P_DEFAULT, dcpl,
                            H5P_DEFAULT);
    if (count > 0) {
        int *buf = (int *)malloc((size_t)count * sizeof(int));
        hsize_t k;
        for (k = 0; k < count; k++) {
            buf[k] = (int)k;
        }
        H5Dwrite(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
        free(buf);
    }
    H5Dclose(dset);
    H5Sclose(space);
    if (dcpl != H5P_DEFAULT) {
        H5Pclose(dcpl);
    }
}

/* One text attribute. */
static void attr_text(hid_t obj, const char *name, const char *value) {
    hid_t type = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, strlen(value));
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(obj, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, type, value);
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(type);
}

/* One attribute holding `n` doubles. */
static void attr_doubles(hid_t obj, const char *name, const double *v,
                         hsize_t n) {
    hid_t space = H5Screate_simple(1, &n, NULL);
    hid_t attr = H5Acreate2(obj, name, H5T_IEEE_F64LE, space, H5P_DEFAULT,
                            H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_DOUBLE, v);
    H5Aclose(attr);
    H5Sclose(space);
}

/* One scalar integer attribute. */
static void attr_int(hid_t obj, const char *name, int value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(obj, name, H5T_STD_I32LE, space, H5P_DEFAULT,
                            H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &value);
    H5Aclose(attr);
    H5Sclose(space);
}

/* Attach one to a dataset by name. */
static void attr_on_dataset(hid_t loc, const char *dataset, const char *name,
                            const char *units, double scale) {
    hid_t d = H5Dopen2(loc, dataset, H5P_DEFAULT);
    attr_text(d, name, units);
    attr_doubles(d, "valid_range", (double[]){-1.0, 0.0, 1.0}, 3);
    attr_int(d, "_Nkeep", (int)scale);
    H5Dclose(d);
}

int main(void) {
    hid_t file =
        H5Fcreate("nested_groups.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t four[1] = {4};
    hsize_t six[1] = {6};
    hsize_t four_four[2] = {4, 4};
    hsize_t six_four[2] = {6, 4};
    hsize_t none[1] = {0};
    hsize_t grow[1] = {3};
    hsize_t grow_max[1] = {H5S_UNLIMITED};

    /* ── /outer, holding /outer/inner ────────────────────────────────────── */
    hid_t outer = H5Gcreate2(file, "outer", H5P_DEFAULT, H5P_DEFAULT,
                             H5P_DEFAULT);
    {
        /* The child is numbered first, so `six` here takes phony_dim_0. Its 6
         * is its own: `outer/pair` below gets a separate dimension. */
        hid_t inner = H5Gcreate2(outer, "inner", H5P_DEFAULT, H5P_DEFAULT,
                                 H5P_DEFAULT);
        mkint(inner, "six", 1, six, NULL);
        /* Two levels down: a group attribute and a variable attribute. */
        attr_text(inner, "note", "two levels down");
        attr_on_dataset(inner, "six", "units", "m", 3);
        H5Gclose(inner);

        /* Two axes of one length in one variable need two dimensions. A third
         * variable of that length reuses the first. */
        mkint(outer, "square", 2, four_four, NULL);
        mkint(outer, "flat", 1, four, NULL);
        mkint(outer, "pair", 2, six_four, NULL);
        attr_text(outer, "note", "one level down");
        attr_int(outer, "level", 1);
        attr_on_dataset(outer, "pair", "units", "s", 7);
    }
    H5Gclose(outer);

    /* ── /edges: empty and growable axes ─────────────────────────────────── */
    hid_t edges = H5Gcreate2(file, "edges", H5P_DEFAULT, H5P_DEFAULT,
                             H5P_DEFAULT);
    {
        /* Both are fixed and empty. netCDF reports each as unlimited, so
         * neither can serve the other, and each gets its own dimension. */
        mkint(edges, "empty_a", 1, none, NULL);
        mkint(edges, "empty_b", 1, none, NULL);

        /* Growable, current length 3. `fixed_three` has the same length but
         * cannot grow, so it does not share the dimension. */
        mkint(edges, "growable", 1, grow, grow_max);
        mkint(edges, "fixed_three", 1, grow, NULL);
        attr_on_dataset(edges, "growable", "units", "count", 5);
    }
    H5Gclose(edges);

    /* ── /scaled: a named dimension beside a plain dataset ───────────────── */
    hid_t scaled = H5Gcreate2(file, "scaled", H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT);
    {
        mkint(scaled, "row", 1, four, NULL);
        hid_t scale = H5Dopen2(scaled, "row", H5P_DEFAULT);
        H5DSset_scale(scale, "row");
        H5Dclose(scale);

        /* No scale is attached to this one. Its axis is 4 long, so it lands on
         * `row` rather than on a dimension of its own. */
        mkint(scaled, "uses_scale", 1, four, NULL);
    }
    H5Gclose(scaled);

    /* ── the root's own variable, numbered after every group ─────────────── */
    mkint(file, "top", 1, six, NULL);
    attr_text(file, "title", "nested groups fixture");
    attr_doubles(file, "bounds", (double[]){0.0, 1.5, -2.5}, 3);
    attr_on_dataset(file, "top", "units", "K", 11);

    H5Fclose(file);
    return 0;
}
