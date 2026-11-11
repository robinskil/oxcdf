# Generates the fixtures of `crates/oxcdf/tests/netcdf_c_parity.rs`.
#
# Each file holds one case where this reader gave a different result from
# netcdf-c.
#
# Rebuild with:
#   python3 generate_netcdf_c_parity.py
import h5py
import numpy as np
from netCDF4 import Dataset

# CDF-5 stores the record count in 64 bits.
with Dataset("cdf5_records.nc", "w", format="NETCDF3_64BIT_DATA") as ds:

    ds.createDimension("time", None)
    ds.createDimension("x", 3)
    rec = ds.createVariable("rec", "i8", ("time",))
    rec[:] = np.array([-(2**62), 0, 1, 2**62], dtype="i8")
    rec.units = "count"
    fixed = ds.createVariable("fixed", "u8", ("x",))
    fixed[:] = np.array([0, 2**63, 2**64 - 2], dtype="u8")

# A variable with the name of a dimension, on other dimensions. netcdf-c stores
# it under `_nc4_non_coord_<name>`.
with Dataset("non_coord_names.nc", "w", format="NETCDF4") as ds:

    ds.createDimension("b", 2)
    ds.createDimension("a", 3)
    b = ds.createVariable("b", "i4", ("a",))
    b[:] = [1, 2, 3]
    b.units = "count"
    a = ds.createVariable("a", "f4", ("b", "a"))
    a[:] = np.arange(6).reshape(2, 3)
    grp = ds.createGroup("grp")
    inner = grp.createVariable("b", "i2", ("a",))
    inner[:] = [7, 8, 9]

# A rename in a group with many links leaves a freed record in the link heap.
with Dataset("renamed_variable.nc", "w", format="NETCDF4") as ds:
    ds.createDimension("x", 2)
    for i in range(12):
        v = ds.createVariable("v%02d" % i, "i4", ("x",))
        v[:] = [i, -i]
    ds["v00"].units = "count"
    ds.renameVariable("v01", "renamed")

# An attribute above 64 KiB does not fit a heap block.
with Dataset("huge_attribute.nc", "w", format="NETCDF4") as ds:
    ds.createDimension("x", 2)
    ds.big = "x" * 66000
    v = ds.createVariable("v", "i4", ("x",))
    v.big = "y" * 66000
    v.small = "s"

# Soft links to a dataset, in an old-style group and a new-style group.
for name, libver in (("soft_link.h5", "earliest"), ("soft_link_latest.h5", "latest")):
    with h5py.File(name, "w", libver=libver) as h:

        h.create_dataset("a", data=np.arange(3, dtype="i4"))
        grp = h.create_group("grp")
        grp.create_dataset("b", data=np.arange(2, dtype="f8"))
        grp["b"].attrs["scale"] = np.float64(2.0)
        h["a"].attrs["scale"] = np.float64(1.0)
        h["soft"] = h5py.SoftLink("/grp/b")
        grp["rel"] = h5py.SoftLink("b")
        h["chain"] = h5py.SoftLink("soft")
