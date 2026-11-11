# Generates `unlimited_uneven.nc`, where variables on one unlimited dimension
# hold different numbers of records.
#
# HDF5 stores only the records a variable holds. netCDF reports the longest
# length for every variable on the dimension, and a read past a shorter one
# returns its fill value.
#
# Rebuild with:
#   python3 generate_unlimited_uneven.py
import numpy as np
from netCDF4 import Dataset

with Dataset("unlimited_uneven.nc", "w", format="NETCDF4") as ds:
    # `time` has a coordinate variable, so its dimension scale holds 6 records.
    ds.createDimension("time", None)
    ds.createDimension("x", 3)
    time = ds.createVariable("time", "f8", ("time",))
    time[:] = np.arange(6)

    # No record written.
    empty = ds.createVariable("empty", "f4", ("time",), fill_value=-999.0)
    # Two of six records written.
    short = ds.createVariable("short", "i4", ("time",), fill_value=-1)
    short[0:2] = [10, 11]
    # Two of six rows written, on a second fixed axis.
    grid = ds.createVariable("grid", "i2", ("time", "x"))
    grid[0:2, :] = np.arange(6, dtype=np.int16).reshape(2, 3)
    # Big-endian storage, one record written.
    big = ds.createVariable("big", "f8", ("time",), endian="big", fill_value=-5.0)
    big[0] = 1.5
    # Strings, one record written.
    names = ds.createVariable("names", str, ("time",))
    names[0] = "first"

    # `obs` has no coordinate variable. Its dimension scale holds no records.
    ds.createDimension("obs", None)
    a = ds.createVariable("a", "f4", ("obs",), fill_value=-1.0)
    a[0:4] = np.arange(4, dtype=np.float32)
    b = ds.createVariable("b", "f4", ("obs",), fill_value=-2.0)

    # A child group sees the dimensions of its parent.
    child = ds.createGroup("child")
    nested = child.createVariable("nested", "i4", ("time",), fill_value=-3)
    nested[0] = 7
