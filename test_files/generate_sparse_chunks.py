# Generates `sparse_chunks.nc`, a chunked file most of which was never written.
#
# A chunk that was never written has no bytes on disk. It must read back as the
# variable's fill value, and netCDF's fill values are not zero. A read that
# assumes its chunks cover the selection returns silently wrong numbers here.
#
# Rebuild with:
#   python3 generate_sparse_chunks.py
import numpy as np
from netCDF4 import Dataset

with Dataset("sparse_chunks.nc", "w", format="NETCDF4") as ds:
    ds.createDimension("y", 8)
    ds.createDimension("x", 8)

    # Four chunks of 4x4. Only the first is written, so three are absent.
    part = ds.createVariable(
        "part", "i4", ("y", "x"), chunksizes=(4, 4), fill_value=-2147483647
    )
    part[0:4, 0:4] = np.arange(16, dtype=np.int32).reshape(4, 4)

    # Every chunk written. The fill value must never appear in a read of this.
    whole = ds.createVariable(
        "whole", "f4", ("y", "x"), chunksizes=(4, 4), fill_value=-9999.0
    )
    whole[:, :] = np.arange(64, dtype=np.float32).reshape(8, 8)

    # One row of chunks written, so a hyperslab can straddle the boundary
    # between stored data and fill value.
    rows = ds.createVariable(
        "rows", "i2", ("y", "x"), chunksizes=(4, 4), fill_value=-32767
    )
    rows[0:4, :] = np.arange(32, dtype=np.int16).reshape(4, 8)
