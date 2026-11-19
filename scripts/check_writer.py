"""Check files from the oxcdf writer with other HDF5 and netCDF readers.

The script writes 80 random files and their models with `cargo test`. Then it
reads each file with h5py, h5dump, ncdump, netCDF4 and xarray, and compares the
content with the model. It also lets the HDF5 library change each file, and
then the oxcdf reader reads the changed files.

It does the same for one file from the netCDF writer: netCDF4 and xarray read
it, netcdf-c appends to it, and the oxcdf reader reads the result.

Usage:
    python scripts/check_writer.py [--dir DIRECTORY]

Without `--dir` the files go to a temporary directory.

Needs: h5py and numpy. Optional: netCDF4, xarray, the `h5dump` and `ncdump`
tools on the PATH, and the Zstandard plugin for HDF5. The script reports each
optional part that it does not find.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

# The netCDF4 package sets the HDF5 plugin path at its import, so it comes before h5py.
try:
    import netCDF4
except ImportError:
    netCDF4 = None

import h5py
import numpy as np

try:
    import xarray
except ImportError:
    xarray = None

# Whether the HDF5 library of h5py finds the Zstandard plugin.
ZSTD = h5py.h5z.filter_avail(32015)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDED = "added_by_h5py"

problems = []
counts = {}


def note(case, message):
    problems.append(f"{case}: {message}")


def count(key):
    counts[key] = counts.get(key, 0) + 1


def cargo(test, directory, package="oxcdf-hdf5", test_file="write_model"):
    """Run one ignored test of a test file with the export directory."""
    env = dict(os.environ, OXCDF_EXPORT_DIR=directory)
    command = ["cargo", "test", "-p", package, "--features", "write", "--test", test_file,
               "--", "--ignored", "--exact", test]
    result = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"cargo test {test} failed:\n{result.stdout}\n{result.stderr}")


def creation_order(obj, links):
    names = []
    if links:
        obj.id.links.iterate(lambda n: names.append(n.decode()), idx_type=h5py.h5.INDEX_CRT_ORDER)
    else:
        h5py.h5a.iterate(obj.id, lambda n: names.append(n.decode()), index_type=h5py.h5.INDEX_CRT_ORDER)
    return names


def check_attributes(obj, attrs, case, added):
    """Compare the attributes of one object with the model, in creation order."""
    want = [name for name, _ in attrs] + ([ADDED] if added else [])
    got = creation_order(obj, links=False)
    if got != want:
        note(case, f"attributes of {obj.name}: {got} != {want}")
        return
    for name, data in attrs:
        count("attributes")
        raw = obj.attrs[name]
        if isinstance(data, list):
            # An attribute of variable-length strings has rank 1, as in netcdf-c.
            if np.ndim(raw) != 1 or [str(v) for v in raw] != data:
                note(case, f"string attribute {obj.name}@{name}: {raw!r}")
            continue
        data = bytes.fromhex(data)
        attribute = h5py.h5a.open(obj.id, name.encode())
        if attribute.get_type().get_class() == h5py.h5t.STRING:
            # A text attribute is one fixed string with no NUL, as in netcdf-c.
            text = raw.encode() if isinstance(raw, str) else bytes(raw)
            if text != data or attribute.get_type().get_size() != len(data):
                note(case, f"text attribute {obj.name}@{name}: {text!r} != {data!r}")
            if attribute.get_space().get_simple_extent_type() != h5py.h5s.SCALAR:
                note(case, f"text attribute {obj.name}@{name} is not scalar")
        else:
            array = np.asarray(raw)
            if array.tobytes() != data or array.ndim != 1:
                note(case, f"attribute {obj.name}@{name}: shape {array.shape}")


def check_h5py(path, model, case, added=False):
    """Compare each object of the file with the model through the HDF5 library."""
    with h5py.File(path, "r") as f:
        for g in model["groups"]:
            count("groups")
            group = f[g["path"]]
            prefix = g["path"].rstrip("/") + "/"
            datasets = [d["path"][len(prefix):] for d in model["datasets"]
                        if d["path"].startswith(prefix) and "/" not in d["path"][len(prefix):]]
            want = sorted(g["groups"] + datasets + ([ADDED] if added else []))
            if sorted(group.keys()) != want:
                note(case, f"children of {g['path']}: {sorted(group.keys())} != {want}")
            if sorted(creation_order(group, links=True)) != want:
                note(case, f"links of {g['path']} by creation order")
            groups = [n for n in creation_order(group, links=True) if isinstance(group[n], h5py.Group)]
            if groups != g["groups"]:
                note(case, f"group order in {g['path']}: {groups} != {g['groups']}")
            check_attributes(group, g["attrs"], case, added)
        for d in model["datasets"]:
            count("datasets")
            dataset = f[d["path"]]
            chunks = None if d["chunks"] is None else tuple(d["chunks"])
            maxshape = dataset.shape if d["maxshape"] is None else tuple(d["maxshape"])
            if dataset.chunks != chunks or dataset.maxshape != maxshape:
                note(case, f"{d['path']}: chunks {dataset.chunks}, largest shape {dataset.maxshape}")
            if chunks is not None:
                count("chunked datasets")
            filters = d["filters"]
            if dataset.shuffle != ("shuffle" in filters) or dataset.fletcher32 != ("fletcher32" in filters):
                note(case, f"{d['path']}: shuffle {dataset.shuffle}, fletcher32 {dataset.fletcher32}")
            if (dataset.compression == "gzip") != ("deflate" in filters):
                note(case, f"{d['path']}: compression {dataset.compression}")
            if ("32015" in dataset._filters) != ("zstd" in filters):
                note(case, f"{d['path']}: filters {dataset._filters}")
            if filters:
                count("datasets with filters")
            if "zstd" in filters and not ZSTD:
                counts.setdefault("Zstandard plugin for h5py not found", 0)
                continue
            if d["dtype"] == "str":
                info = h5py.check_string_dtype(dataset.dtype)
                if info is None or info.length is not None or list(dataset.shape) != d["shape"]:
                    note(case, f"{d['path']}: {dataset.dtype} {dataset.shape}")
                elif np.asarray(dataset.asstr()[()]).ravel().tolist() != d["values"]:
                    note(case, f"{d['path']}: strings differ")
                count("string datasets")
                check_attributes(dataset, d["attrs"], case, added)
                continue
            if dataset.dtype.str != d["dtype"] or list(dataset.shape) != d["shape"]:
                note(case, f"{d['path']}: {dataset.dtype.str} {dataset.shape}")
                continue
            if float(dataset.fillvalue) != float(d["fill"]):
                note(case, f"{d['path']}: fill value {dataset.fillvalue} != {d['fill']}")
            got = np.asarray(dataset[()]).astype(np.float64).ravel().tolist()
            if got != [float(v) for v in d["values"]]:
                note(case, f"{d['path']}: values differ")
            check_attributes(dataset, d["attrs"], case, added)


TOOLS_WITH_ZSTD = {}


def tool_reads_zstd(tool):
    """Whether `tool` reads a Zstandard file from the HDF5 library. One probe for each tool."""
    if tool not in TOOLS_WITH_ZSTD:
        result = False
        if ZSTD:
            probe = os.path.join(tempfile.mkdtemp(prefix="oxcdf-zstd-"), "probe.h5")
            with h5py.File(probe, "w") as f:
                f.create_dataset("z", data=np.arange(64), chunks=(8,), compression=32015, compression_opts=(3,))
            run = subprocess.run([tool, probe], capture_output=True, text=True, errors="replace")
            result = run.returncode == 0 and "error" not in run.stderr.lower()
            shutil.rmtree(os.path.dirname(probe), ignore_errors=True)
        TOOLS_WITH_ZSTD[tool] = result
    return TOOLS_WITH_ZSTD[tool]


def check_tool(command, case, zstd=False):
    tool = command[0]
    if shutil.which(tool) is None:
        counts.setdefault(f"{tool} not found", 0)
        return
    if zstd and not tool_reads_zstd(tool):
        counts.setdefault(f"{tool} has no Zstandard plugin: files with that filter", 0)
        return
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or "error" in result.stderr.lower():
        note(case, f"{tool} exit {result.returncode}: {result.stderr.strip()[:300]}")
    else:
        count(tool)


def check_netcdf4(path, case):
    """Read each variable and each attribute through netcdf-c."""
    if netCDF4 is None:
        counts.setdefault("netCDF4 not found", 0)
        return

    def walk(group):
        for variable in group.variables.values():
            variable.set_auto_maskandscale(False)
            variable[...]
            for name in variable.ncattrs():
                variable.getncattr(name)
        for name in group.ncattrs():
            group.getncattr(name)
        for child in group.groups.values():
            walk(child)

    try:
        with netCDF4.Dataset(path, "r") as dataset:
            walk(dataset)
        count("netCDF4")
    except Exception as error:  # noqa: BLE001
        note(case, f"netCDF4: {type(error).__name__}: {error}")


def check_xarray(path, model, case):
    """Open each group as an xarray dataset and compare each variable with the model."""
    if xarray is None or netCDF4 is None:
        counts.setdefault("xarray not found", 0)
        return
    try:
        for g in model["groups"]:
            prefix = g["path"].rstrip("/") + "/"
            options = {"engine": "netcdf4", "mask_and_scale": False, "decode_times": False}
            with xarray.open_dataset(path, group=g["path"], **options) as dataset:
                dataset.load()
                for d in model["datasets"]:
                    name = d["path"][len(prefix):]
                    if not d["path"].startswith(prefix) or "/" in name:
                        continue
                    count("xarray variables")
                    variable = dataset[name]
                    if d["dtype"] == "str":
                        if np.asarray(variable.values).ravel().tolist() != d["values"]:
                            note(case, f"xarray {d['path']}: strings differ")
                        continue
                    got = variable.values.astype(np.float64).ravel().tolist()
                    if list(variable.shape) != d["shape"] or got != [float(v) for v in d["values"]]:
                        note(case, f"xarray {d['path']}: values differ")
                    if variable.dtype.str.lstrip("<>|=") != d["dtype"].lstrip("<>|="):
                        note(case, f"xarray {d['path']}: dtype {variable.dtype.str}")
                    for attribute, _ in d["attrs"]:
                        if attribute not in variable.attrs:
                            note(case, f"xarray {d['path']}: no attribute {attribute}")
        count("xarray")
    except Exception as error:  # noqa: BLE001
        note(case, f"xarray: {type(error).__name__}: {error}")


def change_with_library(path, model, case, zstd):
    """Add one attribute to each object and one dataset to each group with the HDF5 library."""
    changed = path.replace(".h5", "_mod.h5")
    shutil.copyfile(path, changed)
    try:
        with h5py.File(changed, "r+") as f:
            for g in model["groups"]:
                group = f[g["path"]]
                group.attrs.create(ADDED, np.array([7], dtype="<i4"))
                group.create_dataset(ADDED, data=np.array([1, 2, 3], dtype="<i4"))
            for d in model["datasets"]:
                dataset = f[d["path"]]
                dataset.attrs.create(ADDED, np.array([7], dtype="<i4"))
                if dataset.size > 0:
                    first = tuple(0 for _ in dataset.shape)
                    dataset[first] = dataset[first]
        check_h5py(changed, model, case + " (changed)", added=True)
        check_tool(["h5dump", changed], case + " (changed)", zstd)
        count("changed by the library")
    except Exception as error:  # noqa: BLE001
        note(case, f"the HDF5 library cannot change the file: {type(error).__name__}: {error}")


def check_strings(path):
    """Compare the fixed and the variable-length string datasets with their known content."""
    case = "strings"
    many = [f"string {i}" for i in range(3000)]
    with h5py.File(path, "r") as f:
        names = f["names"]
        if names.dtype.str != "|S5" or names[()].tolist() != [b"alpha", b"be", b"gamma"]:
            note(case, f"names: {names.dtype.str} {names[()].tolist()}")
        chars = f["chars"]
        if chars.dtype.str != "|S1" or chars[()].tolist() != [[b""] * 4, [b"w", b"x", b"y", b"z"]]:
            note(case, f"chars: {chars.dtype.str} {chars[()].tolist()}")
        padded = f["padded"]
        pad = padded.id.get_type().get_strpad()
        # h5py removes the pad spaces.
        if pad != h5py.h5t.STR_SPACEPAD or padded[()].tolist() != [b"wxyz", b"ab"]:
            note(case, f"padded: pad {pad} {padded[()].tolist()}")
        if f["names"].attrs["long_name"] not in (b"station names", "station names"):
            note(case, "attribute long_name")

        vlen = f["vlen"]
        info = h5py.check_string_dtype(vlen.dtype)
        if info is None or info.length is not None or info.encoding != "utf-8":
            note(case, f"vlen: datatype {vlen.dtype} {info}")
        elif vlen.asstr()[()].tolist() != [["alpha", "", "zeeën"], ["", "", ""]]:
            note(case, f"vlen: {vlen.asstr()[()].tolist()}")
        if f["many"].asstr()[()].tolist() != many:
            note(case, "many: values differ")
        if f["long"].asstr()[0] != "x" * 100000:
            note(case, "long: value differs")
        if list(f.attrs["labels"]) != ["a", "bc"] or list(vlen.attrs["one"]) != ["solo"]:
            note(case, f"string attributes: {f.attrs['labels']} {vlen.attrs['one']}")
    count("string files")
    check_tool(["h5dump", path], case)
    check_tool(["ncdump", path], case)
    if netCDF4 is not None:
        try:
            with netCDF4.Dataset(path, "r") as dataset:
                # netcdf-c reads a string of 1 byte as the netCDF character type.
                chars = dataset["chars"]
                chars.set_auto_maskandscale(False)
                if chars.dtype.str != "|S1" or chars[1].tobytes() != b"wxyz":
                    note(case, f"netCDF4 chars: {chars.dtype.str} {chars[1].tobytes()!r}")
                # netcdf-c reads a longer fixed string as the netCDF string type.
                names = dataset["names"]
                if names.dtype is not str or names[...].tolist() != ["alpha", "be", "gamma"]:
                    note(case, f"netCDF4 names: {names.dtype} {names[...].tolist()}")
                vlen = dataset["vlen"]
                if vlen.dtype is not str or vlen[...].tolist() != [["alpha", "", "zeeën"], ["", "", ""]]:
                    note(case, f"netCDF4 vlen: {vlen.dtype} {vlen[...].tolist()}")
                if dataset["many"][...].tolist() != many:
                    note(case, "netCDF4 many: values differ")
                if dataset.getncattr("labels") != ["a", "bc"] or vlen.getncattr("one") != "solo":
                    note(case, f"netCDF4 attributes: {dataset.getncattr('labels')} {vlen.getncattr('one')}")
            count("netCDF4 string variables")
        except Exception as error:  # noqa: BLE001
            note(case, f"netCDF4: {type(error).__name__}: {error}")
    if xarray is not None and netCDF4 is not None:
        try:
            with xarray.open_dataset(path, engine="netcdf4") as dataset:
                dataset.load()
                if dataset["vlen"].values.tolist() != [["alpha", "", "zeeën"], ["", "", ""]]:
                    note(case, f"xarray vlen: {dataset['vlen'].values.tolist()}")
                if dataset["many"].values.tolist() != many:
                    note(case, "xarray many: values differ")
                if dataset["names"].values.tolist() != ["alpha", "be", "gamma"]:
                    note(case, f"xarray names: {dataset['names'].values.tolist()}")
            count("xarray string variables")
        except Exception as error:  # noqa: BLE001
            note(case, f"xarray: {type(error).__name__}: {error}")

    # The HDF5 library writes one string element and adds one string dataset.
    changed = path.replace(".h5", "_mod.h5")
    shutil.copyfile(path, changed)
    try:
        with h5py.File(changed, "r+") as f:
            f["vlen"][1, 0] = "added by h5py"
            f.create_dataset("added", data=["p", "qr"], dtype=h5py.string_dtype("utf-8"))
        check_tool(["h5dump", changed], case + " (changed)")
        count("string file changed by the library")
    except Exception as error:  # noqa: BLE001
        note(case, f"the HDF5 library cannot change the file: {type(error).__name__}: {error}")


def check_chunks(path):
    """Compare the chunked datasets with their known content, then let the library grow them."""
    case = "chunks"

    def grow_values(rows):
        values = np.arange(rows * 3, dtype="<i4").reshape(rows, 3)
        values[100:110] = 9
        return values

    fixed = np.full((10, 7), -1.0)
    fixed[2:9, 1:6] = np.arange(35).reshape(7, 5)
    packed = np.full((40, 50), -1.0, dtype="<f4")
    packed[:35] = (np.arange(35 * 50) // 7).reshape(35, 50)
    zstd = np.arange(80, dtype="<i4").reshape(10, 8)
    with h5py.File(path, "r") as f:
        grow = f["grow"]
        if (grow.shape, grow.chunks, grow.maxshape) != ((700, 3), (2, 3), (None, 3)):
            note(case, f"grow: {grow.shape} {grow.chunks} {grow.maxshape}")
        elif not np.array_equal(grow[()], grow_values(700)) or grow.fillvalue != 9:
            note(case, "grow: values differ")
        if f["fixed"].chunks != (4, 3) or f["fixed"].dtype.str != ">f8" or not np.array_equal(f["fixed"][()], fixed):
            note(case, "fixed: values differ")
        names = f["names"]
        if names.maxshape != (None,) or names.asstr()[()].tolist() != ["", "", "", "d", "", "f"]:
            note(case, f"names: {names.maxshape} {names.asstr()[()].tolist()}")
        if (f["empty"].shape, f["empty"].maxshape, f["empty"].chunks) != ((0,), (None,), (5,)):
            note(case, f"empty: {f['empty'].shape} {f['empty'].maxshape}")
        if f["two"].maxshape != (None, None) or f["two"][()].tolist() != [[1, 2], [3, 4]]:
            note(case, f"two: {f['two'].maxshape} {f['two'][()].tolist()}")
        ds = f["packed"]
        if (ds.compression, ds.compression_opts, ds.shuffle, ds.fletcher32) != ("gzip", 4, True, True):
            note(case, f"packed: {ds.compression} {ds.compression_opts} {ds.shuffle} {ds.fletcher32}")
        if not np.array_equal(ds[()], packed):
            note(case, "packed: values differ")
        if "32015" not in f["zstd"]._filters or (ZSTD and not np.array_equal(f["zstd"][()], zstd)):
            note(case, f"zstd: {f['zstd']._filters}")
    count("chunk files")
    check_tool(["h5dump", path], case, zstd=True)
    check_tool(["ncdump", path], case, zstd=True)
    if netCDF4 is not None:
        try:
            with netCDF4.Dataset(path, "r") as dataset:
                grow = dataset["grow"]
                grow.set_auto_maskandscale(False)
                unlimited = [dataset.dimensions[name].isunlimited() for name in grow.dimensions]
                if unlimited != [True, False] or not np.array_equal(grow[...], grow_values(700)):
                    note(case, f"netCDF4 grow: unlimited {unlimited}")
                if grow.chunking() != [2, 3]:
                    note(case, f"netCDF4 grow: chunks {grow.chunking()}")
                if dataset["names"][...].tolist() != ["", "", "", "d", "", "f"]:
                    note(case, "netCDF4 names: values differ")
                variable = dataset["packed"]
                variable.set_auto_maskandscale(False)
                filters = variable.filters()
                if not (filters["zlib"] and filters["shuffle"] and filters["fletcher32"]) or filters["complevel"] != 4:
                    note(case, f"netCDF4 packed: filters {filters}")
                if not np.array_equal(variable[...], packed):
                    note(case, "netCDF4 packed: values differ")
                variable = dataset["zstd"]
                variable.set_auto_maskandscale(False)
                if not variable.filters()["zstd"] or not np.array_equal(variable[...], zstd):
                    note(case, f"netCDF4 zstd: filters {variable.filters()}")
            count("netCDF4 unlimited variables")
        except Exception as error:  # noqa: BLE001
            note(case, f"netCDF4: {type(error).__name__}: {error}")
    if xarray is not None and netCDF4 is not None:
        try:
            with xarray.open_dataset(path, engine="netcdf4", mask_and_scale=False) as dataset:
                dataset.load()
                if not np.array_equal(dataset["grow"].values, grow_values(700)):
                    note(case, "xarray grow: values differ")
                if not np.array_equal(dataset["fixed"].values, fixed):
                    note(case, "xarray fixed: values differ")
                if not np.array_equal(dataset["packed"].values, packed):
                    note(case, "xarray packed: values differ")
                if not np.array_equal(dataset["zstd"].values, zstd):
                    note(case, "xarray zstd: values differ")
            count("xarray chunked variables")
        except Exception as error:  # noqa: BLE001
            note(case, f"xarray: {type(error).__name__}: {error}")

    # The HDF5 library grows the datasets, so it adds chunks to the index of the writer.
    changed = path.replace(".h5", "_mod.h5")
    shutil.copyfile(path, changed)
    try:
        with h5py.File(changed, "r+") as f:
            f["grow"].resize((1500, 3))
            f["grow"][700:] = grow_values(1500)[700:]
            f["grow"][0, 0] = -5
            f["names"].resize((8,))
            f["names"][7] = "h"
            f["empty"].resize((3,))
            f["empty"][:] = [7, 8, 9]
            # The library reads a chunk of the writer, changes it and compresses it again.
            f["packed"][0, 3] = 99.0
            if ZSTD:
                f["zstd"].resize((12, 8))
                f["zstd"][10:] = np.arange(80, 96).reshape(2, 8)
        with h5py.File(changed, "r") as f:
            want = grow_values(1500)
            want[0, 0] = -5
            if not np.array_equal(f["grow"][()], want):
                note(case, "grow: values differ after the library change")
        check_tool(["h5dump", changed], case + " (changed)", zstd=True)
        count("chunk file changed by the library")
    except Exception as error:  # noqa: BLE001
        note(case, f"the HDF5 library cannot change the file: {type(error).__name__}: {error}")


def check_netcdf(path):
    """Check the file from the netCDF writer, then change it with netcdf-c.

    Return True when netcdf-c changed the file with no problem.
    """
    case = "netcdf"
    nan = float("nan")
    temp = np.full((3, 3, 4), -999.0, dtype="f4")
    temp[0] = np.arange(12, dtype="f4").reshape(3, 4)
    temp[2, 1, :2] = [50.0, 51.0]
    before = len(problems)

    def same(name, got, expected):
        got, expected = np.asarray(got), np.asarray(expected)
        numbers = got.dtype.kind in "fiu" and expected.dtype.kind in "fiu"
        if not np.array_equal(got, expected, equal_nan=numbers):
            note(case, f"{name}: got {got.tolist()}, expected {expected.tolist()}")
        else:
            count("netCDF writer value checks")

    check_tool(["ncdump", "-h", path], case)
    if netCDF4 is None:
        counts.setdefault("netCDF4 not found: the file from the netCDF writer", 0)
        return False
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        dims = {name: (len(d), d.isunlimited()) for name, d in ds.dimensions.items()}
        expected = {"time": (3, True), "lat": (3, False), "lon": (4, False), "nv": (2, False)}
        if dims != expected:
            note(case, f"dimensions {dims}")
        names = list(ds.variables)
        if names != ["lat", "time", "TEMP", "PSAL", "code", "name", "scalar", "nv", "big"]:
            note(case, f"variable order {names}")
        shapes = {name: v.dimensions for name, v in ds.variables.items()}
        if shapes["TEMP"] != ("time", "lat", "lon") or shapes["nv"] != ("lat",) or shapes["code"] != ("lat", "nv"):
            note(case, f"variable dimensions {shapes}")
        same("lat", ds["lat"][:], [-10.0, 0.0, 10.0])
        same("time", ds["time"][:], [0.5, 1.5, netCDF4.default_fillvals["f8"]])
        same("TEMP", ds["TEMP"][:], temp)
        same("PSAL", ds["PSAL"][:], np.full((3, 3), netCDF4.default_fillvals["f8"]))
        same("code", ds["code"][:], np.array(list("abcdef"), dtype="S1").reshape(3, 2))
        same("name", ds["name"][:], np.array(["south", "equator", ""], dtype=object))
        same("scalar", ds["scalar"][...], 42)
        same("nv", ds["nv"][:], [7, 8, 9])
        if ZSTD:
            same("big", ds["big"][:], [1, 2, 3, 4])
        else:
            counts.setdefault("Zstandard plugin not found: the variable big", 0)
        temp_var = ds["TEMP"]
        filters = temp_var.filters()
        if not (filters["zlib"] and filters["shuffle"] and filters["complevel"] == 4):
            note(case, f"TEMP filters {filters}")
        if temp_var.chunking() != [1, 3, 4] or temp_var._FillValue != -999.0 or temp_var.units != "degC":
            note(case, "TEMP chunks, fill value or units")
        if ds.title != "check" or list(ds.levels) != [1, 2, 3] or list(ds.tags) != ["a", "bc"]:
            note(case, "global attributes")
        sub = ds["sub"]
        if sub.kind != "deep" or sub["deep"].dimensions != ("z", "lat") or len(sub.dimensions["z"]) != 5:
            note(case, "group sub")
        same("deep", sub["deep"][:], np.arange(15).reshape(5, 3))

    if xarray is None:
        counts.setdefault("xarray not found: the file from the netCDF writer", 0)
    else:
        with xarray.open_dataset(path, drop_variables=None if ZSTD else ["big"]) as ds:
            if dict(ds.sizes) != {"time": 3, "lat": 3, "lon": 4, "nv": 2}:
                note(case, f"xarray sizes {dict(ds.sizes)}")
            if "lat" not in ds.coords or "time" not in ds.coords:
                note(case, f"xarray coordinates {list(ds.coords)}")
            same("xarray TEMP", ds["TEMP"].values, np.where(temp == -999.0, nan, temp))
            same("xarray name", ds["name"].values.astype(object), np.array(["south", "equator", ""], dtype=object))
        with xarray.open_dataset(path, group="sub") as ds:
            same("xarray deep", ds["deep"].values, np.arange(15).reshape(5, 3))

    # netcdf-c appends records, adds objects and moves attributes to dense storage.
    try:
        with netCDF4.Dataset(path, "a") as ds:
            ds["TEMP"][3] = np.full((3, 4), 7.0, dtype="f4")
            ds["time"][2:4] = [2.5, 3.5]
            ds["name"][2] = "north"
            for i in range(12):
                ds["TEMP"].setncattr(f"extra_{i:02}", float(i))
            ds.createDimension("depth", 2)
            extra = ds.createVariable("extra", "i4", ("time", "depth"))
            extra[:] = np.arange(8).reshape(4, 2)
            ds["sub"].createVariable("more", "f8", ("z",))[:] = np.arange(5) * 0.5
            ds.history = "changed by netcdf-c"
    except Exception as error:  # noqa: BLE001
        note(case, f"netcdf-c change: {type(error).__name__}: {error}")
    return len(problems) == before


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", help="directory for the files; the default is a temporary directory")
    args = parser.parse_args()
    directory = args.dir or tempfile.mkdtemp(prefix="oxcdf-writer-")
    os.makedirs(directory, exist_ok=True)
    stale = ["*.h5", "*.nc", "case_*.json"]
    for old in [path for pattern in stale for path in glob.glob(os.path.join(directory, pattern))]:
        os.remove(old)

    cargo("export_files_for_the_external_check", directory)
    models = sorted(glob.glob(os.path.join(directory, "case_*.json")),
                    key=lambda p: int(os.path.basename(p)[5:-5]))
    for model_path in models:
        case = os.path.basename(model_path)[:-5]
        path = model_path[:-5] + ".h5"
        with open(model_path, encoding="utf-8") as handle:
            model = json.load(handle)
        count("files")
        try:
            check_h5py(path, model, case)
        except Exception as error:  # noqa: BLE001
            note(case, f"h5py: {type(error).__name__}: {error}")
            continue
        zstd = any("zstd" in d["filters"] for d in model["datasets"])
        check_tool(["h5dump", path], case, zstd)
        check_tool(["ncdump", "-h", path], case)
        check_netcdf4(path, case)
        check_xarray(path, model, case)
        change_with_library(path, model, case, zstd)
    check_strings(os.path.join(directory, "strings.h5"))
    check_chunks(os.path.join(directory, "chunks.h5"))
    cargo("export_a_netcdf_file_for_the_external_check", directory, "oxcdf", "write_netcdf")
    if check_netcdf(os.path.join(directory, "netcdf.nc")):
        # The oxcdf reader reads the file that netcdf-c changed.
        cargo("a_file_changed_by_netcdf_c_reads_back", directory, "oxcdf", "write_netcdf")
        count("file changed by netcdf-c read by oxcdf")
    if not problems:
        # The oxcdf reader reads the files that the HDF5 library changed.
        cargo("files_changed_by_the_library_read_back", directory)
        cargo("strings_changed_by_the_library_read_back", directory)
        if ZSTD:
            cargo("chunks_changed_by_the_library_read_back", directory)
        else:
            counts.setdefault("Zstandard plugin for h5py not found", 0)
        count("changed files read by oxcdf")

    for key in sorted(counts):
        print(f"{key}: {counts[key]}" if counts[key] else f"SKIPPED: {key}")
    print(f"{len(problems)} problems in {directory}")
    for problem in problems[:60]:
        print(" -", problem)
    if not args.dir and not problems:
        shutil.rmtree(directory, ignore_errors=True)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
