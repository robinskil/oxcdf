#!/usr/bin/env python
"""Round-trip check for the oxcdf writer.

Python writes many netCDF-4 files into DIR/cases.
A Rust test reads each file and writes a copy into DIR/copies.
Python then compares each copy with its original.

Usage: python scripts/check_copy.py [--dir DIR] [--skip-rust]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

# Import netCDF4 first. It sets the HDF5 plugin path.
import netCDF4
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="endian-ness of dtype")

REPO = Path(__file__).resolve().parent.parent
RUST_CMD = [
    "cargo", "test", "-p", "oxcdf", "--features", "write",
    "--test", "write_copy", "--", "--ignored", "--exact",
    "python_files_copy_through_the_writer",
]

TYPES = ["i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f4", "f8"]
CUSTOM_FILL = {
    "i1": -7, "u1": 200, "i2": -999, "u2": 65000, "i4": -99999,
    "u4": 4000000000, "i8": -(2 ** 62), "u8": 2 ** 63 + 5,
    "f4": -999.0, "f8": 1e300,
}


class Skip(Exception):
    """Raise this in a case to skip it."""


CASES = []


def case(func):
    """Register a case function. The name is the function name without 'case_'."""
    CASES.append((func.__name__[len("case_"):], func))
    return func


def rng(seed):
    return np.random.default_rng(seed)


def rand(r, code, shape):
    """Return random data of one type."""
    dt = np.dtype(code)
    if dt.kind in "iu":
        info = np.iinfo(dt)
        return r.integers(info.min, info.max, size=shape, dtype=dt, endpoint=True)
    return (r.normal(size=shape) * 1000).astype(dt)


def chars(items, width):
    """Return a 2-D array of single characters for a list of byte strings."""
    return np.array(items, "S%d" % width).view("S1").reshape(len(items), width)


def new(path):
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.set_auto_maskandscale(False)
    return ds


# ---------------------------------------------------------------- cases


@case
def case_int_types_default_fill(path):
    r = rng(1)
    with new(path) as ds:
        ds.createDimension("x", 7)
        for t in TYPES:
            ds.createVariable("v_" + t, t, ("x",))[:] = rand(r, t, 7)


@case
def case_types_custom_fill_partial(path):
    r = rng(2)
    with new(path) as ds:
        ds.createDimension("x", 9)
        ds.createDimension("y", 4)
        for t in TYPES:
            v = ds.createVariable("v_" + t, t, ("x",), fill_value=np.dtype(t).type(CUSTOM_FILL[t]))
            v[:4] = rand(r, t, 4)
            w = ds.createVariable("w_" + t, t, ("x", "y"), fill_value=np.dtype(t).type(CUSTOM_FILL[t]))
            w[2:5, 1:3] = rand(r, t, (3, 2))


@case
def case_types_default_fill_partial(path):
    r = rng(3)
    with new(path) as ds:
        ds.createDimension("x", 10)
        for t in TYPES:
            v = ds.createVariable("v_" + t, t, ("x",))
            v[3:6] = rand(r, t, 3)
        ds.createVariable("never_written", "f8", ("x",))


@case
def case_float_special_values(path):
    with new(path) as ds:
        ds.createDimension("x", 12)
        v = ds.createVariable("f4", "f4", ("x",), fill_value=np.float32(-1.5))
        v[:] = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1e-45, 3.4e38, -1.5, 1.0, 2.0, 3.0, 4.0], "f4")
        w = ds.createVariable("f8", "f8", ("x",), fill_value=np.float64(np.nan))
        w[:] = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 5e-324, 1.7e308, 1.0, 2.0, 3.0, 4.0, 5.0])
        z = ds.createVariable("f8_nan_fill_unwritten", "f8", ("x",), fill_value=np.float64(np.nan))
        z[:3] = [1.0, 2.0, 3.0]


@case
def case_big_endian(path):
    r = rng(4)
    with new(path) as ds:
        ds.createDimension("x", 6)
        ds.createDimension("y", 5)
        for t in TYPES:
            ds.createVariable("be_" + t, t, ("x", "y"), endian="big")[:] = rand(r, t, (6, 5))
            ds.createVariable("le_" + t, t, ("x", "y"), endian="little")[:] = rand(r, t, (6, 5))
        v = ds.createVariable("be_f4_fill", "f4", ("x",), endian="big", fill_value=np.float32(-5.0))
        v[:2] = [1.5, 2.5]
        c = ds.createVariable("be_zlib", "i4", ("x", "y"), endian="big", zlib=True, complevel=4, shuffle=True)
        c[:] = rand(r, "i4", (6, 5))


@case
def case_scalars(path):
    r = rng(5)
    with new(path) as ds:
        for t in TYPES:
            ds.createVariable("s_" + t, t, ()).assignValue(rand(r, t, ())[()])
            ds.createVariable("sf_" + t, t, (), fill_value=np.dtype(t).type(CUSTOM_FILL[t]))
        ds.createVariable("s_char", "S1", ()).assignValue(np.array(b"q", "S1"))
        ds.createVariable("s_str", str, ())[0] = "scalar text"
        ds.createVariable("s_unwritten", "f8", ())


@case
def case_char_2d(path):
    with new(path) as ds:
        ds.createDimension("n", 4)
        ds.createDimension("len", 8)
        ds.createDimension("k", 2)
        v = ds.createVariable("names", "S1", ("n", "len"))
        v.set_auto_chartostring(False)
        v[:] = chars([b"alpha", b"be", b"", b"12345678"], 8)
        w = ds.createVariable("grid", "S1", ("k", "n", "len"), fill_value=b"#")
        w.set_auto_chartostring(False)
        w[0] = chars([b"a", b"bb", b"ccc", b"dddd"], 8)


@case
def case_strings(path):
    with new(path) as ds:
        ds.createDimension("n", 6)
        ds.createDimension("m", 2)
        v = ds.createVariable("s", str, ("n",))
        v[:] = np.array(["hello", "", "wörld", "日本語のテキスト", "emoji \U0001F600", "a" * 300], dtype=object)
        w = ds.createVariable("s2", str, ("m", "n"))
        w[:] = np.array([["a", "b", "", "d", "é", "f"], ["", "", "", "", "", "x y z"]], dtype=object)
        e = ds.createVariable("empties", str, ("m",))
        e[:] = np.array(["", ""], dtype=object)


@case
def case_unlimited_records(path):
    r = rng(6)
    with new(path) as ds:
        ds.createDimension("t", None)
        ds.createDimension("x", 3)
        a = ds.createVariable("a", "f4", ("t", "x"))
        b = ds.createVariable("b", "i2", ("t",), fill_value=np.int16(-1))
        c = ds.createVariable("c", "f8", ("t", "x", "x"))
        a[:] = rand(r, "f4", (5, 3))
        b[:] = rand(r, "i2", 5)
        c[:] = rand(r, "f8", (5, 3, 3))
        a[5:8] = rand(r, "f4", (3, 3))
        b[5:8] = rand(r, "i2", 3)
        c[5:8] = rand(r, "f8", (3, 3, 3))


@case
def case_two_unlimited(path):
    r = rng(7)
    with new(path) as ds:
        ds.createDimension("t", None)
        ds.createDimension("s", None)
        a = ds.createVariable("a", "i4", ("t", "s"))
        b = ds.createVariable("b", "f4", ("s",))
        c = ds.createVariable("c", "u2", ("t",))
        d = ds.createVariable("d", "f8", ("s", "t"))
        a[:3, :4] = rand(r, "i4", (3, 4))
        b[:4] = rand(r, "f4", 4)
        c[:3] = rand(r, "u2", 3)
        d[:4, :3] = rand(r, "f8", (4, 3))


@case
def case_unlimited_ragged(path):
    r = rng(8)
    with new(path) as ds:
        ds.createDimension("t", None)
        ds.createDimension("x", 2)
        ds.createVariable("long", "f4", ("t",))[:8] = rand(r, "f4", 8)
        ds.createVariable("short", "i4", ("t",), fill_value=np.int32(-9))[:3] = rand(r, "i4", 3)
        ds.createVariable("mid", "i1", ("t", "x"))[:5] = rand(r, "i1", (5, 2))
        ds.createVariable("none", "f8", ("t",))


@case
def case_unlimited_zero_records(path):
    r = rng(9)
    with new(path) as ds:
        ds.createDimension("t", None)
        ds.createDimension("x", 3)
        ds.createVariable("v", "f4", ("t", "x"))
        ds.createVariable("w", "i2", ("t",), zlib=True)
        ds.createVariable("fixed", "i4", ("x",))[:] = rand(r, "i4", 3)
        ds.setncattr("note", "no records")


@case
def case_unlimited_compressed(path):
    r = rng(10)
    with new(path) as ds:
        ds.createDimension("t", None)
        ds.createDimension("x", 20)
        v = ds.createVariable("v", "f4", ("t", "x"), zlib=True, shuffle=True, complevel=5, chunksizes=(3, 7))
        v[:11] = r.normal(size=(11, 20)).astype("f4")
        w = ds.createVariable("w", "i8", ("t",), zlib=True, complevel=1, shuffle=False)
        w[:11] = rand(r, "i8", 11)


@case
def case_coordinate_variables(path):
    r = rng(11)
    with new(path) as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", 5)
        ds.createDimension("lon", 8)
        ds.createDimension("lev", 3)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "days since 2000-01-01"
        t[:6] = np.arange(6) * 0.5
        la = ds.createVariable("lat", "f4", ("lat",))
        la.units = "degrees_north"
        la[:] = np.linspace(-60, 60, 5).astype("f4")
        lo = ds.createVariable("lon", "f8", ("lon",))
        lo[:] = np.linspace(0, 350, 8)
        ds.createVariable("lev", "i2", ("lev",))[:] = [1000, 850, 500]
        d = ds.createVariable("temp", "f4", ("time", "lev", "lat", "lon"), fill_value=np.float32(-999))
        d.coordinates = "time lev lat lon"
        d[:6] = r.normal(size=(6, 3, 5, 8)).astype("f4")


@case
def case_char_coordinate_2d(path):
    with new(path) as ds:
        ds.createDimension("name", 3)
        ds.createDimension("strlen", 6)
        v = ds.createVariable("name", "S1", ("name", "strlen"))
        v.set_auto_chartostring(False)
        v[:] = chars([b"one", b"two", b"three"], 6)
        ds.createVariable("data", "f4", ("name",))[:] = [1, 2, 3]


@case
def case_name_of_dim_not_used(path):
    r = rng(12)
    with new(path) as ds:
        ds.createDimension("x", 4)
        ds.createDimension("y", 3)
        ds.createVariable("x", "f8", ("y",))[:] = r.normal(size=3)
        ds.createVariable("y", "i4", ("x", "y"))[:] = rand(r, "i4", (4, 3))


@case
def case_dimension_without_variable(path):
    with new(path) as ds:
        ds.createDimension("used", 3)
        ds.createDimension("unused", 5)
        ds.createDimension("unused_unlimited", None)
        ds.createDimension("zero", 0)
        ds.createVariable("v", "i4", ("used",))[:] = [1, 2, 3]


@case
def case_dimension_used_twice(path):
    r = rng(13)
    with new(path) as ds:
        ds.createDimension("x", 4)
        ds.createDimension("t", None)
        ds.createVariable("m", "f8", ("x", "x"))[:] = r.normal(size=(4, 4))
        ds.createVariable("k", "i2", ("t", "x", "x"))[:2] = rand(r, "i2", (2, 4, 4))


@case
def case_nested_groups(path):
    r = rng(14)
    with new(path) as ds:
        ds.createDimension("x", 4)
        ds.createDimension("t", None)
        ds.createVariable("root_v", "f4", ("x",))[:] = rand(r, "f4", 4)
        g1 = ds.createGroup("g1")
        g1.createDimension("y", 3)
        g1.createVariable("v1", "f8", ("x", "y"))[:] = rand(r, "f8", (4, 3))
        g1.createVariable("v2", "i4", ("t", "x"))[:2] = rand(r, "i4", (2, 4))
        g2 = g1.createGroup("g2")
        g2.createDimension("z", 2)
        g2.createVariable("v3", "i2", ("x", "y", "z"))[:] = rand(r, "i2", (4, 3, 2))
        g3 = g2.createGroup("g3")
        g3.createDimension("w", 5)
        g3.createVariable("v4", "u1", ("t", "z", "w"))[:3] = rand(r, "u1", (3, 2, 5))
        g3.createVariable("root_v", "f4", ("x",))[:] = rand(r, "f4", 4)
        ds.createGroup("empty_group")
        ds.createGroup("g1").createGroup("sibling")


@case
def case_group_dimension_shadows_root(path):
    r = rng(15)
    with new(path) as ds:
        ds.createDimension("x", 5)
        ds.createVariable("a", "i4", ("x",))[:] = rand(r, "i4", 5)
        c = ds.createGroup("c")
        c.createDimension("x", 3)
        c.createVariable("b", "f8", ("x",))[:] = rand(r, "f8", 3)
        d = c.createGroup("d")
        d.createVariable("e", "f4", ("x",))[:] = rand(r, "f4", 3)
        u = ds.createGroup("u")
        u.createDimension("x", None)
        u.createVariable("x", "i2", ("x",))[:4] = [1, 2, 3, 4]


@case
def case_chunking_explicit(path):
    r = rng(16)
    with new(path) as ds:
        ds.createDimension("t", 10)
        ds.createDimension("x", 7)
        ds.createDimension("n", 12)
        ds.createDimension("u", None)
        ds.createVariable("contig", "f4", ("t", "x"), contiguous=True)[:] = rand(r, "f4", (10, 7))
        ds.createVariable("edge", "f8", ("t", "x"), chunksizes=(3, 4))[:] = rand(r, "f8", (10, 7))
        ds.createVariable("one_d", "i2", ("n",), chunksizes=(5,))[:] = rand(r, "i2", 12)
        ds.createVariable("partial", "i4", ("t", "x"), chunksizes=(4, 7), fill_value=np.int32(-1))[:5, :3] = rand(r, "i4", (5, 3))
        ds.createVariable("rec", "f4", ("u", "x"), chunksizes=(2, 7))[:5] = rand(r, "f4", (5, 7))
        ds.createVariable("auto", "f4", ("t", "x"), chunksizes=None)[:] = rand(r, "f4", (10, 7))
        ds.createVariable("contig_scalar", "f8", (), contiguous=True).assignValue(1.25)


@case
def case_zlib_levels(path):
    r = rng(17)
    data = np.cumsum(r.normal(size=(40, 30)), axis=1).astype("f4")
    idata = r.integers(0, 50, size=(40, 30)).astype("i4")
    with new(path) as ds:
        ds.createDimension("a", 40)
        ds.createDimension("b", 30)
        for level in (1, 9):
            for shuffle in (False, True):
                tag = "l%d_s%d" % (level, int(shuffle))
                ds.createVariable("f_" + tag, "f4", ("a", "b"), zlib=True, complevel=level, shuffle=shuffle)[:] = data
                ds.createVariable("i_" + tag, "i4", ("a", "b"), zlib=True, complevel=level, shuffle=shuffle,
                                  chunksizes=(10, 30))[:] = idata
        ds.createVariable("shuffle_only", "i2", ("a", "b"), shuffle=True, chunksizes=(8, 8))[:] = idata.astype("i2")
        ds.createVariable("partial_zlib", "f8", ("a", "b"), zlib=True, complevel=6,
                          chunksizes=(10, 10), fill_value=np.float64(-1))[:15, :12] = data[:15, :12]


@case
def case_fletcher32(path):
    r = rng(18)
    with new(path) as ds:
        ds.createDimension("a", 20)
        ds.createDimension("b", 9)
        ds.createVariable("plain", "f4", ("a", "b"), fletcher32=True, chunksizes=(5, 9))[:] = rand(r, "f4", (20, 9))
        ds.createVariable("with_zlib", "i4", ("a", "b"), fletcher32=True, zlib=True, shuffle=True,
                          complevel=3, chunksizes=(7, 4))[:] = rand(r, "i4", (20, 9))
        ds.createVariable("f64", "f8", ("a",), fletcher32=True)[:] = rand(r, "f8", 20)


@case
def case_zstd(path):
    if not getattr(netCDF4, "__has_zstandard_support__", False):
        raise Skip("netCDF4 has no Zstandard support")
    probe = os.path.join(os.path.dirname(path), "_zstd_probe.tmp")
    try:
        with netCDF4.Dataset(probe, "w", format="NETCDF4") as p:
            p.createDimension("x", 50)
            p.createVariable("v", "f4", ("x",), compression="zstd", complevel=3)[:] = np.arange(50, dtype="f4")
        with netCDF4.Dataset(probe) as p:
            if not np.array_equal(p["v"][:], np.arange(50, dtype="f4")):
                raise Skip("Zstandard probe reads wrong values")
    except Skip:
        raise
    except Exception as exc:
        raise Skip("Zstandard probe fails: %s" % exc)
    finally:
        if os.path.exists(probe):
            os.remove(probe)
    r = rng(19)
    with new(path) as ds:
        ds.createDimension("x", 200)
        ds.createDimension("y", 30)
        ds.createVariable("z", "f4", ("x", "y"), compression="zstd", complevel=3,
                          chunksizes=(50, 30))[:] = np.cumsum(r.normal(size=(200, 30)), axis=0).astype("f4")
        ds.createVariable("zs", "i4", ("x", "y"), compression="zstd", complevel=1, shuffle=True)[:] = \
            r.integers(0, 20, size=(200, 30)).astype("i4")


@case
def case_large_compressed(path):
    r = rng(20)
    with new(path) as ds:
        ds.createDimension("a", 2000)
        ds.createDimension("b", 1000)
        v = ds.createVariable("big", "f4", ("a", "b"), zlib=True, complevel=4, shuffle=True, chunksizes=(100, 500))
        for i in range(0, 2000, 500):
            v[i:i + 500] = np.cumsum(r.normal(size=(500, 1000)), axis=1).astype("f4")


@case
def case_many_variables(path):
    r = rng(21)
    with new(path) as ds:
        ds.createDimension("x", 5)
        ds.createDimension("t", None)
        for i in range(60):
            t = TYPES[i % len(TYPES)]
            dims = ("x",) if i % 3 else ("t", "x")
            v = ds.createVariable("var_%02d" % i, t, dims)
            v.index = np.int32(i)
            if i % 3:
                v[:] = rand(r, t, 5)
            else:
                v[:2] = rand(r, t, (2, 5))


@case
def case_many_attributes(path):
    r = rng(22)
    with new(path) as ds:
        ds.createDimension("x", 3)
        v = ds.createVariable("v", "f4", ("x",))
        v[:] = [1, 2, 3]
        g = ds.createGroup("grp")
        for target in (ds, v, g):
            for i in range(40):
                name = "attr_%02d" % i
                if i % 4 == 0:
                    target.setncattr(name, "text value number %d" % i)
                elif i % 4 == 1:
                    target.setncattr(name, np.float64(i) / 7)
                elif i % 4 == 2:
                    target.setncattr(name, rand(r, "i4", 1 + i % 5))
                else:
                    target.setncattr(name, np.int16(i))


@case
def case_attribute_types(path):
    r = rng(23)
    with new(path) as ds:
        ds.createDimension("x", 2)
        v = ds.createVariable("v", "f4", ("x",))
        for target in (ds, v):
            for t in TYPES:
                target.setncattr("scalar_" + t, rand(r, t, ())[()])
                target.setncattr("array_" + t, rand(r, t, 4))
                target.setncattr("single_" + t, rand(r, t, 1))
        ds.setncattr("nan_f4", np.float32(np.nan))
        ds.setncattr("inf_f8", np.array([np.inf, -np.inf, -0.0]))


@case
def case_text_attributes(path):
    with new(path) as ds:
        ds.createDimension("x", 2)
        v = ds.createVariable("v", "i4", ("x",))
        for target in (ds, v):
            target.setncattr("plain", "hello world")
            target.setncattr("empty", "")
            target.setncattr("utf8", "héllo wörld, 日本語, \U0001F600")
            target.setncattr("long", "x" * 5000)
            target.setncattr("with_newline", "line one\nline two\ttab")
            target.setncattr("one_char", "a")


@case
def case_string_array_attributes(path):
    with new(path) as ds:
        ds.createDimension("x", 2)
        v = ds.createVariable("v", "i4", ("x",))
        for target in (ds, v):
            target.setncattr_string("one", "single string")
            target.setncattr_string("many", ["alpha", "", "wörld", "日本"])
            target.setncattr_string("empty_one", "")
            target.setncattr_string("pair", ["a", "b"])


@case
def case_group_attributes(path):
    with new(path) as ds:
        ds.setncattr("title", "root")
        ds.createDimension("x", 2)
        g = ds.createGroup("a")
        g.setncattr("kind", "group a")
        g.setncattr("n", np.int32(3))
        g.setncattr("arr", np.array([1.5, 2.5], "f8"))
        g.setncattr_string("strs", ["p", "q"])
        h = g.createGroup("b")
        h.setncattr("deep", "yes")
        h.createVariable("v", "i2", ("x",)).setncattr("u", "m")
        ds.createGroup("c").setncattr("only", np.uint64(2 ** 63 + 1))


@case
def case_odd_names(path):
    r = rng(24)
    with new(path) as ds:
        ds.createDimension("dim with space", 3)
        ds.createDimension("dim.with.dots", 2)
        ds.createDimension("dim-with-dash", 4)
        ds.createDimension("dimé_ñ", 2)
        ds.createDimension("日本", 2)
        ds.createVariable("my var", "f4", ("dim with space",))[:] = rand(r, "f4", 3)
        ds.createVariable("v.dot", "i4", ("dim.with.dots",))[:] = rand(r, "i4", 2)
        ds.createVariable("v-dash", "i2", ("dim-with-dash", "dim.with.dots"))[:] = rand(r, "i2", (4, 2))
        ds.createVariable("größe", "f8", ("dimé_ñ",))[:] = rand(r, "f8", 2)
        ds.createVariable("温度", "f4", ("日本",))[:] = rand(r, "f4", 2)
        ds.createVariable("dim-with-dash", "f8", ("dim-with-dash",))[:] = rand(r, "f8", 4)
        ds.setncattr("global attr", "a")
        ds.setncattr("a.b", "b")
        ds.setncattr("c-d", "c")
        ds.setncattr("é", "e")
        ds.createVariable("v2", "i4", ("dimé_ñ",)).setncattr("my attr", np.int8(1))
        g = ds.createGroup("grp with space")
        g.setncattr("é-attr", "g")
        g.createGroup("gruppe-é").createVariable("v", "u1", ("dimé_ñ",))[:] = [1, 2]


@case
def case_scale_offset_packed(path):
    r = rng(25)
    raw = r.integers(-3000, 3000, size=(6, 5)).astype("i2")
    raw[0, 0] = -32768
    with new(path) as ds:
        ds.createDimension("x", 6)
        ds.createDimension("y", 5)
        v = ds.createVariable("packed", "i2", ("x", "y"), fill_value=np.int16(-32768))
        v.scale_factor = np.float64(0.01)
        v.add_offset = np.float64(20.0)
        v.units = "K"
        v[:] = raw
        w = ds.createVariable("packed_byte", "u1", ("x",), zlib=True)
        w.scale_factor = np.float32(0.5)
        w.add_offset = np.float32(-10)
        w[:] = rand(r, "u1", 6)


@case
def case_empty_with_global_attributes(path):
    with new(path) as ds:
        ds.setncattr("title", "empty file")
        ds.setncattr("version", np.int32(3))
        ds.setncattr("values", np.array([1.0, 2.0]))


@case
def case_only_dimensions(path):
    with new(path) as ds:
        ds.createDimension("a", 3)
        ds.createDimension("b", None)
        ds.createDimension("c", 1)
        ds.createDimension("d", 0)


@case
def case_empty_file(path):
    with new(path):
        pass


# ------------------------------------------------------------- generate


def generate(cases_dir):
    """Write all case files. Return the names, the skips and the problems."""
    names, skipped, problems = [], [], []
    for name, func in CASES:
        path = os.path.join(cases_dir, name + ".nc")
        try:
            func(path)
            names.append(name)
        except Skip as exc:
            skipped.append((name, str(exc)))
            if os.path.exists(path):
                os.remove(path)
        except Exception as exc:
            problems.append("%s: generator fails: %s" % (name, exc))
            if os.path.exists(path):
                os.remove(path)
    return names, skipped, problems


# -------------------------------------------------------------- compare


class Stats:
    def __init__(self):
        self.values = 0
        self.variables = 0
        self.attributes = 0
        self.groups = 0


def raw_bytes(a):
    """Return a byte string that holds the exact bits of an array in native order."""
    a = np.ascontiguousarray(np.asarray(a))
    if a.dtype.kind in "biufc":
        a = a.astype(a.dtype.newbyteorder("="), copy=False)
    return a.tobytes()


def same_array(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.kind == "O":
        return a.tolist() == b.tolist()
    return raw_bytes(a) == raw_bytes(b)


def same_attr(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return type(a) is type(b) and a == b
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and list(a) == list(b)
    return same_array(a, b)


def describe(x):
    text = repr(x)
    return text if len(text) <= 80 else text[:77] + "..."


def compare_attrs(obj_a, obj_b, where, problems, stats):
    names_a = list(obj_a.ncattrs())
    names_b = list(obj_b.ncattrs())
    if names_a != names_b:
        problems.append("%s: attribute names differ: %s vs %s" % (where, names_a, names_b))
        return
    for n in names_a:
        stats.attributes += 1
        va, vb = obj_a.getncattr(n), obj_b.getncattr(n)
        if not same_attr(va, vb):
            problems.append("%s: attribute %r differs: %s vs %s" % (where, n, describe(va), describe(vb)))


def read_values(var):
    var.set_auto_maskandscale(False)
    var.set_auto_chartostring(False)
    data = var[...]
    if var.dtype is str:
        return np.asarray(data, dtype=object).ravel()
    return np.ma.getdata(np.asarray(data))


def compare_var(va, vb, where, problems, stats):
    stats.variables += 1
    if va.dtype != vb.dtype:
        problems.append("%s: dtype differs: %s vs %s" % (where, va.dtype, vb.dtype))
        return
    if va.endian() != vb.endian():
        problems.append("%s: byte order differs: %s vs %s" % (where, va.endian(), vb.endian()))
    if va.dimensions != vb.dimensions:
        problems.append("%s: dimensions differ: %s vs %s" % (where, va.dimensions, vb.dimensions))
        return
    if va.shape != vb.shape:
        problems.append("%s: shape differs: %s vs %s" % (where, va.shape, vb.shape))
        return
    if va.chunking() != vb.chunking():
        problems.append("%s: chunking differs: %s vs %s" % (where, va.chunking(), vb.chunking()))
    if va.filters() != vb.filters():
        problems.append("%s: filters differ: %s vs %s" % (where, va.filters(), vb.filters()))
    compare_attrs(va, vb, where, problems, stats)
    try:
        da, db = read_values(va), read_values(vb)
    except Exception as exc:
        problems.append("%s: read fails: %s" % (where, exc))
        return
    stats.values += da.size
    if not same_array(da, db):
        problems.append("%s: values differ" % where)


def compare_group(ga, gb, where, problems, stats):
    stats.groups += 1
    ga.set_auto_maskandscale(False)
    gb.set_auto_maskandscale(False)
    dims_a = [(n, len(d), d.isunlimited()) for n, d in ga.dimensions.items()]
    dims_b = [(n, len(d), d.isunlimited()) for n, d in gb.dimensions.items()]
    if dims_a != dims_b:
        problems.append("%s: dimensions differ: %s vs %s" % (where, dims_a, dims_b))
    compare_attrs(ga, gb, where, problems, stats)
    names_a, names_b = list(ga.variables), list(gb.variables)
    if names_a != names_b:
        problems.append("%s: variable names differ: %s vs %s" % (where, names_a, names_b))
    for n in names_a:
        if n in gb.variables:
            compare_var(ga.variables[n], gb.variables[n], "%s var %s" % (where, n), problems, stats)
    groups_a, groups_b = list(ga.groups), list(gb.groups)
    if groups_a != groups_b:
        problems.append("%s: group names differ: %s vs %s" % (where, groups_a, groups_b))
    for n in groups_a:
        if n in gb.groups:
            compare_group(ga.groups[n], gb.groups[n], where.rstrip("/") + "/" + n, problems, stats)


def group_paths(ds, prefix=""):
    yield prefix
    for n, g in ds.groups.items():
        yield from group_paths(g, prefix + "/" + n if prefix else n)


def xarray_opens(path):
    """Return an error text if xarray cannot open every group of the file, else None."""
    import xarray
    warnings.simplefilter("ignore")
    try:
        with netCDF4.Dataset(path) as ds:
            paths = list(group_paths(ds))
        for p in paths:
            with xarray.open_dataset(path, decode_times=False, group=p or None) as x:
                x.load()
    except Exception as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    return None


def compare_case(name, cases_dir, copies_dir, stats):
    problems = []
    orig = os.path.join(cases_dir, name + ".nc")
    copy = os.path.join(copies_dir, name + ".nc")
    if not os.path.exists(copy):
        return ["%s: copy is missing" % name]
    try:
        with netCDF4.Dataset(orig) as a, netCDF4.Dataset(copy) as b:
            found = []
            compare_group(a, b, "/", found, stats)
        problems += ["%s: %s" % (name, m) for m in found]
    except Exception as exc:
        problems.append("%s: compare fails: %s" % (name, exc))
        traceback.print_exc()
    err = xarray_opens(copy)
    if err is not None and xarray_opens(orig) is None:
        problems.append("%s: xarray cannot open the copy: %s" % (name, err))
    return problems


def compare_all(work, names):
    cases_dir = os.path.join(work, "cases")
    copies_dir = os.path.join(work, "copies")
    stats = Stats()
    problems = []
    for name in names:
        problems += compare_case(name, cases_dir, copies_dir, stats)
    return problems, stats


# ----------------------------------------------------------------- main


def run_rust(work):
    env = dict(os.environ, OXCDF_COPY_DIR=work)
    res = subprocess.run(RUST_CMD, cwd=str(REPO), env=env, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        print("cargo fails with exit code %d" % res.returncode)
        print("--- stdout ---")
        print(res.stdout)
        print("--- stderr ---")
        print(res.stderr)
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", help="work directory (default: a temporary directory)")
    ap.add_argument("--skip-rust", action="store_true", help="skip the Rust step and the compare step")
    args = ap.parse_args()

    temporary = args.dir is None
    work = os.path.abspath(args.dir) if args.dir else tempfile.mkdtemp(prefix="oxcdf_copy_")
    cases_dir = os.path.join(work, "cases")
    copies_dir = os.path.join(work, "copies")
    for d in (cases_dir, copies_dir):
        shutil.rmtree(d, ignore_errors=True)
    os.makedirs(cases_dir)
    print("work directory: %s" % work)

    names, skipped, problems = generate(cases_dir)
    print("cases written: %d" % len(names))
    stats = Stats()
    if not args.skip_rust:
        if not run_rust(work):
            return 1
        more, stats = compare_all(work, names)
        problems += more

    print("cases: %d" % len(names))
    print("skipped cases: %d" % len(skipped))
    for n, why in skipped:
        print("  skipped %s: %s" % (n, why))
    print("variables compared: %d" % stats.variables)
    print("values compared: %d" % stats.values)
    print("%d problems" % len(problems))
    for p in problems[:60]:
        print(p)
    if len(problems) > 60:
        print("... %d more problems not shown" % (len(problems) - 60))
    if not problems and temporary:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
