"""Concrete integer operands of a record's Pallas kernel, as 32-bit memory words.

Kernels like splash, paged and gmm steer control flow with integer tables they copy into SMEM
(mask-block tables, page tables, group metadata). Those tables are computed outside the kernel
from the recorded integer inputs (or are constants), so we rebuild them on CPU: trace the corpus
function, evaluate its jaxpr with the recorded inputs up to the pallas_call, and take that call's
operands. Each is checked against the bundle dump's inlined_call_operand shape and laid out in
the HLO layout the kernel saw.
"""

import glob
import json
import os
import re

import numpy as np

SUBS = ("jaxpr", "call_jaxpr", "fun_jaxpr")


def _inner(eqn):
    for k in SUBS:
        if k in eqn.params:
            v = eqn.params[k]
            return v.jaxpr if hasattr(v, "jaxpr") else v, getattr(v, "consts", ())
    return None, ()


def _has_pallas(jaxpr):
    for e in jaxpr.eqns:
        if e.primitive.name == "pallas_call":
            return True
        inner, _ = _inner(e)
        if inner is not None and _has_pallas(inner):
            return True
    return False


def _eval_to_pallas(jaxpr, consts, args):
    from jax.extend.core import Literal

    env = {}

    def read(v):
        return v.val if isinstance(v, Literal) else env[v]

    for v, x in zip(jaxpr.constvars, consts):
        env[v] = x
    for v, x in zip(jaxpr.invars, args):
        env[v] = x
    for e in jaxpr.eqns:
        vals = [read(v) for v in e.invars]
        if e.primitive.name == "pallas_call":
            return vals
        inner, inner_consts = _inner(e)
        if inner is not None and _has_pallas(inner):
            return _eval_to_pallas(inner, inner_consts, vals)
        out = e.primitive.bind(*vals, **e.params)
        for v, x in zip(e.outvars, out if e.primitive.multiple_results else [out]):
            env[v] = x
    raise ValueError("no pallas_call in jaxpr")


def pallas_operands(record):
    """Values of the pallas_call operands (integer ones concrete, float ones zeros)."""
    import jax
    import jax.numpy as jnp

    from corpus import kernels

    with jax.default_device(jax.devices("cpu")[0]):
        fn, specs, _, _ = kernels.build(record.raw["kernel"], record.raw["params"])
        ints = record.raw.get("integer_inputs") or {}
        args = [jnp.asarray(np.array(ints[str(i)], dtype=s.dtype)) if str(i) in ints else jnp.zeros(s.shape, s.dtype)
                for i, s in enumerate(specs)]
        closed = jax.make_jaxpr(fn)(*specs)
        return [np.asarray(v) for v in _eval_to_pallas(closed.jaxpr, closed.consts, args)]


def hlo_operand_layouts(record):
    """Layout strings ('{0:T(128)S(1)}') of the tpu_custom_call operands, in operand order."""
    for f in glob.glob(os.path.join(record.dump, "xla", "*after_optimizations.txt")):
        txt = open(f).read()
        m = re.search(r"custom-call\(([^)]*)\), custom_call_target=\"tpu_custom_call\"", txt)
        if not m:
            continue
        out = []
        for name in re.findall(r"%[\w.\-]+", m.group(1)):
            d = re.search(re.escape(name) + r" = \(?\w+\[[^\]]*\](\{[^}]*\})?", txt)
            out.append(d.group(1) or "" if d else "")
        return out
    raise FileNotFoundError(f"no tpu_custom_call in HLO dumps of {record.dir}")


def words(array, layout):
    """32-bit memory image of `array` under an XLA TPU layout (row-major tiles; sub-32-bit types
    packed (k,1): k consecutive rows of a tile share a word, row r%k in byte r%k)."""
    a = np.asarray(array)
    if a.dtype.itemsize == 4:
        a = a.view(np.uint32)
    tiles = re.findall(r"\(([\d,]+)\)", layout)
    if a.ndim <= 1 or not tiles:
        flat = a.reshape(-1)
        if a.dtype.itemsize == 4:
            return [int(x) for x in flat]
        raise NotImplementedError(f"untiled sub-word array {a.dtype} {layout}")
    tr, tc = (int(x) for x in tiles[0].split(","))
    pack = int(tiles[1].split(",")[0]) if len(tiles) > 1 and a.dtype.itemsize < 4 else 1
    lead = a.reshape(-1, *a.shape[-2:])
    rows, cols = a.shape[-2:]
    pr, pc = -(-rows // tr) * tr, -(-cols // tc) * tc
    out = []
    for m in lead:
        pad = np.zeros((pr, pc), dtype=np.int64)
        pad[:rows, :cols] = m.astype(np.int64) & ((1 << (8 * a.dtype.itemsize)) - 1)
        for ti in range(pr // tr):
            for tj in range(pc // tc):
                t = pad[ti * tr:(ti + 1) * tr, tj * tc:(tj + 1) * tc]
                for r0 in range(0, tr, pack):
                    for c in range(tc):
                        w = 0
                        for k in range(pack):
                            w |= int(t[r0 + k, c]) << (8 * a.dtype.itemsize * k)
                        out.append(w)
    return out


def integer_operands(record, program):
    """{operand index: (memory space, value or word list)} for the kernel's integer operands,
    cached in the record directory."""
    cache = os.path.join(record.dir, "operands_offline.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return {int(k): tuple(v) for k, v in json.load(f).items()}
    decl = {}
    for b in program.bundles:
        for op in b.ops:
            if op.mnemonic.startswith("inlined_call_operand"):
                m = re.search(r"shape: (\w+)\[([^\]]*)\], index: (\d+), kind: (\w+)", op.text)
                space = op.mnemonic.split(".")[1] if "." in op.mnemonic else ""
                decl[int(m.group(3))] = (space, m.group(1), m.group(2), m.group(4))
    vals = pallas_operands(record)
    layouts = hlo_operand_layouts(record)
    out = {}
    for i, v in enumerate(vals):
        space, dtype, dims, kind = decl[i]
        shape = tuple(int(x) for x in dims.split(",")) if dims else ()
        if np.asarray(v).shape != shape:
            raise ValueError(f"operand {i}: pallas {np.asarray(v).shape} vs dump {shape}")
        if not dtype.startswith(("s", "u")) or dtype == "s4":
            continue
        if space in ("hbm", "vmem", "smem"):
            out[i] = (space, words(v, layouts[i]))
        else:  # passed by value in a scalar register
            out[i] = ("scalar", int(np.asarray(v).reshape(-1)[0]) & 0xFFFFFFFF)
    with open(cache, "w") as f:
        json.dump(out, f)
    return out
