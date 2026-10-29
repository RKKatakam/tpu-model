"""Report what this libtpu/jax build accepts: dump/trace flag names and the topology API."""

import importlib.metadata
import inspect
import os
import platform
import re
import socket

print("hostname", socket.gethostname())
print("python", platform.python_version())
for p in ("jax", "jaxlib", "libtpu", "xprof"):
    print(p, importlib.metadata.version(p))

import libtpu  # noqa: E402

so = [os.path.join(r, f) for r, _, fs in os.walk(os.path.dirname(libtpu.__file__)) for f in fs if f.endswith(".so")]
print("libtpu so:", so)
pattern = re.compile(rb"(?:xla_jf|xla_mosaic|xla_llo|xla_xprof|xla_tpu_dump|xla_enable_dump)[a-z0-9_]*")
for path in so:
    with open(path, "rb") as f:
        names = sorted({m.decode() for m in pattern.findall(f.read())})
    print(f"flag-like strings in {os.path.basename(path)}: {len(names)}")
    for n in names:
        print("  ", n)

import jax  # noqa: E402
from jax.experimental import topologies  # noqa: E402

print("get_topology_desc", inspect.signature(topologies.get_topology_desc))
print(inspect.getsource(topologies.get_topology_desc))
try:
    print("devices", jax.devices())
except Exception as e:
    print("devices error", repr(e))
