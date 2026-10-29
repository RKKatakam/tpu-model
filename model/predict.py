"""Glue: record directory -> parsed program -> replay -> predicted device time."""

import json
import os
import re
import statistics
from dataclasses import asdict, dataclass

from capture import xplane
from model.core import CoreParams, Scoreboard
from model.dma import DirFifoHbm, FifoHbm, SharedHbm, TableHbm
from model.machine import Machine
from parser import bundles, dumps

LINE = "XLA Ops"


@dataclass
class Params:
    clock_ghz: float
    overhead_ns: float
    dma_table_ns: list  # [[bytes, single-stream per-copy ns], ...] from bench dma_seq

    def dump(self, path, provenance):
        with open(path, "w") as f:
            json.dump({"params": asdict(self), "provenance": provenance}, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f)["params"])


@dataclass
class Record:
    dir: str
    raw: dict

    @property
    def hlo_name(self):
        return xplane.custom_call_name(self.raw["timing"]["profile_summary"])

    @property
    def tag(self):
        """Dump-file stem: the HLO name without its '.N' suffix."""
        return re.sub(r"\.\d+$", "", self.hlo_name)

    @property
    def dump(self):
        return os.path.join(self.dir, "dump")

    def samples_ns(self):
        by_line = self.raw["timing"]["by_line"]
        if LINE in by_line:
            return by_line[LINE]["samples_ns"]
        # The capture-time tag missed the kernel (corpus1); re-extract from the saved xplane.
        cache = os.path.join(self.dir, "samples_offline.json")
        if not os.path.exists(cache):
            profile = xplane.load(xplane.find_xplane(os.path.join(self.dir, "profile")))
            samples = xplane.kernel_durations(profile, f"%{self.hlo_name} = ")[LINE]
            with open(cache, "w") as f:
                json.dump(samples, f)
        with open(cache) as f:
            return json.load(f)

    def measured_ns(self):
        return statistics.median(self.samples_ns())

    def program(self):
        return bundles.parse_file(dumps.final_bundles_path(self.dump, self.tag))

    def xla_estimated_cycles(self):
        return dumps.static_estimated_cycles(self.dump, self.tag)


def load_records(batch_dir):
    out = []
    for session in sorted(os.listdir(batch_dir)):
        sdir = os.path.join(batch_dir, session)
        if not os.path.isdir(sdir):
            continue
        for rid in sorted(os.listdir(sdir)):
            path = os.path.join(sdir, rid, "record.json")
            if os.path.exists(path):
                with open(path) as f:
                    out.append(Record(os.path.dirname(path), json.load(f)))
    return out


def load_core(path="model/params/core.json"):
    with open(path) as f:
        c = json.load(f)["core"]
    return CoreParams(latency=c["latency"], units=c["units"], issue_lead=c.get("issue_lead", 0.0))


def load_dma(path="model/params/dma.json"):
    with open(path) as f:
        d = json.load(f)
    return d["dma"] | {"engine": d["provenance"].get("engine", "SharedHbm")}


def dma_model(dma, clock_ghz):
    cyc = lambda tbl: [(b, ns * clock_ghz) for b, ns in tbl]  # noqa: E731
    engine = dma.get("engine", "SharedHbm")
    if engine == "DirFifoHbm":
        return DirFifoHbm(
            cyc(dma["read_table_ns"]), cyc(dma["write_table_ns"]),
            dma["bw_read_GBps"] / clock_ghz, dma["bw_write_GBps"] / clock_ghz,
            dma["turnaround_ns_per_byte"] * clock_ghz, dma["issue_ns"] * clock_ghz,
            dma.get("mix_ns", 0.0) * clock_ghz, dma.get("read_lat_adj_ns", 0.0) * clock_ghz,
            dma.get("write_lat_adj_ns", 0.0) * clock_ghz, dma.get("single_queue", False),
            dma["bw_post_GBps"] / clock_ghz if dma.get("bw_post_GBps") else None,
            dma.get("wake_ns", 0.0) * clock_ghz,
        )
    return {"SharedHbm": SharedHbm, "FifoHbm": FifoHbm}[engine](
        cyc(dma["read_table_ns"]), cyc(dma["write_table_ns"]),
        dma["bw_read_GBps"] / clock_ghz, dma["bw_write_GBps"] / clock_ghz, dma["bw_mixed_GBps"] / clock_ghz,
        dma["issue_ns"] * clock_ghz, dma["mix_ns"] * clock_ghz,
    )


def simulate(program, params, core=None, dma=None, operands=None):
    """Replay with the given core and DMA parameters (None = phase-1 minimal model).
    `operands`: integer kernel operands from model.operands.integer_operands."""
    if dma is None:
        engine = TableHbm([(b, ns * params.clock_ghz) for b, ns in params.dma_table_ns])
    else:
        engine = dma_model(dma, params.clock_ghz)
    sb = Scoreboard(core) if core is not None else None
    return Machine(program, core=sb, operands=operands).run(engine)


def predict_ns(trace, params):
    return params.overhead_ns + trace.cycles / params.clock_ghz
