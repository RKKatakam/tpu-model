"""Glue: record directory -> parsed program -> replay -> predicted device time."""

import json
import os
import statistics
from dataclasses import asdict, dataclass

from model.core import CoreParams, Scoreboard
from model.dma import FifoHbm, SharedHbm, TableHbm
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
    def tag(self):
        return self.raw["config"]["kernel_tag"]

    @property
    def dump(self):
        return os.path.join(self.dir, "dump")

    def measured_ns(self):
        return statistics.median(self.raw["timing"]["by_line"][LINE]["samples_ns"])

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
    return CoreParams(latency=c["latency"], units=c["units"])


def load_dma(path="model/params/dma.json"):
    with open(path) as f:
        d = json.load(f)
    return d["dma"] | {"engine": d["provenance"].get("engine", "SharedHbm")}


def dma_model(dma, clock_ghz):
    cyc = lambda tbl: [(b, ns * clock_ghz) for b, ns in tbl]  # noqa: E731
    engine = {"SharedHbm": SharedHbm, "FifoHbm": FifoHbm}[dma.get("engine", "SharedHbm")]
    return engine(
        cyc(dma["read_table_ns"]), cyc(dma["write_table_ns"]),
        dma["bw_read_GBps"] / clock_ghz, dma["bw_write_GBps"] / clock_ghz, dma["bw_mixed_GBps"] / clock_ghz,
        dma["issue_ns"] * clock_ghz, dma["mix_ns"] * clock_ghz,
    )


def simulate(program, params, core=None, dma=None):
    """Replay with the given core and DMA parameters (None = phase-1 minimal model)."""
    if dma is None:
        engine = TableHbm([(b, ns * params.clock_ghz) for b, ns in params.dma_table_ns])
    else:
        engine = dma_model(dma, params.clock_ghz)
    sb = Scoreboard(core) if core is not None else None
    return Machine(program, core=sb).run(engine)


def predict_ns(trace, params):
    return params.overhead_ns + trace.cycles / params.clock_ghz
