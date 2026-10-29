"""Replay real dumps (data/ is local-only; tests skip without it)."""

import collections
import glob
import os

import pytest

from model.dma import TableHbm
from model.machine import Machine
from parser import bundles, dumps

ROOT = "data/raw/phase0_noise/v5e-a"

# kernel -> (grid steps, {(dma kind, granules): count}) from the kernel definitions in corpus/kernels.py
EXPECTED = {
    "add_one": {("dma.hbm_to_vmem", 128): 1, ("dma.vmem_to_hbm", 128): 1},
    "add": {("dma.hbm_to_vmem", 65536): 64, ("dma.vmem_to_hbm", 65536): 32},
    "col_sum": {("dma.hbm_to_vmem", 131072): 32, ("dma.vmem_to_hbm", 512): 1},
    "matmul": {("dma.hbm_to_vmem", 16384): 128, ("dma.vmem_to_hbm", 16384): 16},
}


def program(kernel):
    d = os.path.join(ROOT, f"{kernel}__p0", "dump")
    if not glob.glob(d):
        pytest.skip("capture data not present")
    return bundles.parse_file(dumps.final_bundles_path(d, f"k_{kernel}"))


@pytest.mark.parametrize("kernel", sorted(EXPECTED))
def test_replay_issues_one_dma_per_block_change(kernel):
    trace = Machine(program(kernel)).run(TableHbm([(512, 450), (4 << 20, 9000)]))
    assert collections.Counter((d.kind, d.granules) for d in trace.dmas) == EXPECTED[kernel]
    assert trace.halts == []


def test_branch_targets_resolve_to_region_starts():
    p = program("col_sum")
    assert p.target(24) == 0x1A6
    assert p.target(64) == 0x4
