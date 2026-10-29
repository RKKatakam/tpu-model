"""Stub of tpu_inference.utils: the three pure helpers the kernels use, copied verbatim (the real
module imports vllm, torch and torchax)."""

import jax
from jax._src import dtypes


def align_to(unpadded_dim, pad_multiple):
    return (unpadded_dim + pad_multiple - 1) // pad_multiple * pad_multiple


def get_device_name(num_devices: int | None = None):
    kind = jax.devices()[0].device_kind
    if 'TPU' not in kind:
        raise RuntimeError('Expected TPU devices')
    suffix = ''
    if kind.endswith(' lite'):
        kind = kind[:-len(' lite')]
        suffix = 'e'
    elif kind.endswith('e'):
        kind = kind[:-1]
        suffix = 'e'
    elif kind.endswith('p'):
        kind = kind[:-1]
        suffix = 'p'
    elif kind.endswith('i'):
        kind = kind[:-1]
        suffix = 'i'

    if kind.startswith('TPU7'):
        kind = 'TPU v7'
    elif kind.startswith('TPU8'):
        kind = 'TPU v8'
    assert kind[:-1] == 'TPU v', kind
    kind += suffix
    if num_devices is not None:
        kind += f'-{num_devices}'
    return kind


def get_dtype_packing(dtype):
    bits = dtypes.itemsize_bits(dtype)
    return 32 // bits
