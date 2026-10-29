"""Kernels from other repositories, used only as an external test set (never for calibration).

load_tpu_inference() makes `tpu_inference.kernels.*` importable from the vendored copy without the
package __init__, which imports vllm."""

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def load_tpu_inference():
    root = os.path.join(HERE, "tpu_inference")
    if "tpu_inference" not in sys.modules:
        pkg = types.ModuleType("tpu_inference")
        pkg.__path__ = [root]
        sys.modules["tpu_inference"] = pkg
        kern = types.ModuleType("tpu_inference.kernels")
        kern.__path__ = [os.path.join(root, "kernels")]
        sys.modules["tpu_inference.kernels"] = kern
