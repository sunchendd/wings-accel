import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wings_engine_patch.registry_v1 import _registered_patches


def test_ascend_registry_contract():
  ascend_specs = _registered_patches["vllm_ascend"]["0.17.0"]

  assert ascend_specs["is_default"] is True

  ascend_features = ascend_specs["builder"]()["features"]

  assert set(ascend_features) == {"parallel_spec_decode", "ears"}

  from wings_engine_patch.patch_vllm_ascend_container.v0_17_0.parallel_spec_decode_patch import (
    patch_vllm_ascend_parallel_spec_decode,
  )
  from wings_engine_patch.patch_vllm_container.v0_17_0.ears_patch import patch_vllm_ears

  assert ascend_features["parallel_spec_decode"] == [
    patch_vllm_ascend_parallel_spec_decode,
  ]
  assert ascend_features["ears"] == [patch_vllm_ears]

  vllm_specs = _registered_patches["vllm"]["0.17.0"]
  assert vllm_specs["is_default"] is True
  assert vllm_specs["builder"]()["features"]["ears"] == [patch_vllm_ears]
