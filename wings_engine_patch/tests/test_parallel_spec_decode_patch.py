import inspect
import os
import sys
import types

import pytest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wings_engine_patch.patch_vllm_ascend_container.v0_17_0 import (  # noqa: E402
    parallel_spec_decode_patch,
)


@pytest.mark.parametrize("error_cls", [OSError, TypeError])
def test_patch_eagle_proposer_module_skips_when_source_is_unavailable(
    monkeypatch,
    error_cls,
):
    class FakeSpecDecodeBaseProposer:
        def _run_merged_draft(self):
            return "ok"

    fake_module = types.SimpleNamespace(
        SpecDecodeBaseProposer=FakeSpecDecodeBaseProposer,
    )
    original_method = FakeSpecDecodeBaseProposer._run_merged_draft

    def raise_getsource(_):
        raise error_cls("source unavailable")

    monkeypatch.setattr(inspect, "getsource", raise_getsource)

    parallel_spec_decode_patch._patch_eagle_proposer_module(fake_module)

    assert fake_module.SpecDecodeBaseProposer._run_merged_draft is original_method
