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


def test_patch_eagle_proposer_module_does_not_mutate_shared_model_config(
    monkeypatch,
):
    class TrackingModelConfig:
        def __init__(self, max_model_len):
            object.__setattr__(self, "max_model_len", max_model_len)
            object.__setattr__(self, "mutation_attempts", [])
            object.__setattr__(self, "is_original", True)

        def __copy__(self):
            clone = type(self).__new__(type(self))
            object.__setattr__(clone, "max_model_len", self.max_model_len)
            object.__setattr__(clone, "mutation_attempts", [])
            object.__setattr__(clone, "is_original", False)
            return clone

        def __setattr__(self, name, value):
            if name == "max_model_len" and self.is_original:
                self.mutation_attempts.append(value)
            object.__setattr__(self, name, value)

    class FakeSpecDecodeBaseProposer:
        def _run_merged_draft(self):
            return (
                self.vllm_config,
                self.vllm_config.model_config.max_model_len,
                self.vllm_config.model_config,
            )

    fake_module = types.SimpleNamespace(
        SpecDecodeBaseProposer=FakeSpecDecodeBaseProposer,
    )

    monkeypatch.setattr(
        inspect,
        "getsource",
        lambda _: """
def _run_merged_draft(self):
    return (
        self.vllm_config,
        self.vllm_config.model_config.max_model_len,
        self.vllm_config.model_config,
    )
""",
    )

    parallel_spec_decode_patch._patch_eagle_proposer_module(fake_module)

    proposer = FakeSpecDecodeBaseProposer()
    original_vllm_config = types.SimpleNamespace(model_config=None)
    original_model_config = TrackingModelConfig(max_model_len=8192)
    original_vllm_config.model_config = original_model_config
    proposer.vllm_config = original_vllm_config
    proposer.max_model_len = 4096

    (
        observed_vllm_config,
        observed_max_model_len,
        observed_model_config,
    ) = proposer._run_merged_draft()

    assert observed_max_model_len == 4096
    assert observed_vllm_config is original_vllm_config
    assert observed_model_config is original_model_config
    assert proposer.vllm_config is original_vllm_config
    assert proposer.vllm_config.model_config is original_model_config
    assert original_model_config.max_model_len == 8192
    assert original_model_config.mutation_attempts == []


def test_patch_eagle_proposer_module_preserves_live_globals(
    monkeypatch,
):
    class FakeSpecDecodeBaseProposer:
        def _run_merged_draft(self):
            return self.vllm_config.model_config.max_model_len + OFFSET

    fake_module = types.SimpleNamespace(
        SpecDecodeBaseProposer=FakeSpecDecodeBaseProposer,
    )
    method_globals = FakeSpecDecodeBaseProposer._run_merged_draft.__globals__
    monkeypatch.setitem(method_globals, "OFFSET", 1)
    monkeypatch.setattr(
        inspect,
        "getsource",
        lambda _: """
def _run_merged_draft(self):
    return self.vllm_config.model_config.max_model_len + OFFSET
""",
    )

    parallel_spec_decode_patch._patch_eagle_proposer_module(fake_module)

    proposer = FakeSpecDecodeBaseProposer()
    proposer.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(max_model_len=8192),
    )
    proposer.max_model_len = 4096

    method_globals["OFFSET"] = 3

    assert proposer._run_merged_draft() == 4099
