"""
Fix: AscendDraftModelProposer position out-of-bounds crash when draft model's
max_position_embeddings is smaller than the target model's max_model_len.

Scenario: Qwen3-8B (target, large context) + Qwen3-0.6B (draft, 40960 max positions).
A request with context > 40960 triggers a fatal Ascend CANN gather_v3 assertion:
    Index 40960 out of range[0 40960)

Root cause (two bugs in vllm_ascend v0.17.0rc1):
1. In SpecDecodeBaseProposer.__init__ (via EagleProposer base), max_model_len is set
   from the TARGET model's config:
       self.max_model_len = vllm_config.model_config.max_model_len
   AscendDraftModelProposer passes the target vllm_config, so self.max_model_len
   equals the target model's limit (e.g. 128K), not the draft's (e.g. 40960).

2. SpecDecodeBaseProposer._run_merged_draft hardcodes self.vllm_config.model_config.max_model_len
   for OOB checks (lines 895/902 in eagle_proposer.py) instead of self.max_model_len.

Fix strategy:
1. Patch AscendDraftModelProposer.__init__ to override self.max_model_len with the
   draft model's max_model_len after super().__init__() completes.
2. Patch SpecDecodeBaseProposer._run_merged_draft to replace hardcoded
   self.vllm_config.model_config.max_model_len with self.max_model_len.
"""

import sys

import wrapt


def _register_or_apply_post_import_hook(module_name: str, hook) -> None:
    if module_name in sys.modules:
        hook(sys.modules[module_name])
    else:
        wrapt.register_post_import_hook(hook, module_name)


def _patch_draft_proposer_module(module) -> None:
    cls = getattr(module, "AscendDraftModelProposer", None)
    if cls is None:
        return

    original_init = cls.__init__
    if getattr(original_init, "_wings_parallel_spec_patched", False):
        return

    def patched_init(self, vllm_config, device, runner=None):
        original_init(self, vllm_config, device, runner=runner)
        # Override max_model_len to use draft model's position embedding limit
        # so that OOB checks in _propose use the correct (smaller) limit.
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        if spec_cfg is not None:
            draft_cfg = getattr(spec_cfg, "draft_model_config", None)
            if draft_cfg is not None:
                draft_max = getattr(draft_cfg, "max_model_len", None)
                if draft_max is not None and draft_max < self.max_model_len:
                    self.max_model_len = draft_max

    patched_init._wings_parallel_spec_patched = True  # pylint: disable=protected-access
    cls.__init__ = patched_init


def _patch_eagle_proposer_module(module) -> None:
    cls = getattr(module, "SpecDecodeBaseProposer", None)
    if cls is None:
        return

    # Bug is in _run_merged_draft, not _propose
    original_method = getattr(cls, "_run_merged_draft", None)
    if original_method is None:
        return
    if getattr(original_method, "_wings_parallel_spec_patched", False):
        return

    import inspect

    source = inspect.getsource(original_method)
    # Only patch if the bug is present (hardcoded vllm_config reference)
    if "self.vllm_config.model_config.max_model_len" not in source:
        return

    def patched_run_merged_draft(self, *args, **kwargs):
        # Temporarily shadow vllm_config.model_config.max_model_len with
        # self.max_model_len so that the hardcoded references in the body
        # of _run_merged_draft use the correct draft-model limit.
        original_max = self.vllm_config.model_config.max_model_len
        if self.max_model_len != original_max:
            self.vllm_config.model_config.max_model_len = self.max_model_len
            try:
                return original_method(self, *args, **kwargs)
            finally:
                self.vllm_config.model_config.max_model_len = original_max
        return original_method(self, *args, **kwargs)

    patched_run_merged_draft._wings_parallel_spec_patched = True  # pylint: disable=protected-access
    cls._run_merged_draft = patched_run_merged_draft


def patch_vllm_ascend_parallel_spec_decode():
    print(
        "[Wings Engine Patch] vllm_ascend parallel_spec_decode patch enabled",
        file=sys.stderr,
    )
    _register_or_apply_post_import_hook(
        "vllm_ascend.spec_decode.draft_proposer",
        _patch_draft_proposer_module,
    )
    _register_or_apply_post_import_hook(
        "vllm_ascend.spec_decode.eagle_proposer",
        _patch_eagle_proposer_module,
    )
