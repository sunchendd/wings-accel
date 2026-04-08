import sys
import types


# These tests only exercise Python-side sparse plumbing. They do not require
# the compiled vLLM extension module to be present in the test environment.
sys.modules.setdefault("vllm._C", types.ModuleType("vllm._C"))
