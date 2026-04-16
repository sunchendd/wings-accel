from typing import Callable, Tuple, Optional

import torch

from .base import SparseWorkerBase


LayerHook = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
FFNHook = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


# Pass-through implementation (for scenarios where sparse is disabled)
def _layer_passthrough(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
):
    return positions, hidden_states, residual


def _ffn_passthrough(hidden_states: torch.Tensor, residual: torch.Tensor):
    return hidden_states, residual


# Mutable implementation pointer:
#   1. defaults to pass-through
#   2. after initialization, can be switched to directly invoke the agent’s methods
#
# Use dynamic hook registration to reduce frequent Agent validation calls
_LAYER_BEGIN_IMPL: LayerHook = _layer_passthrough
_LAYER_FINISHED_IMPL: LayerHook = _layer_passthrough
_FFN_BEGIN_IMPL: FFNHook = _ffn_passthrough
_FFN_FINISHED_IMPL: FFNHook = _ffn_passthrough


def maybe_execute_sparse_layer_begin(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
):
    return _LAYER_BEGIN_IMPL(positions, hidden_states, residual)


def maybe_execute_sparse_layer_finished(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
):
    return _LAYER_FINISHED_IMPL(positions, hidden_states, residual)


def maybe_execute_sparse_ffn_begin(hidden_states: torch.Tensor, residual: torch.Tensor):
    return _FFN_BEGIN_IMPL(hidden_states, residual)


def maybe_execute_sparse_ffn_finished(hidden_states: torch.Tensor, residual: torch.Tensor):
    return _FFN_FINISHED_IMPL(hidden_states, residual)


# Install hooks after agent initialized
def install_sparse_hooks(agent: Optional[SparseWorkerBase]) -> None:
    global _LAYER_BEGIN_IMPL, _LAYER_FINISHED_IMPL, _FFN_BEGIN_IMPL, _FFN_FINISHED_IMPL

    _LAYER_BEGIN_IMPL = agent.layer_begin
    _LAYER_FINISHED_IMPL = agent.layer_finished
    _FFN_BEGIN_IMPL = agent.ffn_begin
    _FFN_FINISHED_IMPL = agent.ffn_finished


def uninstall_sparse_hooks() -> None:
    global _LAYER_BEGIN_IMPL, _LAYER_FINISHED_IMPL, _FFN_BEGIN_IMPL, _FFN_FINISHED_IMPL

    _LAYER_BEGIN_IMPL = _layer_passthrough
    _LAYER_FINISHED_IMPL = _layer_passthrough
    _FFN_BEGIN_IMPL = _ffn_passthrough
    _FFN_FINISHED_IMPL = _ffn_passthrough
