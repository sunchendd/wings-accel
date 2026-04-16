from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union, Dict, Tuple

import torch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.worker.gpu_input_batch import InputBatch, CachedRequestState
    from vllm.v1.worker.tpu_input_batch import InputBatch as TpuInputBatch
    from vllm.v1.worker.gpu_model_runner import PerLayerAttnMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheManager, KVCacheBlocks


class SparseRunnerRole(enum.Enum):
    """Role of the current process"""
    SCHEDULER = 0  # Running in the Scheduler process (CPU)
    WORKER = 1     # Running in the Worker process (GPU)


class SparseMetadata(ABC):
    """
    Metadata definition: This serves as the control-plane communication protocol
    between the Scheduler and the Worker for the sparse algorithm.

    The Scheduler generates the metadata, and the Worker consumes it.

    For example, the metadata may include:
        - The currently selected block IDs for each request (e.g., top-k, sliding window, etc.)
        - The sparsity configuration per layer (e.g., mask, sub-window, etc.)

    Note: Some sparse algorithms(CRSA、PAGED-CRSA、BMSA) may use metadata only within the WorkerAgent.
    This is suitable for scenarios where the coordination between the Scheduler and Worker is relatively simple.
    """
    pass


# ==============================
#  Scheduler-side Hooks Agent
# ==============================

class SparseSchedulerBase(ABC):
    """
    [Scheduler Side Interface] Instances of this class exist only in the Scheduler process.

    Responsibilities:
        1. Estimate the number of sparse slots required by a Request, i.e., the token budget.
        2. Allocate KV blocks based on the budget.
    """

    def __init__(self, vllm_config: VllmConfig):
        self._vllm_config = vllm_config

    @property
    def role(self) -> SparseRunnerRole:
        return SparseRunnerRole.SCHEDULER

    def need_sparse(self, request: Request) -> bool:
        """
        控制当前Request是否适合启用稀疏化，注意该方法控制稀疏算法启用的总开关

        比如：
            1. 一些稀疏算法在seq_len > 8K时效果良好，而其它算法可能在seq_len > 16K时才会有不错的表现
            2. 稀疏算法与某些模型可能存在不兼容
            3. 某些任务类型或者其它环境条件对稀疏算法的兼容性差异等
        """
        return False

    @abstractmethod
    def schedule_begin(self, request_id: Union[int, str], prompt_token_ids: List[int]):
        """
        This is called at the beginning of "Scheduler->add_request" function.
        """
        pass

    @abstractmethod
    def schedule_finished(self, request_id: Union[int, str]):
        """
        This is called inside "Scheduler->finish_requests" function.
        Generate the metadata required by sparse worker agent.
        """
        pass

    @abstractmethod
    def estimate_num_slots(self, request: Request) -> int:
        """
        This is called at the beginning of "Scheduler->schedule" function.
        Compute the required sparse slots for a request, i.e., sparse budget control.
        """
        pass

    @abstractmethod
    def update_state_after_alloc(self, request: Request, num_blocks: int):
        """
        Update sparse state after block allocation if necessary.
        """
        pass

    def allocate_slots(
        self,
        kv_cache_manager: KVCacheManager,
        request: Request,
        num_slots_sparse: int,
    ) -> Optional[KVCacheBlocks]:
        """
        This is called in KVCacheManager->allocate_slots, where KV blocks are allocated based on the slots budget.
        """
        pass


# ==============================
#  Worker-side Hooks Agent
# ==============================

class SparseWorkerBase(ABC):
    """
    [Worker Side Interface] Instances of this class exist only in the Worker process (ModelRunner).

    Responsibilities:
        1. Build Metadata
        2. Execute various hooks within the ModelRunner execution flow to realize sparse attention
    """

    def __init__(self, vllm_config: VllmConfig):
        self._vllm_config = vllm_config
        self._sparse_metadata: Optional[SparseMetadata] = None

    @property
    def role(self) -> SparseRunnerRole:
        return SparseRunnerRole.WORKER

    # ==============================
    # Metadata Management
    # ==============================

    def bind_sparse_metadata(self, sparse_metadata: SparseMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            metadata (dict): the connector metadata.
        """
        self._sparse_metadata = sparse_metadata

    def clear_sparse_metadata(self) -> None:
        """Clear the sparse metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self._sparse_metadata = None

    def _get_sparse_metadata(self) -> SparseMetadata:
        """Get the sparse metadata.

        This function should only be called inside the UCMSparse.

        Returns:
            SparseMetadata: the UCM sparse metadata.
        """

        # Should only be called while set to valid metadata.
        assert self._sparse_metadata is not None
        return self._sparse_metadata

    # ==============================
    # Execution Hooks
    # ==============================

    @abstractmethod
    def build_sparse_meta(
        self,
        scheduler_output: SchedulerOutput,
        requests: Dict[str, CachedRequestState],
        input_batch: Union[InputBatch, TpuInputBatch],
        attn_metadata: PerLayerAttnMetadata,
    ) -> SparseMetadata:
        """
        Build the sparse metadata for this step.
        """
        pass

    @abstractmethod
    def execute_model_begin(self, scheduler_output: SchedulerOutput) -> None:
        """
        This is called at the beginning of "ModelRunner->execute_model" function.
        """
        pass

    @abstractmethod
    def execute_model_finished(self, logits_indices: torch.Tensor) -> torch.Tensor:
        """
        This is called at the end of "ModelRunner->execute_model" function.
        """
        return logits_indices

    @abstractmethod
    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        output: Optional[torch.Tensor] = None,
        phase: Optional[str] = None,
        k_hash: Optional[torch.Tensor] = None,
        decode_ql_nope: Optional[torch.Tensor] = None,
        decode_q_pe: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This is called at the beginning of "unified_attention".
        Sparse attention algorithm can modify forward_context.attn_metadata if necessary.
        """
        return query, key, value, output

    @abstractmethod
    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
    ) -> None:
        """
        This is called at the end of "unified_attention".
        """
        pass

    def ffn_begin(
            self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is called at the beginning of ffn in each DecodeLayer.
        """
        return hidden_states, residual

    def ffn_finished(
            self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is called at the end of ffn in each DecodeLayer.
        """
        return hidden_states, residual

    def layer_begin(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This is called at the beginning of DecodeLayer.
        """
        return positions, hidden_states, residual

    def layer_finished(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This is called at the end of DecodeLayer.
        """
        return positions, hidden_states, residual

    @abstractmethod
    def request_finished(self, request_id: Union[int, str]) -> None:
        """
        This function releases the resources of finished requests at worker-side.
        """
        pass
