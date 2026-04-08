from typing import Dict, Union, Optional

from vllm.logger import init_logger
from vllm.config import VllmConfig

from .base import SparseSchedulerBase, SparseWorkerBase, SparseRunnerRole
from .factory import SparseAlgorithmFactory
from .hooks import install_sparse_hooks

logger = init_logger(__name__)


# Global Sparse Agent container:
#   - key: SparseRunnerRole
#   - value: Agent instance for the corresponding role
#
#   Note: In certain vLLM deployment scenarios, such as single-GPU, a single process may simultaneously host
#         instances of both Scheduler and Worker roles
_Global_SPARSE_AGENTS: Dict[
    SparseRunnerRole, Union[SparseSchedulerBase, SparseWorkerBase]
] = {}


def ensure_sparse_algorithm_initialized(
    vllm_config: VllmConfig, role: SparseRunnerRole
) -> None:
    """
    Initialize (or ensure existence of) the singleton Agent for the given Role.

    Note:
        1. In multiprocess mode, a process usually initializes only one Role.
        2. In single-process single-GPU mode, the same process may initialize both
        SCHEDULER and WORKER sequentially; this function creates a separate Agent instance for each Role.
    """
    global _Global_SPARSE_AGENTS

    if vllm_config.sparse_config is None:
        return

    sparse_config = vllm_config.sparse_config
    sparse_algo_name = sparse_config.sparse_algo_type

    if role in _Global_SPARSE_AGENTS:
        existing_agent = _Global_SPARSE_AGENTS[role]
        logger.warning(
            "Sparse agent for role %s already initialized with algo: %s",
            role.name, getattr(existing_agent, "algo_name", sparse_algo_name),
        )
        return

    # No agent has been initialized for this role; create a new one
    logger.info("Initializing Sparse agent (%s) with algorithm: %s", role.name, sparse_algo_name)

    agent = SparseAlgorithmFactory.create_sparse_algorithm(vllm_config, role=role)
    _Global_SPARSE_AGENTS[role] = agent

    # Install dynamic common hooks
    if role == SparseRunnerRole.WORKER:
        install_sparse_hooks(agent)

    # If an agent for another role already exists in the current process, we are in single-process mode
    other_roles = [r for r in _Global_SPARSE_AGENTS.keys() if r is not role]
    if other_roles:
        logger.info(
            "Sparse agents for multiple roles are initialized in the same process: %s."
            "This is expected in single-process setups.",
            [r.name for r in other_roles] + [role.name])


def get_sparse_agent(role: Optional[SparseRunnerRole] = None) -> Union[SparseSchedulerBase, SparseWorkerBase]:
    """
    Get Sparse Agent instance

    role:
        - If a Role is specified, return the corresponding Agent. Raises an error if that Role has not been initialized.

        - If None (default):
            1. When the current process hosts only one Agent (common in distributed setups),
            return that Agent directly.

            2. When the current process hosts multiple Agents (single-machine single-GPU),
            raise an ambiguity error and force the caller to specify a Role.
    """
    global _Global_SPARSE_AGENTS

    if not _Global_SPARSE_AGENTS:
        raise RuntimeError("No sparse agents initialized. Call ensure_sparse_algorithm_initialized first.")

    # Role explicitly specified
    if role is not None:
        if role not in _Global_SPARSE_AGENTS:
            raise RuntimeError(f"Sparse agent for role {role.name} is not initialized in this process.")
        return _Global_SPARSE_AGENTS[role]

    # Role not specified (attempting to infer)
    if len(_Global_SPARSE_AGENTS) == 1:
        return next(iter(_Global_SPARSE_AGENTS.values()))

    # Ambiguous (both Scheduler and Worker exist in single process)
    else:
        raise ValueError(
            "Multiple sparse agents (Scheduler & Worker) found in the same process. "
            "You must explicitly specify the 'role' argument to get_sparse_agent()."
        )


def has_sparse_agent(role: Optional[SparseRunnerRole] = None) -> bool:
    """
    Check whether the Agent has been initialized

    role:
        - If None, check whether any-role Agent has been initialized.
        - If a specific Role is provided, check only whether that Role has been initialized.
    """
    global _Global_SPARSE_AGENTS

    if role is None:
        return bool(_Global_SPARSE_AGENTS)

    return role in _Global_SPARSE_AGENTS
