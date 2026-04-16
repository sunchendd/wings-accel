import importlib
from typing import Callable, Tuple, Type, Union

from vllm.logger import init_logger
from vllm.config import VllmConfig

from .base import SparseSchedulerBase, SparseWorkerBase, SparseRunnerRole

logger = init_logger(__name__)

# Define loader type: a tuple returning (SchedulerClass, WorkerClass)
CommonSparseLoader = Callable[[], Tuple[Type[SparseSchedulerBase], Type[SparseWorkerBase]]]


class SparseAlgorithmFactory:
    _registry: dict[str, CommonSparseLoader] = {}

    @classmethod
    def register_sparse_algorithm(
            cls,
            name: str,
            module_path: str,
            scheduler_class_name: str,
            worker_class_name: str
    ) -> None:
        """
        Register sparse attention algorithms
        Need to provide both Scheduler and Worker class names

        Note:
            module_path can be a package, e.g. vllm.v1.sparse.crsa
            Then in that package’s __init__.py:
                from .scheduler import CRSAScheduler
                from .worker import CRSAWorker
            This way they are not forced to be in a single *.py file
        """
        if name in cls._registry:
            raise ValueError(f"Sparse algorithm '{name}' is already registered.")

        def loader() -> Tuple[Type[SparseSchedulerBase], Type[SparseWorkerBase]]:
            module = importlib.import_module(module_path)

            # Locate the class whose name matches in the specified module
            try:
                sche_cls = getattr(module, scheduler_class_name)
            except AttributeError as e:
                raise ImportError(
                    f"Scheduler class '{scheduler_class_name}' not found in module '{module_path}'."
                ) from e

            try:
                worker_cls = getattr(module, worker_class_name)
            except AttributeError as e:
                raise ImportError(
                    f"Worker class '{worker_class_name}' not found in module '{module_path}'."
                ) from e

            # Validate type
            if not issubclass(sche_cls, SparseSchedulerBase):
                raise TypeError(
                    f"Scheduler class '{scheduler_class_name}' in '{module_path}' "
                    f"must inherit from SparseSchedulerBase."
                )

            if not issubclass(worker_cls, SparseWorkerBase):
                raise TypeError(
                    f"Worker class '{worker_class_name}' in '{module_path}' "
                    f"must inherit from SparseWorkerBase."
                )

            return sche_cls, worker_cls

        cls._registry[name] = loader

    @classmethod
    def create_sparse_algorithm(
            cls, vllm_config: VllmConfig, role: SparseRunnerRole
    ) -> Union[SparseSchedulerBase, SparseWorkerBase]:
        """
        Create the corresponding Sparse instance based on
        the configured name in config and the current Role
        """

        if vllm_config.sparse_config is None:
            raise ValueError("Sparse config not found in vllm config")

        sparse_algorithm_name = vllm_config.sparse_config.sparse_algo_type

        if sparse_algorithm_name not in cls._registry:
            raise ValueError(f"Unsupported sparse algorithm type: {sparse_algorithm_name}")

        sche_cls, worker_cls = cls._registry[sparse_algorithm_name]()

        if role == SparseRunnerRole.SCHEDULER:
            logger.info(f"Creating Sparse Scheduler for algorithm: {sparse_algorithm_name}")
            return sche_cls(vllm_config)

        elif role == SparseRunnerRole.WORKER:
            logger.info(f"Creating Sparse Worker for algorithm: {sparse_algorithm_name}")
            return worker_cls(vllm_config)

        else:
            raise ValueError(f"Invalid role: {role}")


# Register BMSA Algorithm
SparseAlgorithmFactory.register_sparse_algorithm(
    name="BMSA",
    module_path="vsparse.bmsa",
    scheduler_class_name="BMSAScheduler",
    worker_class_name="BMSAWorker"
)

