"""Legacy import shim for SparseConnector helpers.

Older code imports SparseConnector helpers via
``vsparse.connector.sparse_connector``. The canonical implementation now lives
under ``vsparse.connectors``.
"""

from vsparse.connectors import sparse_connector as _impl
from vsparse.connectors.sparse_connector import *  # noqa: F401,F403

_count_contiguous_external_hits_tp_aware = _impl._count_contiguous_external_hits_tp_aware  # pylint: disable=protected-access
