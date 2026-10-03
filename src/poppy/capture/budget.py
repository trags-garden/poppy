"""Shared capture budgets, sized to finish before the single-flight lock expires."""

from poppy.capture.lock import LOCK_TTL_S

# One extraction and one verdict per candidate share the lock's lifetime.
HOST_CLI_TIMEOUT_S = 120
CONFLICT_LLM_TIMEOUT_S = 20

# Leave room for transcript I/O, retrieval, ingest and capture bookkeeping.
CAPTURE_OVERHEAD_S = 30
MAX_CAPTURE_ITEMS = max(0, (LOCK_TTL_S - CAPTURE_OVERHEAD_S - HOST_CLI_TIMEOUT_S) // CONFLICT_LLM_TIMEOUT_S)
