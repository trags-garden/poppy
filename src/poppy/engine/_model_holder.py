"""Shared lazy model holder with timer-driven idle unloading."""

from __future__ import annotations

import gc
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

_DEFAULT_IDLE_TIMEOUT_S = 600.0
_UNLOAD_CHECK_INTERVAL_S = 60.0


def _resolve_idle_timeout(explicit: float | None) -> float:
    if explicit is not None:
        value = explicit
    else:
        raw = os.environ.get("POPPY_MODEL_IDLE_S")
        if raw is None:
            value = _DEFAULT_IDLE_TIMEOUT_S
        else:
            try:
                value = float(raw)
            except ValueError:
                return 0.0
    return value if value > 0 else 0.0


class IdleUnloadingModels:
    """Lazy bi/cross model holder that releases both after an idle timeout."""

    def __init__(
        self,
        *,
        bi_encoder: Any = None,
        cross_encoder: Any = None,
        idle_timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._idle_timeout_s = _resolve_idle_timeout(idle_timeout_s)
        self._clock = clock
        self._injected = bi_encoder is not None or cross_encoder is not None
        self._bi = bi_encoder
        self._cross = cross_encoder
        self._last_use = float("-inf")
        self._lock = threading.RLock()
        self._unload_thread: threading.Thread | None = None

    def _announce_once(self) -> None:
        """Announce a first load. Backend holders override this hook."""

    def _release_backend_caches(self) -> None:
        """Return allocator-cached memory to the OS after an unload.

        Dropping the encoder references only frees Python objects; a backend
        with a caching allocator (torch MPS/CUDA) keeps that memory committed
        for reuse and needs an explicit release. Default is a no-op so
        backends without the problem (ONNX) never pay for it.
        """

    def _load_model(self, kind: str) -> Any:
        raise NotImplementedError

    def _model(self, kind: str) -> Any:
        attr = "_bi" if kind == "bi" else "_cross"
        with self._lock:
            model = getattr(self, attr)
            if model is None:
                self._announce_once()
                model = self._load_model(kind)
                setattr(self, attr, model)
                self._last_use = self._clock()
                self._start_timer_locked()
            return model

    def _mark_used(self) -> None:
        with self._lock:
            self._last_use = self._clock()

    def _start_timer_locked(self) -> None:
        if self._injected or self._idle_timeout_s <= 0:
            return
        if self._unload_thread is not None and self._unload_thread.is_alive():
            return
        thread = threading.Thread(target=self._timer_loop, name="poppy-model-idle-unload", daemon=True)
        self._unload_thread = thread
        thread.start()

    def _check_idle_unload_locked(self) -> bool:
        if self._injected or self._idle_timeout_s <= 0:
            return False
        if self._bi is None and self._cross is None:
            return True
        if self._clock() - self._last_use < self._idle_timeout_s:
            return False
        self._bi = None
        self._cross = None
        self._last_use = float("-inf")
        # Collect first so freed tensors land back in the backend allocator,
        # then ask the backend to hand its cache to the OS.
        gc.collect()
        self._release_backend_caches()
        return True

    def check_idle_unload(self) -> bool:
        """Synchronously unload idle models; return whether the timer may exit."""
        with self._lock:
            return self._check_idle_unload_locked()

    def runtime_status(self) -> tuple[bool, str | None]:
        """Return residency and the projected idle deadline without loading models."""
        with self._lock:
            loaded = self._bi is not None or self._cross is not None
            if not loaded or self._injected or self._idle_timeout_s <= 0:
                return loaded, None
            remaining = max(0.0, self._last_use + self._idle_timeout_s - self._clock())
            deadline = datetime.now(timezone.utc) + timedelta(seconds=remaining)
            return loaded, deadline.isoformat()

    def _timer_loop(self) -> None:
        delay = min(_UNLOAD_CHECK_INTERVAL_S, self._idle_timeout_s)
        while True:
            time.sleep(delay)
            with self._lock:
                if self._check_idle_unload_locked():
                    if self._unload_thread is threading.current_thread():
                        self._unload_thread = None
                    return


def model_runtime_status(engine: object) -> tuple[bool, str | None]:
    """Inspect an engine's optional lazy model holder without forcing a load."""
    holder = getattr(engine, "_models", None)
    if isinstance(holder, IdleUnloadingModels):
        return holder.runtime_status()
    return False, None
