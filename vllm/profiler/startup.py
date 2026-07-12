# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consolidated startup-time profiler.

vLLM startup is spread across several coarse phases (worker/executor bring-up,
weight loading, memory profiling, ``torch.compile``, CUDA graph capture and
model warmup). Individual phases already emit their own timing logs, but there
is no single place that shows where total startup time goes, which makes it
hard to prioritize optimization work.

This module records nested phase timings inside a process and, when enabled via
``VLLM_STARTUP_PROFILE=1``, emits one consolidated breakdown once startup
completes. It is process-local and adds negligible overhead when disabled: the
:meth:`StartupProfiler.record` context manager short-circuits to a no-op.

Example output (illustrative numbers)::

    vLLM startup profile [EngineCore] - total 41.48s across 4 measured phase(s)
      load_plugins                    0.12s    0.3%
      init_executor                  18.42s   44.4%   (workers + weight load)
      determine_available_memory      3.11s    7.5%   (memory profiling)
      initialize_from_config         19.83s   47.8%   (kv alloc + compile + capture)
"""

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _Phase:
    """A single timed startup phase, possibly containing nested sub-phases."""

    name: str
    start: float
    note: str = ""
    end: float | None = None
    children: list["_Phase"] = field(default_factory=list)

    @property
    def duration(self) -> float:
        end = self.end if self.end is not None else time.perf_counter()
        return end - self.start


class StartupProfiler:
    """Process-local recorder for nested startup phase timings.

    A single module-level instance (:data:`startup_profiler`) is shared within a
    process. Phases are recorded with :meth:`record` and rendered with
    :meth:`report`. When ``VLLM_STARTUP_PROFILE`` is disabled every method is a
    cheap no-op, so call sites can be left in place unconditionally.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._roots: list[_Phase] = []
        self._stack: list[_Phase] = []

    @property
    def enabled(self) -> bool:
        return envs.VLLM_STARTUP_PROFILE

    @contextmanager
    def record(self, name: str, note: str = "") -> Iterator[None]:
        """Time the wrapped block as a startup phase.

        Nesting follows the call stack: a ``record`` block entered inside
        another becomes its child in the reported tree.

        Args:
            name: Short, stable identifier for the phase.
            note: Optional human-readable hint shown alongside the timing.
        """
        if not self.enabled:
            yield
            return
        phase = _Phase(name=name, start=time.perf_counter(), note=note)
        with self._lock:
            if self._stack:
                self._stack[-1].children.append(phase)
            else:
                self._roots.append(phase)
            self._stack.append(phase)
        try:
            yield
        finally:
            phase.end = time.perf_counter()
            with self._lock:
                # Pop back to this phase. Balanced usage pops exactly one entry;
                # the loop is defensive against unbalanced/overlapping blocks.
                while self._stack and self._stack[-1] is not phase:
                    self._stack.pop()
                if self._stack:
                    self._stack.pop()

    def report(self, role: str) -> None:
        """Log the consolidated breakdown of everything recorded so far.

        Args:
            role: Name of the process/component being profiled, e.g.
                ``"EngineCore"``. Included in the log header so breakdowns from
                different processes are distinguishable.
        """
        if not self.enabled:
            return
        with self._lock:
            roots = list(self._roots)
        if not roots:
            return
        total = sum(phase.duration for phase in roots)
        lines = [
            f"vLLM startup profile [{role}] - total {total:.2f}s "
            f"across {len(roots)} measured phase(s)"
        ]
        for root in roots:
            _render(root, total, 1, lines)
        logger.info("\n".join(lines))

    def reset(self) -> None:
        """Drop all recorded phases. Intended for tests."""
        with self._lock:
            self._roots.clear()
            self._stack.clear()


def _render(phase: _Phase, total: float, depth: int, lines: list[str]) -> None:
    duration = phase.duration
    pct = (duration / total * 100.0) if total > 0 else 0.0
    indent = "  " * depth
    note = f"   ({phase.note})" if phase.note else ""
    lines.append(f"{indent}{phase.name:<32} {duration:8.2f}s {pct:6.1f}%{note}")
    for child in phase.children:
        _render(child, total, depth + 1, lines)
    if phase.children:
        unattributed = duration - sum(child.duration for child in phase.children)
        if unattributed > 0.05:
            child_indent = "  " * (depth + 1)
            lines.append(f"{child_indent}{'(other)':<32} {unattributed:8.2f}s")


startup_profiler = StartupProfiler()
