# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the consolidated startup profiler.

These exercise the pure timing/aggregation logic and require no GPU.
"""

import time

import pytest

import vllm.envs as envs
from vllm.profiler.startup import StartupProfiler


@pytest.fixture
def profiler(monkeypatch: pytest.MonkeyPatch) -> StartupProfiler:
    monkeypatch.setattr(envs, "VLLM_STARTUP_PROFILE", True)
    return StartupProfiler()


def test_disabled_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "VLLM_STARTUP_PROFILE", False)
    prof = StartupProfiler()
    with prof.record("phase"):
        pass
    # Nothing recorded and reporting is a no-op even with no phases.
    assert prof._roots == []
    assert prof.report("test") is None


def test_records_top_level_phases(profiler: StartupProfiler) -> None:
    with profiler.record("a"):
        pass
    with profiler.record("b"):
        pass
    assert [p.name for p in profiler._roots] == ["a", "b"]
    # Balanced usage leaves the stack empty.
    assert profiler._stack == []


def test_nesting_builds_tree(profiler: StartupProfiler) -> None:
    with profiler.record("outer"), profiler.record("inner"):
        pass
    assert len(profiler._roots) == 1
    outer = profiler._roots[0]
    assert outer.name == "outer"
    assert [c.name for c in outer.children] == ["inner"]
    assert profiler._stack == []


def test_durations_are_ordered(profiler: StartupProfiler) -> None:
    with profiler.record("slow"):
        time.sleep(0.02)
    with profiler.record("fast"):
        pass
    by_name = {p.name: p.duration for p in profiler._roots}
    assert by_name["slow"] > by_name["fast"]
    assert by_name["slow"] >= 0.02


def test_report_contains_role_and_phases(
    profiler: StartupProfiler, caplog: pytest.LogCaptureFixture
) -> None:
    with profiler.record("init_executor", "load weights"):
        pass
    with caplog.at_level("INFO"):
        profiler.report("EngineCore")
    text = caplog.text
    assert "startup profile [EngineCore]" in text
    assert "init_executor" in text
    # The note is surfaced alongside the timing.
    assert "load weights" in text


def test_report_without_phases_is_silent(
    profiler: StartupProfiler, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO"):
        profiler.report("EngineCore")
    assert "startup profile" not in caplog.text


def test_reset_clears_state(profiler: StartupProfiler) -> None:
    with profiler.record("a"):
        pass
    profiler.reset()
    assert profiler._roots == []
    assert profiler._stack == []


def test_stack_recovers_after_exception(profiler: StartupProfiler) -> None:
    with pytest.raises(ValueError), profiler.record("boom"):
        raise ValueError("boom")
    # The phase is still recorded and the stack is unwound.
    assert [p.name for p in profiler._roots] == ["boom"]
    assert profiler._stack == []
