# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The zygote must start processes that behave like spawned ones, and fall
back to spawn whenever it cannot."""

import atexit
import multiprocessing as mp
import multiprocessing.spawn
import os
import signal
import sys
import tempfile
import threading
import time

import psutil
import pytest

from vllm.utils import zygote
from vllm.utils.system_utils import kill_process_tree

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the zygote is Linux-only"
)


def _report(conn) -> None:
    conn.send(
        {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "parent": mp.parent_process().pid,  # type: ignore[union-attr]
            "name": mp.current_process().name,
            "env": os.environ.get("ZYGOTE_TEST"),
            "hash_randomization": sys.flags.hash_randomization,
        }
    )
    conn.close()


def _exit(code: int) -> None:
    sys.exit(code)


def _sleep() -> None:
    time.sleep(60)


def _report_num_threads(conn) -> None:
    import torch

    conn.send(torch.get_num_threads())


def _touch_at_exit(path: str) -> None:
    atexit.register(lambda: open(path, "w").close())


def _report_grandchild(conn) -> None:
    ctx = zygote.get_context()
    r, w = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_report, args=(w,))
    proc.start()
    w.close()
    conn.send(r.recv())
    proc.join()


def _start_sleeper(conn) -> None:
    proc = zygote.get_context().Process(target=_sleep)
    proc.start()
    conn.send(proc.pid)
    proc.join()


def _run(ctx, target, *args, **kwargs) -> dict:
    r, w = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=target, args=(w, *args), **kwargs)
    proc.start()
    w.close()
    info = r.recv()
    proc.join()
    assert proc.exitcode == 0
    return info


@pytest.fixture(scope="module")
def ctx():
    old = os.environ.pop(zygote.ADDRESS_ENV, None)
    zygote.start(preload=["torch"])
    yield zygote.get_context()
    if old is None:
        os.environ.pop(zygote.ADDRESS_ENV, None)
    else:
        os.environ[zygote.ADDRESS_ENV] = old


def test_child_runs_like_a_spawned_process(ctx, monkeypatch):
    monkeypatch.setenv("ZYGOTE_TEST", "set after the zygote started")
    info = _run(ctx, _report, name="Reporter")
    assert info["env"] == "set after the zygote started"
    assert info["parent"] == os.getpid()
    assert info["name"] == "Reporter"
    # Forked by the zygote, not by us.
    assert info["ppid"] not in (os.getpid(), info["pid"])


def test_exit_code_and_terminate(ctx):
    proc = ctx.Process(target=_exit, args=(3,))
    proc.start()
    proc.join()
    assert proc.exitcode == 3

    proc = ctx.Process(target=_sleep, daemon=True)
    proc.start()
    proc.terminate()
    proc.join(10)
    assert proc.exitcode == -signal.SIGTERM


def test_child_sizes_torch_threads_from_its_environment(ctx, monkeypatch):
    # The zygote imported torch under its own environment.
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert _run(ctx, _report_num_threads) == 3


def test_exit_handlers_run_as_in_spawned_process(ctx, tmp_path):
    marker = tmp_path / "exited"
    proc = ctx.Process(target=_touch_at_exit, args=(str(marker),))
    proc.start()
    proc.join()
    assert proc.exitcode == 0 and marker.exists()


def test_grandchild_is_forked_by_the_same_zygote(ctx):
    child = _run(ctx, _report)
    grandchild = _run(ctx, _report_grandchild)
    assert grandchild["ppid"] == child["ppid"]


def test_kill_process_tree_reaches_processes_forked_for_the_tree(ctx):
    # The sleeper descends from proc, but its parent is the zygote.
    r, w = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_start_sleeper, args=(w,))
    proc.start()
    w.close()
    sleeper = psutil.Process(r.recv())
    kill_process_tree(proc.pid)
    proc.join(10)
    assert proc.exitcode == -signal.SIGKILL
    sleeper.wait(timeout=10)


def test_changed_exec_time_variable_falls_back_to_spawn(ctx, monkeypatch):
    # Hash randomization is fixed when an interpreter starts: a forked child
    # could not honor the new value.
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    info = _run(ctx, _report)
    assert info["ppid"] == os.getpid()
    assert info["hash_randomization"] == 0


def test_other_interpreter_falls_back_to_spawn(ctx, monkeypatch):
    # The same interpreter under another path, as another virtualenv would be.
    executable = os.path.join(os.path.dirname(sys.executable), ".", "python3")
    assert os.path.exists(executable)
    monkeypatch.setattr(multiprocessing.spawn, "_python_exe", os.fsencode(executable))
    assert _run(ctx, _report)["ppid"] == os.getpid()


def test_unreachable_zygote_falls_back_to_spawn(ctx, monkeypatch, tmp_path):
    monkeypatch.setenv(zygote.ADDRESS_ENV, str(tmp_path / "missing.sock"))
    assert _run(ctx, _report)["ppid"] == os.getpid()


def test_start_failure_is_reported(monkeypatch, tmp_path):
    # Unix socket paths are limited to about 100 bytes.
    long_dir = tmp_path / ("d" * 100)
    long_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(long_dir))
    monkeypatch.setattr(zygote, "_zygote", None)
    address = os.environ.get(zygote.ADDRESS_ENV)
    with pytest.raises(zygote.ZygoteUnavailableError):
        zygote.start()
    assert os.environ.get(zygote.ADDRESS_ENV) == address


def test_child_env():
    start = {"A": "1", "LD_PRELOAD": "x", "CHANGED": "old"}
    current = {**start, "CHANGED": "new", "ADDED": "1"}
    depends_on = ["LD_PRELOAD"]

    env = zygote._child_env(start, current, {**start, "OWN": "1"}, depends_on)
    # The requester's variables, plus what preloading changed.
    assert env == {**current, "OWN": "1"}
    # A requester that already made the same changes is served too.
    assert zygote._child_env(start, current, current, depends_on) == current

    changed = {**start, "LD_PRELOAD": "y"}
    assert "LD_PRELOAD" in zygote._child_env(start, current, changed, depends_on)
    changed = {**start, "CHANGED": "other"}
    assert "CHANGED" in zygote._child_env(start, current, changed, depends_on)


def test_fork_hazard_names_threads():
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, name="preload-helper")
    thread.start()
    try:
        hazard = zygote._fork_hazard()
        assert hazard is not None and "preload-helper" in hazard
    finally:
        stop.set()
        thread.join()
