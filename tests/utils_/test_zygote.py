# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The zygote must start processes that behave like spawned ones, and fall
back to spawn whenever it cannot."""

import multiprocessing as mp
import os
import signal
import sys
import threading
import time

import pytest

from vllm.utils import zygote

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the zygote is Linux-only"
)


def _report(conn) -> None:
    conn.send(
        {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "parent": mp.parent_process().pid,
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


def _report_grandchild(conn) -> None:
    ctx = zygote.get_context()
    r, w = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_report, args=(w,))
    proc.start()
    w.close()
    conn.send(r.recv())
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
    zygote.start(preload=["json"])
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


def test_grandchild_is_forked_by_the_same_zygote(ctx):
    child = _run(ctx, _report)
    grandchild = _run(ctx, _report_grandchild)
    assert grandchild["ppid"] == child["ppid"]


def test_changed_exec_time_variable_falls_back_to_spawn(ctx, monkeypatch):
    # Hash randomization is fixed when an interpreter starts: a forked child
    # could not honor the new value.
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    info = _run(ctx, _report)
    assert info["ppid"] == os.getpid()
    assert info["hash_randomization"] == 0


def test_unreachable_zygote_falls_back_to_spawn(ctx, monkeypatch, tmp_path):
    monkeypatch.setenv(zygote.ADDRESS_ENV, str(tmp_path / "missing.sock"))
    assert _run(ctx, _report)["ppid"] == os.getpid()


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
        assert "preload-helper" in zygote._fork_hazard()
    finally:
        stop.set()
        thread.join()
