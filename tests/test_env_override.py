# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`import vllm` does not import torch: env_override patches torch modules when
they are first imported. A patch whose hook never fires is silently lost."""

import subprocess
import sys
import textwrap
import uuid

from vllm.env_override import _run_after_import


def test_run_after_import_waits_for_the_module(tmp_path, monkeypatch):
    name = f"hook_target_{uuid.uuid4().hex}"
    (tmp_path / f"{name}.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    calls = []

    _run_after_import(name, lambda: calls.append(sys.modules[name].VALUE))
    assert calls == []
    __import__(name)
    assert calls == [1]

    # Already imported: runs right away.
    _run_after_import(name, lambda: calls.append(2))
    assert calls == [1, 2]


def test_inductor_patches_apply_when_inductor_is_imported_after_vllm():
    # A fresh interpreter, since this one may have imported Inductor already.
    code = textwrap.dedent(
        """
        import sys

        import vllm
        from vllm.env_override import is_torch_equal_or_newer

        assert "torch" not in sys.modules
        import torch._inductor.compile_fx
        from torch._inductor import lowering, pattern_matcher

        assert getattr(lowering.FALLBACK_ALLOW_LIST, "_vllm_patched", False)
        if not is_torch_equal_or_newer("2.16.0.dev"):
            patched = pattern_matcher.fallback_node_due_to_unsupported_type
            assert patched.__name__ == "fallback_for_builtin"
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
