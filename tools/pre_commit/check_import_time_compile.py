# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forbid calling torch.compile / torch.compiler / torch._dynamo at import time.

Decorating a function with `torch.compile` (or `torch.compiler.disable`, ...)
imports Dynamo and Inductor, which costs every process importing the module
2-3s. Module-level and class-level decorators and calls run at import time;
use `vllm.utils.torch_utils.lazy_torch_compile` instead, or apply the
decorator inside a function.
"""

import ast
import sys
from pathlib import Path

FORBIDDEN_PREFIXES = ("torch.compile", "torch.compiler.", "torch._dynamo.")

# Files allowed to do so, with the reason.
ALLOWED_FILES = {
    # Only imported when the flex attention backend is selected, and imports
    # torch.nn.attention.flex_attention (and so Dynamo) anyway.
    "vllm/v1/attention/backends/flex_attention.py",
    # Worker-only; the fused MoE runner is imported where Dynamo is too.
    "vllm/model_executor/layers/fused_moe/runner/moe_runner.py",
    # Worker-only model that imports torch._dynamo for its config patch.
    "vllm/model_executor/models/diffusion_gemma.py",
    # Imported only for Cohere ASR; `stft` must be excluded from Dynamo before
    # the feature extractor is compiled.
    "vllm/transformers_utils/processors/cohere_asr.py",
}
EXCLUDED_DIRS = ("vllm/third_party/",)


def _dotted_name(node: ast.expr) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _forbidden_call(node: ast.expr) -> str | None:
    """The forbidden callable `node` evaluates, if any."""
    target = node.func if isinstance(node, ast.Call) else node
    name = _dotted_name(target)
    if name is not None and (
        name == "torch.compile" or name.startswith(FORBIDDEN_PREFIXES)
    ):
        return name
    return None


def _import_time_nodes(body: list[ast.stmt]):
    """Yield (node, expr) for expressions evaluated when `body` runs at import
    time: decorators, and calls in assignments and expression statements.
    Class bodies run at import time too; function bodies do not."""
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in stmt.decorator_list:
                yield stmt, decorator
            if isinstance(stmt, ast.ClassDef):
                yield from _import_time_nodes(stmt.body)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.Expr)):
            if isinstance(stmt.value, ast.Call):
                yield stmt, stmt.value
        elif isinstance(stmt, (ast.If, ast.Try)):
            yield from _import_time_nodes(stmt.body)
            yield from _import_time_nodes(stmt.orelse)


def check_file(path: str) -> list[str]:
    source = Path(path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return []
    errors = []
    for stmt, expr in _import_time_nodes(tree.body):
        if (name := _forbidden_call(expr)) is not None:
            errors.append(
                f"{path}:{stmt.lineno}: `{name}` runs at import time and imports "
                "Dynamo; use vllm.utils.torch_utils.lazy_torch_compile or apply "
                "it inside a function."
            )
    return errors


def main(paths: list[str]) -> int:
    errors = []
    for path in paths:
        posix = Path(path).as_posix()
        if (
            not posix.startswith("vllm/")
            or posix.startswith(EXCLUDED_DIRS)
            or posix in ALLOWED_FILES
        ):
            continue
        errors.extend(check_file(path))
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
