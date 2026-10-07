# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module
from typing import TYPE_CHECKING, Any

from .protocol import TokenizerLike

if TYPE_CHECKING:
    from .hf import maybe_make_thread_pool
    from .registry import (
        TokenizerRegistry,
        cached_get_tokenizer,
        cached_tokenizer_from_config,
        get_tokenizer,
    )

# Exported name -> submodule. Imported on first access: they pull in the
# Hugging Face tokenizer stack, while many modules only need `TokenizerLike`.
_LAZY_ATTRS = {
    "maybe_make_thread_pool": "hf",
    "TokenizerRegistry": "registry",
    "cached_get_tokenizer": "registry",
    "cached_tokenizer_from_config": "registry",
    "get_tokenizer": "registry",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        return getattr(import_module(f".{_LAZY_ATTRS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TokenizerLike",
    "TokenizerRegistry",
    "cached_get_tokenizer",
    "get_tokenizer",
    "cached_tokenizer_from_config",
    "maybe_make_thread_pool",
]
