# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseRenderer
    from .params import ChatParams, TokenizeParams, merge_kwargs
    from .registry import RendererRegistry, renderer_from_config

# Exported name -> submodule. Imported on first access: renderers pull in the
# multimodal processors (transformers, torchvision), which the API server
# loads in the background while the engine starts.
_LAZY_ATTRS = {
    "BaseRenderer": "base",
    "ChatParams": "params",
    "TokenizeParams": "params",
    "merge_kwargs": "params",
    "RendererRegistry": "registry",
    "renderer_from_config": "registry",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        return getattr(import_module(f".{_LAZY_ATTRS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseRenderer",
    "RendererRegistry",
    "renderer_from_config",
    "ChatParams",
    "TokenizeParams",
    "merge_kwargs",
]
