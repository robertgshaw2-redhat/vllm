# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .hasher import MultiModalHasher
    from .inputs import BatchedTensorInputs, MultiModalKwargsItems, NestedTensors
    from .registry import MultiModalRegistry

    MULTIMODAL_REGISTRY: MultiModalRegistry
    """
    The global [`MultiModalRegistry`][vllm.multimodal.registry.MultiModalRegistry]
    is used by model runners to dispatch data processing according to the target
    model.

    Info:
        [mm_processing](../../../design/mm_processing.md)
    """

# Exported name -> submodule. Imported on first access: the registry pulls in
# the Hugging Face processors and the hasher the media decoders, which
# processes that only pass multimodal data structures around never use.
_LAZY_ATTRS = {
    "MultiModalHasher": "hasher",
    "BatchedTensorInputs": "inputs",
    "MultiModalKwargsItems": "inputs",
    "NestedTensors": "inputs",
    "MultiModalRegistry": "registry",
}

_registry_lock = threading.RLock()


def __getattr__(name: str) -> Any:
    if name == "MULTIMODAL_REGISTRY":
        with _registry_lock:
            if "MULTIMODAL_REGISTRY" not in globals():
                from .registry import MultiModalRegistry

                globals()["MULTIMODAL_REGISTRY"] = MultiModalRegistry()
        return globals()["MULTIMODAL_REGISTRY"]
    if name in _LAZY_ATTRS:
        return getattr(import_module(f".{_LAZY_ATTRS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BatchedTensorInputs",
    "MultiModalHasher",
    "MultiModalKwargsItems",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]
