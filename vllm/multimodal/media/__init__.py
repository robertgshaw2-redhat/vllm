# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .audio import AudioEmbeddingMediaIO, AudioMediaIO
    from .base import MediaIO, MediaWithBytes
    from .connector import MEDIA_CONNECTOR_REGISTRY, MediaConnector
    from .image import ImageEmbeddingMediaIO, ImageMediaIO
    from .video import VIDEO_LOADER_REGISTRY, VideoMediaIO

# Exported name -> submodule. Imported on first access: the decoders pull in
# torchcodec, cv2, scipy and aiohttp, which most processes never use.
_LAZY_ATTRS = {
    "AudioEmbeddingMediaIO": "audio",
    "AudioMediaIO": "audio",
    "MediaIO": "base",
    "MediaWithBytes": "base",
    "MEDIA_CONNECTOR_REGISTRY": "connector",
    "MediaConnector": "connector",
    "ImageEmbeddingMediaIO": "image",
    "ImageMediaIO": "image",
    "VIDEO_LOADER_REGISTRY": "video",
    "VideoMediaIO": "video",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        return getattr(import_module(f".{_LAZY_ATTRS[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MediaIO",
    "MediaWithBytes",
    "AudioEmbeddingMediaIO",
    "AudioMediaIO",
    "ImageEmbeddingMediaIO",
    "ImageMediaIO",
    "VIDEO_LOADER_REGISTRY",
    "VideoMediaIO",
    "MEDIA_CONNECTOR_REGISTRY",
    "MediaConnector",
]
