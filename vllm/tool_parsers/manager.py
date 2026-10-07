# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry of tool parsers.

Kept apart from `ToolParser`, whose protocol imports are heavy: the CLI lists
the registered names before the engine starts.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import import_plugin

if TYPE_CHECKING:
    from vllm.tool_parsers.abstract_tool_parser import ToolParser

logger = init_logger(__name__)


class ToolParserManager:
    """Central registry for ToolParser implementations.

    Supports two modes:
      - Eager (immediate) registration via `register_module`
      - Lazy registration via `register_lazy_module`
    """

    tool_parsers: dict[str, type[ToolParser]] = {}
    lazy_parsers: dict[str, tuple[str, str]] = {}  # name -> (module_path, class_name)

    @classmethod
    def get_tool_parser(cls, name: str) -> type[ToolParser]:
        """Retrieve a registered or lazily registered ToolParser class.

        If the parser is lazily registered,
        it will be imported and cached on first access.
        Raises KeyError if not found.
        """
        if name in cls.tool_parsers:
            return cls.tool_parsers[name]

        if name in cls.lazy_parsers:
            return cls._load_lazy_parser(name)

        raise KeyError(f"Tool parser '{name}' not found.")

    @classmethod
    def _load_lazy_parser(cls, name: str) -> type[ToolParser]:
        """Import and register a lazily loaded parser."""
        from vllm.tool_parsers.abstract_tool_parser import ToolParser

        module_path, class_name = cls.lazy_parsers[name]
        try:
            mod = importlib.import_module(module_path)
            parser_cls = getattr(mod, class_name)
            if not issubclass(parser_cls, ToolParser):
                raise TypeError(
                    f"{class_name} in {module_path} is not a ToolParser subclass."
                )
            cls.tool_parsers[name] = parser_cls  # cache
            return parser_cls
        except Exception as e:
            logger.exception(
                "Failed to import lazy tool parser '%s' from %s: %s",
                name,
                module_path,
                e,
            )
            raise

    @classmethod
    def _register_module(
        cls,
        module: type[ToolParser],
        module_name: str | list[str] | None = None,
        force: bool = True,
    ) -> None:
        """Register a ToolParser class immediately."""
        from vllm.tool_parsers.abstract_tool_parser import ToolParser

        if not issubclass(module, ToolParser):
            raise TypeError(
                f"module must be subclass of ToolParser, but got {type(module)}"
            )

        if module_name is None:
            module_name = module.__name__

        if isinstance(module_name, str):
            module_names = [module_name]
        elif is_list_of(module_name, str):
            module_names = module_name
        else:
            raise TypeError("module_name must be str, list[str], or None.")

        for name in module_names:
            if not force and name in cls.tool_parsers:
                existed = cls.tool_parsers[name]
                raise KeyError(f"{name} is already registered at {existed.__module__}")
            cls.tool_parsers[name] = module

    @classmethod
    def register_lazy_module(cls, name: str, module_path: str, class_name: str) -> None:
        """Register a lazy module mapping.

        Example:
            ToolParserManager.register_lazy_module(
                name="kimi_k2",
                module_path="vllm.tool_parsers.kimi_k2_parser",
                class_name="KimiK2ToolParser",
            )

        """
        cls.lazy_parsers[name] = (module_path, class_name)

    @classmethod
    def register_module(
        cls,
        name: str | list[str] | None = None,
        force: bool = True,
        module: type[ToolParser] | None = None,
    ) -> type[ToolParser] | Callable[[type[ToolParser]], type[ToolParser]]:
        """Register module immediately or lazily (as a decorator).

        Usage:
            @ToolParserManager.register_module("kimi_k2")
            class KimiK2ToolParser(ToolParser):
                ...

        Or:
            ToolParserManager.register_module(module=SomeToolParser)
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # Immediate registration
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # Decorator usage
        def _decorator(obj: type[ToolParser]) -> type[ToolParser]:
            module_path = obj.__module__
            class_name = obj.__name__

            if isinstance(name, str):
                names = [name]
            elif name is not None and is_list_of(name, str):
                names = name
            else:
                names = [class_name]

            for n in names:
                # Lazy mapping only: do not import now
                cls.lazy_parsers[n] = (module_path, class_name)

            return obj

        return _decorator

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return names of all eagerly and lazily registered tool parsers."""
        return sorted(set(cls.tool_parsers.keys()) | set(cls.lazy_parsers.keys()))

    @classmethod
    def import_tool_parser(cls, plugin_path: str) -> None:
        """Import a user-defined tool parser."""
        import_plugin(plugin_path)
