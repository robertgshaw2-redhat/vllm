# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chat template options and checks needed to parse CLI arguments.

Kept free of heavy imports: the API server parses its arguments before
starting the engine.
"""

from pathlib import Path
from typing import Literal

from vllm.exceptions import VLLMValidationError

# Passed in by user
ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]


def validate_chat_template(chat_template: Path | str | None):
    """Raises if the provided chat template appears invalid."""
    if chat_template is None:
        return

    elif isinstance(chat_template, Path) and not chat_template.exists():
        raise FileNotFoundError("the supplied chat template path doesn't exist")

    elif isinstance(chat_template, str):
        JINJA_CHARS = "{}\n"
        if (
            not any(c in chat_template for c in JINJA_CHARS)
            and not Path(chat_template).exists()
        ):
            # Try to find the template in the built-in templates directory
            from vllm.transformers_utils.chat_templates.registry import (
                CHAT_TEMPLATES_DIR,
            )

            builtin_template_path = CHAT_TEMPLATES_DIR / chat_template
            if not builtin_template_path.exists():
                raise VLLMValidationError(
                    f"The supplied chat template string ({chat_template}) "
                    f"appears path-like, but doesn't exist! "
                    f"Tried: {chat_template} and {builtin_template_path}",
                    parameter="chat_template",
                )

    else:
        raise VLLMValidationError(
            f"{type(chat_template)} is not a valid chat template type",
            parameter="chat_template",
        )
