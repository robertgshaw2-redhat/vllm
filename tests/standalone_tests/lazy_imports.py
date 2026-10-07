# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Description: Check that importing a vLLM entrypoint does not import modules
# that only later stages of startup need.
# The utility function cannot be placed in `vllm.utils`
# this needs to be a standalone script
import subprocess
import sys

# Entrypoint -> modules it must not import. Each entrypoint is imported in a
# fresh interpreter with these modules set to None in `sys.modules`, so that
# importing one is a hard error whose stacktrace shows the importer.
CONTRACTS = {
    # `import vllm` is cheap, so that `vllm serve` can start the zygote first.
    # Lazy import `torch._inductor.async_compile` to avoid creating too many
    # processes before we set the number of compiler threads.
    # Lazy import `cv2` to avoid bothering users who only use text models.
    # `cv2` can easily mess up the environment.
    "vllm": ["torch", "torch._inductor.async_compile", "cv2"],
    # The API server starts the engine before importing its HTTP stack.
    "vllm.entrypoints.cli.serve": [
        "fastapi",
        "uvicorn",
        "openai",
        "torch._dynamo",
        "vllm.renderers.base",
        "vllm.v1.engine.core",
    ],
    # EngineCore neither compiles nor serves HTTP; only multimodal models need
    # the multimodal registry.
    "vllm.v1.engine.core": [
        "fastapi",
        "openai",
        "torch._dynamo",
        "torch._inductor",
        "vllm.multimodal.registry",
    ],
}

CHECK = """
import sys
for name in {forbidden!r}:
    sys.modules[name] = None
import {module}
"""

failed = []
for module, forbidden in CONTRACTS.items():
    code = CHECK.format(module=module, forbidden=forbidden)
    if subprocess.run([sys.executable, "-c", code]).returncode != 0:
        failed.append(module)
if failed:
    sys.exit(f"Importing {failed} imported a forbidden module; see above.")
