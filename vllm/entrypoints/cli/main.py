# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM.

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys
from importlib.util import find_spec

from vllm.logger import configure_logging_from_args, init_logger

logger = init_logger(__name__)


def maybe_run_omni() -> bool:
    # If `--omni` arg is passed to the CLI, delegate to vLLM Omni's entrypoint handling
    if "--omni" not in sys.argv:
        return False

    # NOTE: Check the spec instead of importing directly here, since things could
    # fail with ImportError due to mismatched versions if things are moved around.
    spec = find_spec("vllm_omni")
    if spec is None:
        logger.error(
            "--omni flag requires a valid instance of vllm-omni to be installed."
        )
        sys.exit(1)

    from vllm_omni.entrypoints.cli.main import main as omni_main

    logger.info("Delegating entrypoint handling to vllm-omni")
    omni_main()

    return True


# Subcommand -> module that defines it, in `vllm --help` order. Only the
# invoked subcommand's module is imported: importing all of them costs seconds
# (e.g. `preload` pulls in the model loader and Inductor).
_SUBCOMMAND_MODULES = {
    "chat": "vllm.entrypoints.cli.openai",
    "complete": "vllm.entrypoints.cli.openai",
    "serve": "vllm.entrypoints.cli.serve",
    "launch": "vllm.entrypoints.cli.launch",
    "bench": "vllm.entrypoints.cli.benchmark.main",
    "collect-env": "vllm.entrypoints.cli.collect_env",
    "preload": "vllm.entrypoints.cli.preload",
    "run-batch": "vllm.entrypoints.cli.run_batch",
    "snapshot": "vllm.entrypoints.cli.snapshot",
}


# Subcommands that start an engine, and so may use the zygote.
_ENGINE_SUBCOMMANDS = ("serve", "run-batch")


def main():
    if maybe_run_omni():
        return

    import os
    from importlib import import_module

    from vllm.utils.gc_utils import gc_paused_for_imports

    subcommand = sys.argv[1] if len(sys.argv) > 1 else None
    if (
        subcommand in _ENGINE_SUBCOMMANDS
        and os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "zygote"
        and not any(a in ("-h", "--help") or a.startswith("--help=") for a in sys.argv)
    ):
        # First, so that the zygote's imports overlap with ours.
        from vllm.utils import zygote

        zygote.start()

    with gc_paused_for_imports():
        from vllm.entrypoints.serve.utils.api_utils import (
            VLLM_SUBCMD_PARSER_EPILOG,
            cli_env_setup,
        )
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        if subcommand in _SUBCOMMAND_MODULES:
            module_names = [_SUBCOMMAND_MODULES[subcommand]]
        else:
            module_names = list(dict.fromkeys(_SUBCOMMAND_MODULES.values()))
        CMD_MODULES = [import_module(name) for name in module_names]

    if sys.argv[1:2] != ["snapshot"]:
        cli_env_setup()

    if subcommand == "bench":
        import vllm.entrypoints.cli.benchmark.main

        vllm.entrypoints.cli.benchmark.main.maybe_exec_rust_bench()

    # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        logger.debug(
            "Bench command detected, must ensure current platform is not "
            "UnspecifiedPlatform to avoid device type inference error"
        )
        from vllm import platforms

        if platforms.current_platform.is_unspecified():
            from vllm.platforms.cpu import CpuPlatform

            platforms.current_platform = CpuPlatform()
            logger.info(
                "Unspecified platform detected, switching to CPU Platform instead."
            )

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("vllm"),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        if cmd_module.__name__ == "vllm.entrypoints.cli.snapshot":
            new_cmds = cmd_module.cmd_init(
                create_requested=sys.argv[1:3] == ["snapshot", "create"]
            )
        else:
            new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmd = cmds[args.subparser]
        configure_logging_from_args(args)
        cmd.validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
