# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end reproducer: OffloadingConnector async-load holders wedge a rank.

Mechanism (vllm/v1/core/sched/scheduler.py):

1. Requests whose prompt is fully offloaded are admitted as async loads with
   ``num_new_tokens=0`` and no lookahead, so they only take their prefix
   blocks. Nothing is running, so no watermark applies, and a full hit reserves
   nothing for anyone else. Enough holders drain the pool to zero.
2. When the loads land, only the head of ``skipped_waiting`` is promoted. With
   speculative decoding on (``num_lookahead_tokens > 0``) the promoted head now
   needs one block it did not have. ``allocate_slots`` returns None and the
   waiting loop ``break``s.
3. Holders are not preemptible and nothing is running, so nothing ever frees a
   block. Every step repeats. The other holders stay in WAITING_FOR_REMOTE_KVS
   forever although their loads finished; new requests are never admitted.

Any model works; MLA (DeepSeek-V2-Lite) is used because pure full-attention
groups give exact full-prompt hits. Needs one GPU with ~40 GB.

Usage:
    python repro_offload_promotion_deadlock.py            # deadlocks
    python repro_offload_promotion_deadlock.py --no-spec  # control, completes
"""

import argparse
import os
import random
import threading
import time

# Run the engine in-process so the scheduler can be inspected on hang.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams, TokensPrompt  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402

BLOCK_SIZE = 64
BLOCKS_PER_PROMPT = 32
NUM_HOLDERS = 4
# +1 for the null block the pool always holds back: the usable pool is exactly
# NUM_HOLDERS full prompts, so admitting them all leaves zero free blocks.
NUM_GPU_BLOCKS = NUM_HOLDERS * BLOCKS_PER_PROMPT + 1
PROMPT_TOKENS = BLOCKS_PER_PROMPT * BLOCK_SIZE


def make_prompt(seed: int, vocab_size: int) -> TokensPrompt:
    rng = random.Random(seed)
    ids = [rng.randrange(1000, vocab_size - 1000) for _ in range(PROMPT_TOKENS)]
    return TokensPrompt(prompt_token_ids=ids)


def scheduler_of(llm: LLM):
    core = getattr(llm.llm_engine.engine_core, "engine_core", None)
    return getattr(core, "scheduler", None)


def dump_scheduler(llm: LLM) -> str | None:
    sched = scheduler_of(llm)
    if sched is None:
        return None
    free_blocks = sched.kv_cache_manager.block_pool.get_num_free_blocks()
    header = (
        f"running={len(sched.running)} waiting={len(sched.waiting)} "
        f"skipped_waiting={len(sched.skipped_waiting)} free_blocks={free_blocks} "
        f"finished_recving_kv_req_ids={sorted(sched.finished_recving_kv_req_ids)}"
    )
    lines = [header]
    for req in sched.requests.values():
        lines.append(
            f"  req {req.request_id}: {req.status.name} "
            f"num_computed_tokens={req.num_computed_tokens} "
            f"num_tokens={req.num_tokens}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    parser.add_argument("--no-spec", action="store_true", help="control run")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    speculative_config = None
    if not args.no_spec:
        speculative_config = {
            "method": "ngram",
            "num_speculative_tokens": 2,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        }

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=2 * PROMPT_TOKENS,
        max_num_batched_tokens=2 * PROMPT_TOKENS,
        max_num_seqs=16,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        num_gpu_blocks_override=NUM_GPU_BLOCKS,
        speculative_config=speculative_config,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"cpu_bytes_to_use": 4 * 1024**3},
        ),
    )
    vocab_size = llm.llm_engine.model_config.get_vocab_size()
    prompts = [make_prompt(seed, vocab_size) for seed in range(NUM_HOLDERS)]

    # Phase 1: run each prompt once so its KV is stored in the CPU tier.
    for prompt in prompts:
        llm.generate(prompt, SamplingParams(max_tokens=1))
    time.sleep(3)  # let deferred GPU->CPU stores drain
    assert llm.reset_prefix_cache(), "GPU prefix cache reset failed"
    print("phase 1 done: prompts offloaded, GPU prefix cache cleared")

    # Phase 2: resubmit all prompts at once on the idle engine. Each is a full
    # CPU hit, so all become async-load holders and drain the pool.
    results: list = []

    def run():
        results.extend(llm.generate(prompts, SamplingParams(max_tokens=4)))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(args.timeout)

    if not worker.is_alive():
        print(f"completed: {len(results)} outputs (no deadlock)")
        return

    print(f"DEADLOCK: no output after {args.timeout:.0f}s")
    state = dump_scheduler(llm)
    if state is not None:
        print(state)
        sched = scheduler_of(llm)
        head = next(
            (r for r in sched.requests.values() if r.status.name == "WAITING"),
            None,
        )
        if head is not None:
            # Removing the stuck head lets the finished loads behind it be
            # promoted: this is an ordering artifact, not a resource cycle.
            print(f"aborting head request {head.request_id} ...")
            llm.llm_engine.abort_request([head.request_id])
            worker.join(60)
            print("resumed" if not worker.is_alive() else "still stuck")
            if state := dump_scheduler(llm):
                print(state)
    else:
        for metric in llm.get_metrics():
            if metric.name in (
                "vllm:num_requests_running",
                "vllm:num_requests_waiting",
                "vllm:kv_cache_usage_perc",
            ):
                print(metric)
    os._exit(1)


if __name__ == "__main__":
    main()
