# SPDX-License-Identifier: Apache-2.0

# noqa: UP007
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List

import torch

from vllm.logger import init_logger

try:
    import xgrammar as xgr
    xgr_installed = True
except ImportError:
    xgr_installed = False
    pass

from vllm.model_executor.guided_decoding.utils import (convert_lark_to_gbnf,
                                                       grammar_is_likely_lark)
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from vllm.config import ModelConfig
    from vllm.model_executor.guided_decoding.reasoner import Reasoner
    from vllm.sampling_params import GuidedDecodingParams

logger = init_logger(__name__)


def get_local_xgrammar_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams,
        tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig,
        reasoner: Reasoner | None,
        max_threads: int = 8):
    config = GrammarConfig.from_guided_params(guided_params=guided_params,
                                              model_config=model_config,
                                              tokenizer=tokenizer,
                                              max_threads=max_threads)
    return XGrammarLogitsProcessor(config=config, reasoner=reasoner)


@dataclass(frozen=True)
class TokenizerData:
    """Immutable container for cached tokenizer data."""
    metadata: str
    encoded_vocab: list[str] = field(default_factory=list)


class TokenizerDataCache:
    """Cache manager for tokenizer data to avoid repeated processing."""
    _cache: dict[int, TokenizerData] = {}

    @classmethod
    def get_tokenizer_data(
        cls,
        tokenizer: PreTrainedTokenizer,
        vocab_size: int,
    ) -> TokenizerData:
        tokenizer_hash = hash(tokenizer)

        if tokenizer_hash not in cls._cache:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                tokenizer,
                vocab_size=vocab_size,
            )
            metadata = json.loads(tokenizer_info.dump_metadata())

            # Vendored from xgrammar logic to get encoded_vocab
            # https://github.com/mlc-ai/xgrammar/blob/989222175c2a30fb7987d8bcce35bec1bf6817f2/python/xgrammar/tokenizer_info.py#L127 # noqa: E501
            try:
                vocab_dict = tokenizer.get_vocab()
            except AttributeError as e:
                raise ValueError(
                    f"Cannot get the vocabulary of the tokenizer {type(tokenizer)}. The tokenizer should have a get_vocab method."  # noqa: E501
                ) from e

            # maintain tokenizer's indexing
            encoded_vocab = [""] * tokenizer_info.vocab_size
            for token, idx in vocab_dict.items():
                if idx < tokenizer_info.vocab_size:
                    encoded_vocab[idx] = token

            if isinstance(tokenizer, MistralTokenizer):
                # REF: https://github.com/mlc-ai/xgrammar/blob/5e141f6ff1ca02bc31f9e512e68b61f2a8ae88e5/tests/python/test_tokenizer_info.py#L43 # noqa: E501
                metadata.update({
                    "vocab_type": xgr.VocabType.BYTE_FALLBACK,
                    "add_prefix_space": True
                })

            cls._cache[tokenizer_hash] = TokenizerData(
                encoded_vocab=encoded_vocab,
                metadata=json.dumps(metadata),
            )

        return cls._cache[tokenizer_hash]


class GrammarCompilerCache:
    """
    Cache for GrammarCompiler instances based on tokenizer.

    This cache reduces the overhead of creating new compiler instances when
    using the same tokenizer configuration.
    """
    _cache: dict[str, xgr.GrammarCompiler] = {}

    @classmethod
    def get_compiler(cls, config: GrammarConfig) -> xgr.GrammarCompiler:
        cache_key = str(config.tokenizer_hash)

        if cache_key not in cls._cache:
            config_data = config.tokenizer_data

            # In TokenizerDataCache.get_tokenizer_data, a serializable
            # tokenizer_data is created and cached. This data is used to build
            # a tokenizer_info and create an xgrammar compiler.
            tokenizer_info = xgr.TokenizerInfo.from_vocab_and_metadata(
                encoded_vocab=config_data.encoded_vocab,
                metadata=config_data.metadata,
            )
            cls._cache[cache_key] = xgr.GrammarCompiler(
                tokenizer_info, max_threads=config.max_threads)

        return cls._cache[cache_key]


@dataclass
class GrammarConfig:
    """Serializable configuration for grammar compilation"""
    tokenizer_hash: int
    tokenizer_data: TokenizerData
    json_str: str | None = None
    regex_str: str | None = None
    grammar_str: str | None = None
    json_object: bool | None = None
    any_whitespace: bool = True
    max_threads: int = 8

    @classmethod
    def from_guided_params(cls,
                           guided_params: GuidedDecodingParams,
                           model_config: ModelConfig,
                           tokenizer: PreTrainedTokenizer,
                           max_threads: int = 8) -> GrammarConfig:

        tokenizer_hash = hash(tokenizer)
        tokenizer_data = TokenizerDataCache.get_tokenizer_data(
            tokenizer=tokenizer,
            vocab_size=model_config.hf_text_config.vocab_size,
        )

        if guided_params.json:
            if not isinstance(guided_params.json, str):
                json_str = json.dumps(guided_params.json)
            else:
                json_str = guided_params.json

            backend_options = guided_params.backend_options()
            any_whitespace = 'disable-any-whitespace' not in backend_options

            # Check and log if model with xgrammar and whitespace have history
            # of runaway generation of whitespaces.
            # References:
            # https://github.com/vllm-project/vllm/pull/12744
            # https://github.com/mlc-ai/xgrammar/issues/212
            model_with_warn = None

            if 'Mistral' in model_config.model:
                model_with_warn = 'Mistral'
            elif 'Qwen' in model_config.model:
                model_with_warn = 'Qwen'

            if model_with_warn is not None and any_whitespace:
                logger.info_once(
                    f"{model_with_warn} model detected, consider set `guided_backend=xgrammar:disable-any-whitespace` to prevent runaway generation of whitespaces."  # noqa: E501
                )
            # Validate the schema and raise ValueError here if it is invalid.
            # This is to avoid exceptions in model execution, which will crash
            # the engine worker process.
            try:
                xgr.Grammar.from_json_schema(json_str,
                                             any_whitespace=any_whitespace)
            except RuntimeError as err:
                raise ValueError(str(err)) from err

            return cls(json_str=json_str,
                       tokenizer_hash=tokenizer_hash,
                       max_threads=max_threads,
                       tokenizer_data=tokenizer_data,
                       any_whitespace=any_whitespace)
        elif guided_params.grammar:
            # XGrammar only supports GBNF grammars, so we must convert Lark
            if grammar_is_likely_lark(guided_params.grammar):
                try:
                    grammar_str = convert_lark_to_gbnf(guided_params.grammar)
                except ValueError as e:
                    raise ValueError(
                        "Failed to convert the grammar from Lark to GBNF. "
                        "Please either use GBNF grammar directly or specify"
                        " --guided-decoding-backend=outlines.\n"
                        f"Conversion error: {str(e)}") from e
            else:
                grammar_str = guided_params.grammar

            # Validate the grammar and raise ValueError here if it is invalid.
            # This is to avoid exceptions in model execution, which will crash
            # the engine worker process.
            try:
                xgr.Grammar.from_ebnf(grammar_str)
            except RuntimeError as err:
                raise ValueError(str(err)) from err

            return cls(grammar_str=grammar_str,
                       tokenizer_hash=tokenizer_hash,
                       max_threads=max_threads,
                       tokenizer_data=tokenizer_data)
        elif guided_params.regex:
            try:
                xgr.Grammar.from_regex(guided_params.regex)
            except RuntimeError as err:
                raise ValueError(str(err)) from err
            return cls(
                regex_str=guided_params.regex,
                tokenizer_hash=tokenizer_hash,
                max_threads=max_threads,
                tokenizer_data=tokenizer_data,
            )
        elif guided_params.json_object:
            return cls(
                json_object=True,
                tokenizer_hash=tokenizer_hash,
                max_threads=max_threads,
                tokenizer_data=tokenizer_data,
            )
        elif guided_params.choice:
            choice_str = GrammarConfig.choice_as_grammar(guided_params.choice)
            try:
                xgr.Grammar.from_ebnf(choice_str)
            except RuntimeError as err:
                raise ValueError(str(err)) from err

            return cls(
                grammar_str=choice_str,
                tokenizer_hash=tokenizer_hash,
                max_threads=max_threads,
                tokenizer_data=tokenizer_data,
            )
        else:
            raise ValueError(
                "Currently only support JSON and EBNF grammar mode for xgrammar"
            )

    @staticmethod
    def escape_ebnf_string(s: str) -> str:
        """Escape special characters in a EBNF string."""
        # Escape double quotes and backslashes
        return re.sub(r'(["\\])', r'\\\1', s)

    @staticmethod
    def choice_as_grammar(choice: List[str] | None) -> str:
        if choice is None:
            raise ValueError("Choice is not set")
        escaped_choices = (GrammarConfig.escape_ebnf_string(c) for c in choice)
        grammar = ('root ::= ' + ' | '.join(f'"{c}"' for c in escaped_choices))
        return grammar

    @staticmethod
    def tokenizer_info(tokenizer_data: TokenizerData) -> xgr.TokenizerInfo:
        return xgr.TokenizerInfo.from_vocab_and_metadata(
            encoded_vocab=tokenizer_data.encoded_vocab,
            metadata=tokenizer_data.metadata,
        )


@dataclass
class XGrammarLogitsProcessor:
    """Wrapper class to support pickle protocol"""
    config: GrammarConfig
    reasoner: Reasoner | None = None

    ctx: xgr.CompiledGrammar | None = None
    tokenizer_info: xgr.TokenizerInfo = None  # type: ignore[assignment]
    token_bitmask: torch.Tensor = None  # type: ignore[assignment]
    matcher: xgr.GrammarMatcher = None  # type: ignore[assignment]
    prefilled: bool = field(default=False)

    def __post_init__(self):
        self.tokenizer_info = self.config.tokenizer_info(
            self.config.tokenizer_data)

    def __getstate__(self) -> dict[str, Any]:
        return {'config': self.config, 'reasoner': self.reasoner}

    def __setstate__(self, state: dict[str, Any]):
        self.config = state['config']
        self.reasoner = state['reasoner']

        self.tokenizer_info = GrammarConfig.tokenizer_info(
            self.config.tokenizer_data)
        self.ctx = None
        self.matcher = None
        self.token_bitmask = None  # type: ignore[assignment]
        self.prefilled = False
        """Lazily initialize the processor in the worker process"""

    def __call__(self, input_ids: list[int],
                 scores: torch.Tensor) -> torch.Tensor:

        # Skip the structured logits processing if reasoning is not finished.
        # reasoner is not None only when `--enable-reasoning` is set.
        if self.reasoner is not None and \
        not self.reasoner.is_reasoning_end(
                input_ids):
            return scores

        if self.ctx is None:
            compiler = GrammarCompilerCache.get_compiler(self.config)
            if self.config.json_str is not None:
                self.ctx = compiler.compile_json_schema(
                    self.config.json_str,
                    any_whitespace=self.config.any_whitespace,
                )
            elif self.config.grammar_str is not None:
                self.ctx = compiler.compile_grammar(self.config.grammar_str)
            elif self.config.regex_str is not None:
                self.ctx = compiler.compile_regex(self.config.regex_str)
            elif self.config.json_object:
                self.ctx = compiler.compile_builtin_json_grammar()

        if self.matcher is None:
            self.matcher = xgr.GrammarMatcher(self.ctx)
        if self.token_bitmask is None:
            self.token_bitmask = xgr.allocate_token_bitmask(
                1, self.tokenizer_info.vocab_size)

        if not self.prefilled:
            # Have not sampled a token yet
            self.prefilled = True
        else:
            if not self.matcher.is_terminated():
                sampled_token = input_ids[-1]
                assert self.matcher.accept_token(sampled_token)

        if not self.matcher.is_terminated():
            # @ubospica: ideally, fill_next_token_bitmask should be
            # parallelized with model decoding
            # See https://github.com/vllm-project/vllm/pull/10785/files#r1864278303
            self.matcher.fill_next_token_bitmask(self.token_bitmask)

        # token_bitmask is a CPU tensor for use with accept_token and
        # fill_next_token_bitmask so we move it to the device of scores
        device_type = scores.device.type
        dtype = scores.dtype
        if device_type != "cuda":
            # xgrammar on cpu only supports float32 scores
            # see: https://github.com/mlc-ai/xgrammar/blob/c1b64920cad24f44f235778c1c00bb52d57da01a/python/xgrammar/kernels/apply_token_bitmask_inplace_cpu.py#L22
            scores = scores.to("cpu").float().unsqueeze(0)

        # Note: In this method, if the tensors have different dimensions
        # on CPU device fails, but on GPU it runs without error. Hence the
        # unsqueeze above for scores, to match the token bitmask shape
        xgr.apply_token_bitmask_inplace(
            scores, self.token_bitmask.to(scores.device, non_blocking=True))
        if device_type != "cuda":
            scores = scores.to(dtype).to(device_type).squeeze()

        return scores

    def clone(self) -> XGrammarLogitsProcessor:
        """Create a new instance with shared compiled grammar
          but separate state"""
        new_processor = XGrammarLogitsProcessor(self.config, self.reasoner)

        # Share the compiled grammar context (immutable after compilation)
        new_processor.ctx = self.ctx

        # Create fresh matchers for the new sequence
        if self.ctx is not None:
            new_processor.matcher = xgr.GrammarMatcher(self.ctx)

        # Create a new token bitmask with the same size
        if hasattr(self, 'token_bitmask') and self.token_bitmask is not None:
            new_processor.token_bitmask = self.token_bitmask

        # Reset prefilled state for new sequence
        new_processor.prefilled = False

        return new_processor
