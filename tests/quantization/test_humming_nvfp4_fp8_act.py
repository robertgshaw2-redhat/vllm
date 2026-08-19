# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for running humming NVFP4 weights with 8-bit activations.

The humming GEMM requires weight scale groups to span at least one MMA K-tile
(256 // activation_bits elements). NVFP4 checkpoints carry group-16 FP8 weight
scales, so 8-bit activations (fp8 per-token, fp8-block via
VLLM_HUMMING_INPUT_QUANT_CONFIG={"dtype": "float8e4m3", "group_size": 128},
int8) require the weights to be requantized to group-32 scales. This is what
enables NVFP4 checkpoints to run with fp8(-block) activations on Hopper.
"""

import pytest
import torch

pytest.importorskip("humming")

from vllm.model_executor.layers.quantization.utils.humming_utils import (  # noqa: E402
    input_schema_to_quant_key,
    maybe_regroup_weight_schema_for_quantized_input,
    weight_schema_to_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    GroupShape,
    ScaleDesc,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kNvfp4Static,
)
from vllm.utils.humming import HummingInputSchema, HummingWeightSchema  # noqa: E402


def nvfp4_weight_schema(**overrides):
    kwargs = dict(
        b_dtype="float4e2m1",
        bs_dtype="float8e4m3",
        weight_scale_group_size=16,
        weight_scale_type="group",
        weight_scale_2_type="tensor",
    )
    kwargs.update(overrides)
    return HummingWeightSchema(**kwargs)


class TestRegroupForQuantizedInput:
    def test_nvfp4_with_fp8_per_token_regroups_to_32(self):
        target = maybe_regroup_weight_schema_for_quantized_input(
            nvfp4_weight_schema(),
            HummingInputSchema(a_dtype="float8e4m3"),
        )
        assert target is not None
        assert target.weight_scale_group_size == 32
        # Everything but the group size is preserved.
        assert target.b_dtype == nvfp4_weight_schema().b_dtype
        assert target.bs_dtype == nvfp4_weight_schema().bs_dtype
        assert target.weight_scale_2_type == nvfp4_weight_schema().weight_scale_2_type

    def test_nvfp4_with_fp8_block_regroups_to_32(self):
        target = maybe_regroup_weight_schema_for_quantized_input(
            nvfp4_weight_schema(),
            HummingInputSchema(a_dtype="float8e4m3", input_scale_group_size=128),
        )
        assert target is not None
        assert target.weight_scale_group_size == 32

    def test_nvfp4_with_int8_regroups_to_32(self):
        target = maybe_regroup_weight_schema_for_quantized_input(
            nvfp4_weight_schema(),
            HummingInputSchema(a_dtype="int8"),
        )
        assert target is not None
        assert target.weight_scale_group_size == 32

    def test_nvfp4_with_bf16_input_unchanged(self):
        assert (
            maybe_regroup_weight_schema_for_quantized_input(
                nvfp4_weight_schema(), HummingInputSchema()
            )
            is None
        )

    def test_nvfp4_with_fp4_input_unchanged(self):
        # 4-bit activations consume group-16 scales natively (Blackwell).
        assert (
            maybe_regroup_weight_schema_for_quantized_input(
                nvfp4_weight_schema(),
                HummingInputSchema(a_dtype="float4e2m1", input_scale_group_size=16),
            )
            is None
        )

    def test_group_32_weights_unchanged(self):
        assert (
            maybe_regroup_weight_schema_for_quantized_input(
                nvfp4_weight_schema(weight_scale_group_size=32),
                HummingInputSchema(a_dtype="float8e4m3"),
            )
            is None
        )

    def test_channelwise_weights_unchanged(self):
        schema = HummingWeightSchema(b_dtype="float8e4m3")
        assert (
            maybe_regroup_weight_schema_for_quantized_input(
                schema, HummingInputSchema(a_dtype="float8e4m3")
            )
            is None
        )

    def test_block_scaled_weights_unchanged(self):
        schema = HummingWeightSchema(
            b_dtype="float8e4m3",
            weight_scale_group_size=128,
            weight_scale_group_size_n=128,
            weight_scale_type="block",
        )
        assert (
            maybe_regroup_weight_schema_for_quantized_input(
                schema, HummingInputSchema(a_dtype="float8e4m3")
            )
            is None
        )

    def test_non_humming_schema_unchanged(self):
        assert (
            maybe_regroup_weight_schema_for_quantized_input(
                object(), HummingInputSchema(a_dtype="float8e4m3")
            )
            is None
        )


class TestQuantKeyMapping:
    def test_nvfp4_weight_schema_maps_to_nvfp4_static(self):
        # Regression: schemas with a per-tensor secondary scale used to hit a
        # nonexistent WeightScaleType.GROUP_TENSOR attribute.
        assert weight_schema_to_quant_key(nvfp4_weight_schema()) == kNvfp4Static

    def test_fp8_per_token_input_maps_to_dynamic_token(self):
        key = input_schema_to_quant_key(HummingInputSchema(a_dtype="float8e4m3"))
        assert key == kFp8DynamicTokenSym

    def test_fp8_block_input_maps_to_dynamic_128(self):
        key = input_schema_to_quant_key(
            HummingInputSchema(a_dtype="float8e4m3", input_scale_group_size=128)
        )
        assert key == kFp8Dynamic128Sym

    def test_fp4_grouped_input_keeps_e8m0_scales(self):
        key = input_schema_to_quant_key(
            HummingInputSchema(a_dtype="float4e2m1", input_scale_group_size=32)
        )
        assert key.scale == ScaleDesc(torch.uint8, False, GroupShape(1, 32))


def test_moe_supports_nvfp4_weights_with_fp8_block_activations():
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingExpertsBase,
    )

    assert HummingExpertsBase._supports_quant_scheme(kNvfp4Static, kFp8Dynamic128Sym)
    assert HummingExpertsBase._supports_quant_scheme(kNvfp4Static, kFp8DynamicTokenSym)
