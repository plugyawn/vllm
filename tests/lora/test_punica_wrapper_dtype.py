# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that punica wrapper buffer dtypes match input tensor dtypes.
This specifically tests the fix for XPU/CPU backends that previously
hardcoded float32 buffers regardless of input dtype.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.lora.punica_wrapper.punica_cpu import PunicaWrapperCPU


@pytest.fixture
def cpu_punica_wrapper():
    """Create a CPU punica wrapper for testing."""
    batch_size = 32
    wrapper = PunicaWrapperCPU(
        max_num_batched_tokens=8192,
        max_batches=256,
        device="cpu",
    )
    # Set required attributes for the wrapper to function
    wrapper.is_prefill = False
    wrapper.no_lora = False
    # indices_len stores: [token_lora_len, sampler_len, sampler_padded_len, embeddings_len]
    wrapper.indices_len = [batch_size, batch_size, batch_size, batch_size]
    # Initialize index tensors with valid lora indices (0 = first lora)
    wrapper._token_lora_indices[:batch_size].fill_(0)
    wrapper._sampler_indices[:batch_size].fill_(0)
    return wrapper


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_add_lora_linear_buffer_dtype(cpu_punica_wrapper, dtype):
    """
    Test that add_lora_linear creates buffers with the same dtype as input x.

    Previously, CPU/XPU backends hardcoded torch.float32 for buffers,
    causing dtype mismatches when using bfloat16 models.
    """
    batch_size = 32
    hidden_size = 256
    rank = 8
    num_loras = 4
    output_size = 256

    # Create test tensors with specified dtype
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cpu")
    y = torch.zeros(batch_size, output_size, dtype=dtype, device="cpu")

    # Create LoRA weights - shape: (num_loras, 1, rank, hidden_size) for lora_a
    # and (num_loras, 1, output_size, rank) for lora_b
    lora_a = torch.randn(num_loras, 1, rank, hidden_size, dtype=dtype, device="cpu")
    lora_b = torch.randn(num_loras, 1, output_size, rank, dtype=dtype, device="cpu")

    lora_a_stacked = (lora_a,)
    lora_b_stacked = (lora_b,)
    output_slices = (output_size,)

    # Track buffer creation to verify dtype
    created_buffers = []
    original_zeros = torch.zeros

    def tracking_zeros(*args, **kwargs):
        result = original_zeros(*args, **kwargs)
        created_buffers.append(result)
        return result

    with patch("torch.zeros", side_effect=tracking_zeros):
        cpu_punica_wrapper.add_lora_linear(
            y=y,
            x=x,
            lora_a_stacked=lora_a_stacked,
            lora_b_stacked=lora_b_stacked,
            scale=1.0,
            output_slices=output_slices,
        )

    # Verify all created buffers have the correct dtype
    for i, buf in enumerate(created_buffers):
        assert buf.dtype == dtype, (
            f"Buffer {i} has dtype {buf.dtype}, expected {dtype}. "
            f"This indicates the fix for buffer dtype is not working."
        )

    # Verify output has correct dtype
    assert y.dtype == dtype, f"Output dtype {y.dtype} != input dtype {dtype}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_add_lora_logits_buffer_dtype(cpu_punica_wrapper, dtype):
    """
    Test that add_lora_logits creates buffers with the same dtype as input x.
    """
    batch_size = 32
    hidden_size = 256
    rank = 8
    num_loras = 4
    vocab_size = 1024

    # Create test tensors with specified dtype
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cpu")
    y = torch.zeros(batch_size, vocab_size, dtype=dtype, device="cpu")

    # Create LoRA weights for logits
    lora_a = torch.randn(num_loras, 1, rank, hidden_size, dtype=dtype, device="cpu")
    lora_b = torch.randn(num_loras, 1, vocab_size, rank, dtype=dtype, device="cpu")

    # Track buffer creation to verify dtype
    created_buffers = []
    original_zeros = torch.zeros

    def tracking_zeros(*args, **kwargs):
        result = original_zeros(*args, **kwargs)
        created_buffers.append(result)
        return result

    with patch("torch.zeros", side_effect=tracking_zeros):
        cpu_punica_wrapper.add_lora_logits(
            y=y,
            x=x,
            lora_a_stacked=lora_a,
            lora_b_stacked=lora_b,
            scale=1.0,
        )

    # Verify all created buffers have the correct dtype
    for i, buf in enumerate(created_buffers):
        assert buf.dtype == dtype, (
            f"Buffer {i} has dtype {buf.dtype}, expected {dtype}. "
            f"This indicates the fix for buffer dtype is not working."
        )

    # Verify output has correct dtype
    assert y.dtype == dtype, f"Output dtype {y.dtype} != input dtype {dtype}"
