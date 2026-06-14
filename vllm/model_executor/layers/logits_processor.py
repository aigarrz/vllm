# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that compute logits from hidden_stats."""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform


# --8<-- [start:logits_processor]
@PluggableLayer.register("logits_processor")
class LogitsProcessor(PluggableLayer):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    # --8<-- [end:logits_processor]

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_all_gather = current_platform.use_all_gather()

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
        else:
            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale
        return logits

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """gather/all-gather the logits tensor across model parallel group."""
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not hasattr(self, "_int8v2_initialized"):
            self._int8v2_initialized = True
            weight = lm_head.weight.data
            if (
                weight.dtype in (torch.bfloat16, torch.float16)
                and weight.shape[0] > 100000
            ):
                scales = weight.float().abs().amax(dim=1) / 127.0
                scales = scales.clamp(min=1e-12)
                weight_int8 = (
                    (weight.float() / scales.unsqueeze(1))
                    .round()
                    .clamp(-127, 127)
                    .to(torch.int8)
                )
                lm_head._ww_int8 = weight_int8
                lm_head._ww_scales = scales.to(torch.float16)
                original_size = weight.numel() * weight.element_size()
                lm_head.weight.data = torch.empty(
                    0, device=weight.device, dtype=weight.dtype
                )

                import sys as _sys

                print(
                    "DGX_SPARK_V2: LM Head -> INT8 Batched Triton "
                    f"({list(weight_int8.shape)}, "
                    f"saved {original_size // 1024 // 1024}MB)",
                    file=_sys.stderr,
                    flush=True,
                )

                import triton
                import triton.language as tl

                autotune_configs = [
                    triton.Config(
                        {"BLOCK_M": 64, "BLOCK_K": 256},
                        num_warps=4,
                        num_stages=3,
                    ),
                    triton.Config(
                        {"BLOCK_M": 128, "BLOCK_K": 128},
                        num_warps=4,
                        num_stages=3,
                    ),
                    triton.Config(
                        {"BLOCK_M": 128, "BLOCK_K": 256},
                        num_warps=4,
                        num_stages=2,
                    ),
                    triton.Config(
                        {"BLOCK_M": 128, "BLOCK_K": 256},
                        num_warps=4,
                        num_stages=3,
                    ),
                    triton.Config(
                        {"BLOCK_M": 128, "BLOCK_K": 256},
                        num_warps=8,
                        num_stages=2,
                    ),
                    triton.Config(
                        {"BLOCK_M": 128, "BLOCK_K": 512},
                        num_warps=8,
                        num_stages=2,
                    ),
                    triton.Config(
                        {"BLOCK_M": 256, "BLOCK_K": 128},
                        num_warps=8,
                        num_stages=3,
                    ),
                    triton.Config(
                        {"BLOCK_M": 256, "BLOCK_K": 256},
                        num_warps=8,
                        num_stages=2,
                    ),
                ]

                @triton.autotune(
                    configs=autotune_configs,
                    key=["M", "K", "NUM_BATCH"],
                )
                @triton.jit
                def _int8_lm_head_kernel(
                    out_ptr,
                    w_ptr,
                    x_ptr,
                    s_ptr,
                    M,
                    K,
                    stride_ob,
                    stride_xb,
                    NUM_BATCH: tl.constexpr,
                    BLOCK_M: tl.constexpr,
                    BLOCK_K: tl.constexpr,
                ):
                    pid_m = tl.program_id(0)
                    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                    row_mask = rows < M
                    acc0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    acc3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
                    for ks in range(0, K, BLOCK_K):
                        cols = ks + tl.arange(0, BLOCK_K)
                        col_mask = cols < K
                        w = tl.load(
                            w_ptr + rows[:, None] * K + cols[None, :],
                            mask=row_mask[:, None] & col_mask[None, :],
                            other=0,
                        ).to(tl.float32)
                        x0 = tl.load(
                            x_ptr + cols,
                            mask=col_mask,
                            other=0.0,
                        ).to(tl.float32)
                        acc0 += tl.sum(w * x0[None, :], axis=1)
                        if NUM_BATCH > 1:
                            x1 = tl.load(
                                x_ptr + stride_xb + cols,
                                mask=col_mask,
                                other=0.0,
                            ).to(tl.float32)
                            acc1 += tl.sum(w * x1[None, :], axis=1)
                        if NUM_BATCH > 2:
                            x2 = tl.load(
                                x_ptr + 2 * stride_xb + cols,
                                mask=col_mask,
                                other=0.0,
                            ).to(tl.float32)
                            acc2 += tl.sum(w * x2[None, :], axis=1)
                        if NUM_BATCH > 3:
                            x3 = tl.load(
                                x_ptr + 3 * stride_xb + cols,
                                mask=col_mask,
                                other=0.0,
                            ).to(tl.float32)
                            acc3 += tl.sum(w * x3[None, :], axis=1)
                    scale = tl.load(s_ptr + rows, mask=row_mask, other=1.0).to(
                        tl.float32
                    )
                    tl.store(
                        out_ptr + rows,
                        (acc0 * scale).to(tl.float16),
                        mask=row_mask,
                    )
                    if NUM_BATCH > 1:
                        tl.store(
                            out_ptr + stride_ob + rows,
                            (acc1 * scale).to(tl.float16),
                            mask=row_mask,
                        )
                    if NUM_BATCH > 2:
                        tl.store(
                            out_ptr + 2 * stride_ob + rows,
                            (acc2 * scale).to(tl.float16),
                            mask=row_mask,
                        )
                    if NUM_BATCH > 3:
                        tl.store(
                            out_ptr + 3 * stride_ob + rows,
                            (acc3 * scale).to(tl.float16),
                            mask=row_mask,
                        )

                lm_head._ww_kernel_v2 = _int8_lm_head_kernel

        if hasattr(lm_head, "_ww_int8"):
            vocab_size, hidden_size = lm_head._ww_int8.shape
            inputs = hidden_states.view(-1, hidden_size)
            batch_size = inputs.shape[0]
            output = torch.empty(
                batch_size,
                vocab_size,
                dtype=torch.float16,
                device=inputs.device,
            )
            grid = lambda meta: (  # noqa: E731
                (vocab_size + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
            )
            if batch_size <= 4:
                lm_head._ww_kernel_v2[grid](
                    output,
                    lm_head._ww_int8,
                    inputs.to(torch.float16),
                    lm_head._ww_scales,
                    vocab_size,
                    hidden_size,
                    output.stride(0),
                    inputs.stride(0),
                    NUM_BATCH=batch_size,
                )
            else:
                for idx in range(batch_size):
                    lm_head._ww_kernel_v2[grid](
                        output[idx : idx + 1],
                        lm_head._ww_int8,
                        inputs[idx : idx + 1].to(torch.float16),
                        lm_head._ww_scales,
                        vocab_size,
                        hidden_size,
                        vocab_size,
                        hidden_size,
                        NUM_BATCH=1,
                    )
            logits = output.view(hidden_states.shape[:-1] + (vocab_size,))
            if embedding_bias is not None:
                logits = logits + embedding_bias
        else:
            # Get the logits for the next tokens.
            logits = lm_head.quant_method.apply(
                lm_head,
                hidden_states,
                bias=embedding_bias,
            )

        # Gather logits for TP
        logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel argmax without all-gathering full logits.

        Each TP rank computes local argmax, then only the (value, index) pairs
        are gathered and reduced. Communication: O(batch * 2 * tp_size) vs
        O(batch * vocab_size).
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = get_tensor_model_parallel_world_size()

        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        # Mask out padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        local_max_vals, local_max_indices = logits.max(dim=-1)

        # Convert shard-local indices to global vocab indices.
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start

        if tp_size == 1:
            return global_indices

        # All-gather (value, index) pairs, then reduce to global argmax.
        # Use float32 to avoid bf16 precision loss on large vocab indices.
        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        # [batch, 2] -> [batch, 2 * tp_size]
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        # [batch, tp_size, 2] where [:, :, 0]=values, [:, :, 1]=indices
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
