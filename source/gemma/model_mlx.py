"""
Inference-only Gemma model implementation for MLX
"""

import re
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
import mlx
import mlx.nn as mx
import mlx.core as mx_core
from source.gemma import config as gemma_config
from source.gemma import tokenizer


# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> mx_core.array:
#     """
#     Precomputes the frequency cis.
#     """
#     freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
#     t     = torch.arange(end, device=freqs.device)
#     freqs     = torch.outer(t, freqs).float()
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#     return freqs_cis


# def apply_rotary_emb(x: mx_core.array, freqs_cis: mx_core.array) -> mx_core.array:
#     """
#     Applies the rotary embedding to the query and key tensors.
#     """
#     x_    = torch.view_as_complex(torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
#     x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
#     x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
#     x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)
#     return x_out
    

class MLXEmbedding(mx.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        """
        1. in_features, out_feature를 받아서 embedding 레이어를 만든다.
        2. Quantization을 한다면, out_feuatre로 weight_scaler를 만들어서 곱한다.
        """
        if quant:
            self.embedding     = mx.Embedding(num_embeddings, embedding_dim)
            self.weight_scaler = mx_core.zeros(shape = [num_embeddings])
        else:
            self.embedding     = mx.Embedding(num_embeddings, embedding_dim)
        self.quant = quant
    
    def __call__(self, x):
        if self.quant:
            self.embedding.weight = self.embedding.weight * self.weight_scaler[ :, None]
        output = self.embedding(x)
        return output
    

class MLXRMSNorm(mx.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
        ):
        super().__init__()
        """
        https://arxiv.org/abs/1910.07467
        """
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = mx_core.zeros(shape = [dim])

    def _norm(self, x):
        x = x ** 2
        x = x.mean(-1, keepdims = True)
        return x * mx_core.rsqrt(x + self.eps)

    def __call__(self, x):
        x = x.astype(mx_core.float32)
        x = self._norm(x).astype(x.dtype)

        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
        return output
    

class MLXLinear(mx.Module):
    def __init__(self, in_features: int, out_features: int, quant: bool):
        super().__init__()
        """
        torch: [in_dim, out_dim] @ we
        1. in_features, out_feature를 받아서 MLP 레이어를 만든다.
        2. Quantization을 한다면, out_feuatre로 weight_scaler를 만들어서 곱한다.
        """
        if quant:
            self.weight        = mx_core.zeros(shape = [in_features, out_features], dtype = mx_core.int8)
            self.weight_scaler = mx_core.zeros(shape = [out_features])
        else:
            self.weight        = mx_core.zeros(shape = [in_features, out_features], dtype = mx_core.int8)
        self.quant = quant

    def __call__(self, x):
        weight = self.weight
        if self.quant:
            weight = weight * self.weight_scaler[:, None]
        output = x @ weight.T
        return output


class MLXGemmaMLP(mx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant: bool,
        ):
        super().__init__()
        self.phi = mx_core.array(np.pi)
        self.gate_proj = MLXLinear(hidden_size, intermediate_size, quant)
        self.up_proj   = MLXLinear(hidden_size, intermediate_size, quant)
        self.down_proj = MLXLinear(intermediate_size, hidden_size, quant)

    def gelu_appro_tanh(self, x):
        output = 0.5 * x * (1 + mx_core.tanh(mx_core.sqrt(2/self.phi) * (x + 0.044715 * (x ** 3))))
        return output
        
    def __call__(self, x):
        gate = self.gate_proj(x)
        gate = self.gelu_appro_tanh(gate)
        up   = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs
    

## HC: MLXGemmaAttention을 수정해야함
class MLXGemmaAttention(mx.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, quant: bool):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim    = head_dim

        self.q_size      = self.num_heads * self.head_dim
        self.kv_size     = self.num_kv_heads * self.head_dim
        self.scaling     = self.head_dim**-0.5
        self.qkv_proj    = MLXLinear(self.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, quant=quant)
        self.o_proj      = MLXLinear(self.num_heads * self.head_dim, self.hidden_size, quant=quant)

    def __call__(self, 
        hidden_states: mx_core.array, 
        freqs_cis: mx_core.array,
        kv_write_indices: mx_core.array,
        kv_cache: Tuple[mx_core.array, mx_core.array], 
        mask: mx_core.array) -> mx_core.array:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key   = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(batch_size, input_len, -1))
        output = self.o_proj(output)
        return output


class MLXGemmaDecoderLayer(mx.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.self_attn = MLXGemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            quant=config.quant,
            )
        self.mlp = MLXGemmaMLP(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size, quant=config.quant)
        self.input_layernorm = MLXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MLXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
        hidden_states: mx_core.array,
        freqs_cis: mx_core.array,
        kv_write_indices: mx_core.array,
        kv_cache: Tuple[mx_core.array, mx_core.array],
        mask: mx_core.array,
        ) -> mx_core.array:

        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MLXGemmaModel(mx.Module):
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        """
        gemma-2b의 경우 num_hidden_layers = 18
        gemma-7b의 경우 num_hidden_layers = 28
        """
        self.config     = config
        self.vocab_size = config.vocab_size

        self.layers = []
        for _ in range(config.num_hidden_layers):
            self.layers.append(MLXGemmaDecoderLayer(config))
        self.norm   = MLXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: mx_core.array,
        freqs_cis: mx_core.array,
        kv_write_indices: mx_core.array,
        kv_caches: List[Tuple[mx_core.array, mx_core.array]],
        mask: mx_core.array,
        ) -> mx_core.array:

        for i in range(len(self.layers)):
            # nn.ModuleList의 GemmaDecoderLayer를 순회
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
                )
            
        # 마지막 RMSNorm 레이어 포워드
        hidden_states = self.norm(hidden_states)
        return hidden_states