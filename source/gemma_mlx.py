# Inference-only Gemma model implementation for MLX
import re
from typing import (
    Any, 
    List, 
    Optional, 
    Sequence, 
    Tuple, 
    Union)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors
import mlx
import mlx.nn as mx
import mlx.core as mxc
from source.config import *
from source.tokenizer import *


def MLXprecompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> mxc.array:
    """
    Precomputes the frequencey cis.
    """
    freqs = 1.0 / (theta ** (mxc.arange(0, dim, 2)[:(dim // 2)].astype(mxc.float32) / dim))
    t = mxc.arange(end)
    freqs = mxc.outer(t, freqs).astype(mxc.float32)
    cos = mxc.ones_like(freqs) * mxc.cos(freqs)
    sin = mxc.ones_like(freqs) * mxc.sin(freqs)
    freq_cis = mxc.stack([cos, sin], axis = -1)
    return freq_cis


def MLXapply_rotary_emb(x: mxc.array, freqs_cis: mxc.array) -> mxc.array:
    x_transpose = x.transpose(0, 2, 1, 3).astype(mxc.float32) # step 1
    x_real = x_transpose[:, :, :, :x_transpose.shape[3]//2] # step 2
    x_imag = x_transpose[:, :, :, x_transpose.shape[3]//2:]
    x_     = mxc.stack([x_real, x_imag], axis = -1) # step 3 ~ step4

    x_out_real = x_[:, :, :, :, 0] * freqs_cis[:, :, 0] - x_[:, :, :, :, 1] * freqs_cis[:, :, 1] # step 5
    x_out_imag = x_[:, :, :, :, 1] * freqs_cis[:, :, 0] + x_[:, :, :, :, 0] * freqs_cis[:, :, 1]
    x_out = mxc.stack([x_out_real, x_out_imag], axis = -1)

    # 해결해야 할 부분
    x_out__real = x_out[:, :, :, :, 0][:, :, :, :, None]
    x_out__imag = x_out[:, :, :, :, 1][:, :, :, :, None]
    x_out = mxc.concatenate([x_out__real, x_out__imag], axis = 3)
    x_out = mxc.reshape(x_out, (x_out.shape[0], x_out.shape[1], x_out.shape[2], -1)).transpose(0, 2, 1, 3)
    return x_out
    

class MLXSampler(mx.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    # @torch.no_grad()
    def __call__(self,
        embedding: mxc.array,
        hidden_states: mxc.array,
        output_positions: mxc.array,
        temperatures: Union[mxc.array, None],
        top_ps: mxc.array,
        top_ks: mxc.array,
        embedding_bias: Optional[mxc.array] = None,
        ) -> mxc.array:

        # Select the last element for each sequence.
        # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
        # output_position에 해당하는 인덱스 값의 dim = 1을 읽기
        hidden_states = hidden_states[:, output_positions, :]
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states[:, None, :]
        hidden_states = hidden_states.squeeze(axis=1)

        # embedding.t()와 matmul하여 256000개의 단어 사전 로짓을 계산
        logits = mxc.matmul(hidden_states, embedding.T)
        if embedding_bias is not None:
            logits += embedding_bias

        # temperature가 None이면, 가장 큰 값을 로짓으로 선택
        # 아니면, temperature 스케일링 적용
        if temperatures is None:
            return mxc.argmax(logits, dim=-1).squeeze(dim=-1)
        logits = logits / (temperatures[None, :])

        # 1. 모든 가능한 단어에 대한 모델 예측의 확률 분포를 계산
        # 2. 내림차순으로 정렬, probs_idx는 내림차순한 원소들이 몇 번 인덱스 인지를 반환
        probs = mxc.softmax(logits, axis=-1).astype(mxc.float32)
        probs_sort = mxc.sort(probs, axis = -1)
        probs_idx  = mxc.argsort(probs, axis = -1)
        
        ## 오름차순으로 정렬되어있어서 이 기준으로 연산
        # 1. 정렬된 확률의 누적합을 계산 -> 모든 값을 더하면 끝에서 1
        # 2. 누적합과 내림차순확률의 차이를 계산 -> 차이가 top_pos보다 큰 것만 선택 
        # 3. 누적확률이 top p에 도달하기 단어만 선택
        # 4. [0.5, 0.3, 0.2] top p = 0.8이면 [0.5, 0.3] 만 선택
        probs_sum   = mxc.cumsum(probs_sort, axis = -1, reverse = True)
        top_ps_mask = (probs_sum - probs_sort) > top_ps[None, :]
        probs_sort  = mxc.where(top_ps_mask, 0, probs_sort)

        # 1. probs_idx 길이만큼의 0 ~ 숫자 텐서 생성
        # 2. top_ks보다 큰 것은 True로 마스킹
        # 3. 마스킹한 위치가 True이면 0, False이면 probs_sort로 하여 선택
        top_ks_mask = mxc.arange(probs_idx.shape[-1])
        top_ks_mask = top_ks_mask[None, :]
        # top_ks_mask = top_ks_mask >= top_ks[None, :] 오름차순으로 사용하기 때문에 주석 처리
        top_ks_mask = top_ks_mask < top_ks[None, :]
        probs_sort  = mxc.where(top_ks_mask, 0, probs_sort)

        # 1. top-p, top-k로 필터링된 probs_sort를 재정규화
        # 2. 필터링된 과정에서 prob_sort를 probs_idx에 따라 재정렬
        probs_sort = probs_sort / probs_sort.sum(axis = -1, keepdims = True)
        # 오름차순으로 사용하기 때문에 주석 처리
        probs_sort = torch.Tensor(np.array(probs_sort))
        probs_sort = torch.flip(probs_sort, dims = [1])
        probs_idx  = torch.Tensor(np.array(probs_idx.astype(mxc.int64)))
        probs_idx  = torch.flip(probs_idx, dims = [1])
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))
        # MLX의 mxc.multinomial을 구현해보자.
        next_token_ids = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(dim=-1)
        next_token_ids = mxc.array(next_token_ids.numpy())
        return next_token_ids
    

class MLXEmbedding(mx.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, quant: bool):
        super().__init__()
        """
        1. in_features, out_feature를 받아서 embedding 레이어를 만든다.
        2. Quantization을 한다면, out_feuatre로 weight_scaler를 만들어서 곱한다.
        """
        if quant:
            self.embedding     = mx.Embedding(num_embeddings, embedding_dim)
            self.weight_scaler = mxc.zeros(shape = [num_embeddings])
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
        self.weight = mxc.zeros(shape = [dim])

    def _norm(self, x):
        x_ = x ** 2
        x_ = x_.mean(-1, keepdims = True)
        return x * mxc.rsqrt(x_ + self.eps)

    def __call__(self, x):
        x = x.astype(mxc.float32)
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
            self.weight        = mxc.zeros(shape = [in_features, out_features], dtype = mxc.int8)
            self.weight_scaler = mxc.zeros(shape = [out_features])
        else:
            self.weight        = mxc.zeros(shape = [in_features, out_features], dtype = mxc.int8)
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
        self.phi = mxc.array(np.pi)
        self.gate_proj = MLXLinear(hidden_size, intermediate_size, quant)
        self.up_proj   = MLXLinear(hidden_size, intermediate_size, quant)
        self.down_proj = MLXLinear(intermediate_size, hidden_size, quant)

    def gelu_appro_tanh(self, x):
        output = 0.5 * x * (1 + mxc.tanh(mxc.sqrt(2/self.phi) * (x + 0.044715 * (x ** 3))))
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
        self.qkv_proj    = MLXLinear(
            self.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, quant=quant)
        self.o_proj      = MLXLinear(
            self.num_heads * self.head_dim, self.hidden_size, quant=quant)

    def __call__(self, 
        hidden_states: mxc.array, 
        freqs_cis: mxc.array,
        kv_write_indices: mxc.array,
        kv_cache: Tuple[mxc.array, mxc.array], 
        mask: mxc.array) -> mxc.array:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape
        qkv = self.qkv_proj(hidden_states)
        xq = qkv[:, :, :self.q_size]
        xk = qkv[:, :, self.q_size:self.q_size + self.kv_size]
        xv = qkv[:, :, self.q_size+self.kv_size:]
        xq = xq.reshape(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.reshape(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.reshape(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Positional embedding.
        xq = MLXapply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = MLXapply_rotary_emb(xk, freqs_cis=freqs_cis)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        # cache의 kv_write_indices 인덱스를 xk의 kv_write_indices로 채우기
        k_cache, v_cache = kv_cache
        k_cache[:, kv_write_indices, ...] = xk[:, kv_write_indices, ...]
        v_cache[:, kv_write_indices, ...] = xv[:, kv_write_indices, ...]
    
        key   = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key   = mxc.repeat(key, self.num_queries_per_kv, axis = 2)
            value = mxc.repeat(value, self.num_queries_per_kv, axis = 2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(0, 2, 1, 3)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(0, 2, 1, 3)
        v = value.transpose(0, 2, 1, 3)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        scores = mxc.matmul(q, k.transpose(0, 1, 3, 2)) * self.scaling
        scores = scores + mask
        # 240507: 소프트맥스 연산 전후가 다름
        scores = mxc.softmax(scores.astype(mxc.float32), axis=-1).astype(q.dtype)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = mxc.matmul(scores, v)
        # print(v[0][0][0][:5])
        # print(output[0][0][0][:5])

        # [batch_size, input_len, hidden_dim]
        # output = (output.transpose(0, 2, 1, 3).contiguous().view(batch_size, input_len, -1))
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, input_len, -1)
        output = self.o_proj(output)
        return output


class MLXGemmaDecoderLayer(mx.Module):
    def __init__(self, config):
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
        self.input_layernorm          = MLXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MLXRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self,
        hidden_states: mxc.array,
        freqs_cis: mxc.array,
        kv_write_indices: mxc.array,
        kv_cache: Tuple[mxc.array, mxc.array],
        mask: mxc.array,
        ) -> mxc.array:
        """
        1. hidden -> RMSNorm -> GemmaAttention = hidden + residual
        2. hidden -> RMXNorm -> GemmaMLP -> hidden + residual
        """
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        print("어텐션 첫 번쨰 레이어 노름", hidden_states)
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
    def __init__(self, config):
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

    def __call__(self,
        hidden_states: mxc.array,
        freqs_cis: mxc.array,
        kv_write_indices: mxc.array,
        kv_caches: List[Tuple[mxc.array, mxc.array]],
        mask: mxc.array,
        ) -> mxc.array:

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
            # if i == 0:
            #     break
            
        # 마지막 RMSNorm 레이어 포워드
        hidden_states = self.norm(hidden_states)
        return hidden_states
    

class MLXGemmaForCausalLM(mx.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0
        print("dtype :   ", config.dtype)
        max_seq_len    = config.max_position_embeddings
        head_dim       = config.head_dim
        vocab_size     = config.vocab_size
        self.tokenizer = tokenizer.Tokenizer(config.tokenizer)
        self.embedder  = MLXEmbedding(vocab_size, config.hidden_size, config.quant)
        self.model     = MLXGemmaModel(config)
        self.sampler   = MLXSampler(vocab_size)

        rope_theta = getattr(config, "rope_theta", 10000)
        self.freqs_cis = MLXprecompute_freqs_cis(head_dim, max_seq_len * 2, theta = rope_theta)

        self.tensors = self.get_weight()
        value        = self.tensors.get("embedder.weight", self.tensors.get("embed_tokens.weight", "Key not found"))
        self.embedder.embedding.weight = mxc.array(value.numpy()).astype(mxc.float32)

        mlx_weight  = mxc.load("model/gemma-1.1-2b-it/safe_mlx_model.safetensors")
        for idx in range(len(self.model.layers)):
            self.model.layers[idx].input_layernorm.weight = mlx_weight[
                "model.layers.{}.input_layernorm.weight".format(idx)]
            self.model.layers[idx].self_attn.qkv_proj.weight = mlx_weight[
                "model.layers.{}.self_attn.qkv_proj.weight".format(idx)]
            self.model.layers[idx].self_attn.o_proj.weight = mlx_weight[
                "model.layers.{}.self_attn.o_proj.weight".format(idx)]
            self.model.layers[idx].mlp.gate_proj.weight = mlx_weight[
                "model.layers.{}.mlp.gate_proj.weight".format(idx)]
            self.model.layers[idx].mlp.up_proj.weight = mlx_weight[
                "model.layers.{}.mlp.up_proj.weight".format(idx)]
            self.model.layers[idx].mlp.down_proj.weight = mlx_weight[
                "model.layers.{}.mlp.down_proj.weight".format(idx)]
            self.model.layers[idx].post_attention_layernorm.weight = mlx_weight[
                "model.layers.{}.post_attention_layernorm.weight".format(idx)]
        self.model.norm.weight = mlx_weight["model.norm.weight"]
        print("MLX weight 업로드")

    def __call__(self,
        input_token_ids: mxc.array,
        input_positions: mxc.array,
        kv_write_indices: mxc.array,
        kv_caches: List[Tuple[mxc.array, mxc.array]],
        mask: mxc.array,
        output_positions: mxc.array,
        temperatures: Union[mxc.array, None],
        top_ps: mxc.array,
        top_ks: mxc.array,
        **kwargs,
        ) -> mxc.array:
        freqs_cis        = self.freqs_cis[mxc.array(input_positions.numpy())]
        kv_write_indices = mxc.array(input_positions.numpy())

        # 프롬프트 아이디를 임베딩: 해당되는 단어 아이디만 2048 차원 벡터로 변환하여 행렬 구성
        # embedder.weight.shape = [batch_size, 256000, 2048]
        # hidden_states.shape = [batch_size, input_len, 2048]
        hidden_states = self.embedder(mxc.array(input_token_ids.numpy()))
        # Gemma normalizes the embedding by sqrt(hidden_size).
        hidden_states = hidden_states * (self.config.hidden_size**0.5)
        hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mxc.array(mask.numpy()),
            )

        # HC: embedder의 weight를 reuse한다.
        embedder_weight = self.embedder.embedding.weight
        if self.config.quant:
            embedder_weight = (embedder_weight * self.embedder.weight_scaler.unsqueeze(-1))
        next_tokens = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=mxc.array(output_positions.numpy()),
            temperatures=mxc.array(temperatures.numpy()),
            top_ps=mxc.array(top_ps.numpy()),
            top_ks=mxc.array(top_ks.numpy()),
            )
        return next_tokens
    
    def generate(self,
        prompts: Union[str, Sequence[str]],
        output_len: int = 100,
        temperature: Union[float, None] = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
        ) -> Union[str, Sequence[str]]:
        """
        Generates responses for given prompts using Gemma model.
        HC: Mac에서 추론할 것이므로 .to(device)는 모두 제거
        """
        # If a single prompt is provided, treat it as a batch of 1.
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size     = len(prompts) # 1개의 문장이면 batch_size = 1
        prompt_tokens  = [self.tokenizer.encode(prompt) for prompt in prompts] # 배치의 각 프롬프트트들을 인코딩
        min_prompt_len = min(len(p) for p in prompt_tokens) # 숫자로 표현한 프롬프트들 중 가장 짧은 프롬프트 길이
        max_prompt_len = max(len(p) for p in prompt_tokens) # 숫자로 표현한 프롬프트들 중 가장 긴 프롬프트 길이
        max_seq_len    = max_prompt_len + output_len # 출력 길이는 100
        assert max_seq_len <= self.config.max_position_embeddings

        # MLX KV 캐시 빌드
        # num_hidden_layers 수 많큼 size, dtype 크기의 torch.zeros k, v를 kv_caches에 담기
        kv_caches = []
        for _ in range(self.config.num_hidden_layers):
            size    = (batch_size, max_seq_len, self.config.num_key_value_heads, self.config.head_dim)
            # dtype   = config.get_dtype()
            k_cache = mxc.zeros(shape=size, dtype=mxc.float32)
            v_cache = mxc.zeros(shape=size, dtype=mxc.float32)
            kv_caches.append((k_cache, v_cache))

        # HC: 프롬프트를 토크나이징하고, 숫자 아이디로 매핑
        token_ids_tensor       = torch.full((batch_size, max_seq_len), self.tokenizer.pad_id, dtype=torch.int64)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len), self.tokenizer.pad_id, dtype=torch.int64)
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(p[:min_prompt_len])

        prompt_mask_tensor     = token_ids_tensor != self.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64) # tensor([0, 1, 2, 3, 4, 5])

        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1)

        curr_mask_tensor        = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1])
        temperatures_tensor = None if not temperature else torch.FloatTensor([temperature] * batch_size)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size)
        output_index  = torch.tensor(min_prompt_len, dtype=torch.int64)

        # HC: 실제 모델 포워드, 
        # max_sqe_len - min_prompt_len의 의미: 아마도 2개 이상의 배치를 추론할때 토큰 길이를 맞추기 위해서이지 않을까?
        for i in range(max_seq_len - min_prompt_len):
            # 처음에는 입력 프롬프트 전체를 넣고, 입력 프롬프트를 통해 K, V를 연산하여 보관
            # 두 번째부터는 출력 토큰을 다시 입력으로 넣어서 다음 언어를 K, V를 참고하여 예측
            # 각 출력마다 다음 단어를 예측하고 이들을 모아서 하나의 출력 문장을 구성
            next_token_ids = self(
                input_token_ids=input_token_ids_tensor, # tensor([[   2,  651, 6996,  576, 1913,  603]])
                input_positions=input_positions_tensor, # tensor([0, 1, 2, 3, 4, 5])
                kv_write_indices=None, # None
                kv_caches=kv_caches, # torch.zeros의 K, V size를 num_hidden_layer만큼 리스트 선언
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor, # 상수: min_prompt_len - 1
                temperatures=temperatures_tensor, # 상수: 0.95
                top_ps=top_ps_tensor, # tensor([1.])
                top_ks=top_ks_tensor, # tensor([100])
                )
            
            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(dim=1)
            curr_token_ids   = token_ids_tensor.index_select(1, output_index).squeeze(dim=1)
            torch_next_token_ids = torch.tensor(np.array(next_token_ids))
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids, torch_next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor  = output_token_ids
            input_positions_tensor  = output_index.unsqueeze(dim=-1)
            curr_mask_tensor        = mask_tensor.index_select(2, input_positions_tensor)
            output_positions_tensor = torch.tensor(0, dtype=torch.int64)
            output_index = output_index + 1

        # HC: 디토크나이징 과정, token_ids_tensor를 문장으로 치환
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i]) + output_len]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))
            
        # 하나의 문장으로 반환
        return results[0] if is_str_prompt else results


    def get_weight(self):
        tensors     = {}
        tmp_tensors = {}

        # *.safetensor 버전
        # path        = "./model/gemma-1.1-2b-it/"
        # model_name  = "model-{}-of-{}.safetensors"
        # model1 = safetensors.safe_open(path+model_name.format("00001", "00002"), framework="pt")
        # model2 = safetensors.safe_open(path+model_name.format("00002", "00002"), framework="pt")

        # # saftetensor를 그대로 쓰지 않고, q_proj, k_proj, v_proj를 하나로 엮어서 qkv_proj로 만들어야 올바른 추론이 됨
        # for key in model1.keys() + model2.keys():
        #     if key in model1.keys():
        #         tmp_tensors[key] = model1.get_tensor(key).type(torch.float32)
        #     elif key in model2.keys():
        #         tmp_tensors[key] = model2.get_tensor(key).type(torch.float32)

        # for idx in range(18):
        #     qkv_weight = torch.cat([
        #         tmp_tensors["model.layers.{}.self_attn.q_proj.weight".format(idx)],
        #         tmp_tensors["model.layers.{}.self_attn.k_proj.weight".format(idx)],
        #         tmp_tensors["model.layers.{}.self_attn.v_proj.weight".format(idx)]],
        #         dim = 0)
        #     tmp_tensors["model.layers.{}.self_attn.qkv_proj.weight".format(idx)] = qkv_weight

        # for key in tmp_tensors.keys():
        #     # q, k, v 행렬은 concat하여 qkv_proj 형태로 저장하므로 continue
        #     if "q_proj" in key or "k_proj" in key or "v_proj" in key and "qkv_proj" not in key:
        #         continue
        #     else:
        #         tensors[key] = tmp_tensors[key]
        
        # *.ckpt 버전
        weight = torch.load("model/gemma-1.1-2b-it/gemma-1.1-2b-it.ckpt", mmap=True, weights_only=True)["model_state_dict"]
        for key in weight.keys():
            if "complex" in str(weight[key].dtype):
                continue
            tensors[key] = weight[key].type(torch.float32)
        return tensors