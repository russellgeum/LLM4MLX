import copy
import numpy as np
import mlx
import mlx.nn as mx
import mlx.core as mxc
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from pprint import pprint
from source.gemma.model import *
from source.gemma.model_mlx import *
from source.gemma.config import *
from source.gemma.tokenizer import *


def getWeight():
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


def getInput(tensors):
    global embedder
    config = GemmaConfig(
                num_hidden_layers=18,
                num_attention_heads=8,
                num_key_value_heads=1,
                hidden_size=2048,
                intermediate_size=16384
                )


    prompts        = "The meaning of life is"
    output_len     = 100
    temperature    = 0.95
    top_p          = 1.0
    top_k          = 100.
    max_seq_len    = config.max_position_embeddings
    head_dim       = config.head_dim
    vocab_size     = config.vocab_size
    tokenizer      = Tokenizer(config.tokenizer)

    # Pre-compute rotary embedding table.
    rope_theta = getattr(config, 'rope_theta', 10000)
    freqs_cis  = precompute_freqs_cis(head_dim, max_seq_len * 2, theta=rope_theta)
    mlx_freqs_cis = MLXprecompute_freqs_cis(head_dim, max_seq_len * 2, theta=rope_theta)


    # If a single prompt is provided, treat it as a batch of 1.
    is_str_prompt  = isinstance(prompts, str)
    if is_str_prompt:
        prompts = [prompts]

    batch_size     = len(prompts) # 1개의 문장이면 batch_size = 1
    prompt_tokens  = [tokenizer.encode(prompt) for prompt in prompts] # 배치의 각 프롬프트트들을 인코딩
    min_prompt_len = min(len(p) for p in prompt_tokens) # 숫자로 표현한 프롬프트들 중 가장 짧은 프롬프트 길이
    max_prompt_len = max(len(p) for p in prompt_tokens) # 숫자로 표현한 프롬프트들 중 가장 긴 프롬프트 길이
    max_seq_len    = max_prompt_len + output_len # 출력 길이는 100
    assert max_seq_len <= config.max_position_embeddings


    # KV 캐시 빌드
    # num_hidden_layers 수 많큼 size, dtype 크기의 torch.zeros k, v를 kv_caches에 담기
    kv_caches = []
    for _ in range(config.num_hidden_layers):
        size    = (batch_size, max_seq_len, config.num_key_value_heads, config.head_dim)
        # dtype   = config.get_dtype()
        k_cache = torch.zeros(size=size, dtype=torch.float32, device = "cpu")
        v_cache = torch.zeros(size=size, dtype=torch.float32, device = "cpu")
        kv_caches.append((k_cache, v_cache))


    # MLX KV 캐시 빌드
    # num_hidden_layers 수 많큼 size, dtype 크기의 torch.zeros k, v를 kv_caches에 담기
    mlx_kv_caches = []
    for _ in range(config.num_hidden_layers):
        size    = (batch_size, max_seq_len, config.num_key_value_heads, config.head_dim)
        # dtype   = config.get_dtype()
        k_cache = mxc.zeros(shape=size, dtype=mxc.float32)
        v_cache = mxc.zeros(shape=size, dtype=mxc.float32)
        mlx_kv_caches.append((k_cache, v_cache))


    # HC: 프롬프트를 토크나이징하고, 숫자 아이디로 매핑
    token_ids_tensor       = torch.full((batch_size, max_seq_len), tokenizer.pad_id, dtype=torch.int64)
    input_token_ids_tensor = torch.full((batch_size, min_prompt_len), tokenizer.pad_id, dtype=torch.int64)
    for i, p in enumerate(prompt_tokens):
        token_ids_tensor[i, :len(p)] = torch.tensor(p)
        input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(p[:min_prompt_len])

    prompt_mask_tensor     = token_ids_tensor != tokenizer.pad_id
    input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64) # tensor([0, 1, 2, 3, 4, 5])

    mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1)

    curr_mask_tensor        = mask_tensor.index_select(2, input_positions_tensor)
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1])
    temperatures_tensor = None if not temperature else torch.FloatTensor([temperature] * batch_size)
    top_ps_tensor = torch.FloatTensor([top_p] * batch_size)
    top_ks_tensor = torch.LongTensor([top_k] * batch_size)
    output_index  = torch.tensor(min_prompt_len, dtype=torch.int64)

    freqs_cis        = freqs_cis.index_select(0, input_positions_tensor)
    mlx_freqs_cis    = mlx_freqs_cis[mxc.array(input_positions_tensor.numpy())]
    kv_write_indices = input_positions_tensor


    ## Model Define
    embedder  = Embedding(vocab_size, config.hidden_size, config.quant)
    value     = tensors.get("embedder.weight", tensors.get("embed_tokens.weight", "Key not found"))
    embedder.weight = nn.Parameter(value.type(torch.float32))

    # 프롬프트 아이디를 임베딩: 해당되는 단어 아이디만 2048 차원 벡터로 변환하여 행렬 구성
    # embedder.weight.shape = [batch_size, 256000, 2048]
    # hidden_states.shape = [batch_size, input_len, 2048]
    hidden_states = embedder(input_token_ids_tensor)
    # Gemma normalizes the embedding by sqrt(hidden_size).
    hidden_states = hidden_states * (config.hidden_size**0.5) 

    return hidden_states, freqs_cis, mlx_freqs_cis, kv_write_indices, kv_caches, mlx_kv_caches, curr_mask_tensor, config


## Load weight torch
# torch_weight = torch.load("model/gemma-1.1-2b-it/gemma-1.1-2b-it.ckpt", mmap=True, weights_only=True)["model_state_dict"]

# 1. 모델 웨이트 로드
torch_weight = getWeight()
# 2. 모델 웨이트의 model. 키 제거 torch_weight -> new_weight
new_weight = {}
for key in torch_weight:
    new_key = key.replace("model.", "")
    new_weight[new_key] = torch_weight[key]


## PyTorch Gemma
hidden_states, freqs_cis, mlx_freqs_cis, kv_write_indices, kv_caches, mlx_kv_caches, mask, config = getInput(new_weight)
print("hidden_state shape:  ", hidden_states.shape, hidden_states.device)
print("freqs_cis shape   :  ", freqs_cis.shape, freqs_cis.device)
print("kv_caches         :  ", kv_caches[0][0].shape, kv_caches[0][0].device)
print("kv_write_indices  :  ", kv_write_indices.shape, kv_write_indices.device)
print("curr_mask_tensor  :  ", mask.shape, mask.device)


torch_model = GemmaModel(config)
torch_model.load_state_dict(new_weight, strict=False)
with torch.no_grad():
    output = torch_model(
        hidden_states=hidden_states,
        freqs_cis=freqs_cis,
        kv_write_indices=kv_write_indices,
        kv_caches=kv_caches,
        mask=mask,
        )
    print("파이토치 모델 추론 결과")
    print(output, output.dtype)


## Load weight mlx
mlx_weight = mxc.load("model/gemma-1.1-2b-it/safe_mlx_model.safetensors")
model = MLXGemmaModel(config)
for idx in range(len(model.layers)):
    model.layers[idx].input_layernorm.weight = mlx_weight[
        "model.layers.{}.input_layernorm.weight".format(idx)]
    model.layers[idx].self_attn.qkv_proj.weight = mlx_weight[
        "model.layers.{}.self_attn.qkv_proj.weight".format(idx)]
    model.layers[idx].self_attn.o_proj.weight = mlx_weight[
        "model.layers.{}.self_attn.o_proj.weight".format(idx)]
    model.layers[idx].mlp.gate_proj.weight = mlx_weight[
        "model.layers.{}.mlp.gate_proj.weight".format(idx)]
    model.layers[idx].mlp.up_proj.weight = mlx_weight[
        "model.layers.{}.mlp.up_proj.weight".format(idx)]
    model.layers[idx].mlp.down_proj.weight = mlx_weight[
        "model.layers.{}.mlp.down_proj.weight".format(idx)]
    model.layers[idx].post_attention_layernorm.weight = mlx_weight[
        "model.layers.{}.post_attention_layernorm.weight".format(idx)]
model.norm.weight = mlx_weight["model.norm.weight"]

with torch.no_grad():
    output = model(
        hidden_states=mxc.array(hidden_states.detach().numpy()),
        freqs_cis=mlx_freqs_cis,
        kv_write_indices=mxc.array(kv_write_indices.numpy()),
        kv_caches=mlx_kv_caches,
        mask=mxc.array(mask.numpy()),
        )
    print("safe from MLX 모델 추론 결과")
    print(output)

mlx_weight = mxc.load("model/gemma-1.1-2b-it/ckpt_mlx_model.safetensors")
model = MLXGemmaModel(config)
for idx in range(len(model.layers)):
    model.layers[idx].input_layernorm.weight = mlx_weight[
        "model.layers.{}.input_layernorm.weight".format(idx)]
    model.layers[idx].self_attn.qkv_proj.weight = mlx_weight[
        "model.layers.{}.self_attn.qkv_proj.weight".format(idx)]
    model.layers[idx].self_attn.o_proj.weight = mlx_weight[
        "model.layers.{}.self_attn.o_proj.weight".format(idx)]
    model.layers[idx].mlp.gate_proj.weight = mlx_weight[
        "model.layers.{}.mlp.gate_proj.weight".format(idx)]
    model.layers[idx].mlp.up_proj.weight = mlx_weight[
        "model.layers.{}.mlp.up_proj.weight".format(idx)]
    model.layers[idx].mlp.down_proj.weight = mlx_weight[
        "model.layers.{}.mlp.down_proj.weight".format(idx)]
    model.layers[idx].post_attention_layernorm.weight = mlx_weight[
        "model.layers.{}.post_attention_layernorm.weight".format(idx)]
model.norm.weight = mlx_weight["model.norm.weight"]

with torch.no_grad():
    output = model(
        hidden_states=mxc.array(hidden_states.detach().numpy()),
        freqs_cis=mlx_freqs_cis,
        kv_write_indices=mxc.array(kv_write_indices.numpy()),
        kv_caches=mlx_kv_caches,
        mask=mxc.array(mask.numpy()),
        )
    print("ckpt from MLX 모델 추론 결과")
    print(output)