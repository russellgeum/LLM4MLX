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
from huggingface_hub import notebook_login
notebook_login()

gemma_tokenizer    = AutoTokenizer.from_pretrained("google/gemma-2b-it")
gemma_ko_tokenizer = AutoTokenizer.from_pretrained("beomi/gemma-ko-2b")

input_text = "Write me a poem about Machine Learning."
print(" ")
print("입력 텍스트: ", input_text)
result = gemma_tokenizer(input_text, return_tensors="pt")
print(result["input_ids"])


print(" ")
print("토크나이저로 텍스트 인코드")
result = gemma_tokenizer.encode(input_text, return_tensors = "pt")
print(result)


# 텍스트를 토크나이징하고 id로 변환하는 과정만 추가
print(" ")
print("토크나이저로 텍스트를 토크나이징하고, 토큰들을 ID로 변환하는 과정")
result = gemma_tokenizer.tokenize(input_text)
result = gemma_tokenizer.convert_tokens_to_ids(result)
print(result)



print(" ")
print("텍스트를 토크나이저로 ID 인코딩할 것을 다시 디코딩하여 텍스트로 복원")
result = gemma_tokenizer(input_text, return_tensors="pt")
result = gemma_tokenizer.decode(result["input_ids"][0])
print(result)


print(" ")
print("encode, decode로만 토크나이저 활용")
result = gemma_tokenizer.encode(input_text, return_tensors = "pt")
result = gemma_tokenizer.decode(result[0])
print(result)



# 텍스트를 토크나이징하고 id로 변환하는 과정만 추가
print(" ")
print("tokenizer -> tokens_to_ids -> ids_to_tokens -> ''.join")
result = gemma_tokenizer.tokenize(input_text)
result = gemma_tokenizer.convert_tokens_to_ids(result)
result = gemma_tokenizer.convert_ids_to_tokens(result)
print(result)
# 토큰들을 공백으로 구분하여 하나의 문자열로 합치기
joined_sentence = ' '.join(result)
print(joined_sentence)


# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
#     """
#     Precomputes the frequency cis.
#     """
#     freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
#     t = torch.arange(end, device=freqs.device)
#     freqs     = torch.outer(t, freqs).float()
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#     return freqs_cis


# def MLXprecompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> mxc.array:
#     """
#     Precomputes the frequencey cis.
#     """
#     freqs = 1.0 / (theta ** (mxc.arange(0, dim, 2)[:(dim // 2)].astype(mxc.float32) / dim))
#     t = mxc.arange(end)
#     freqs = mxc.outer(t, freqs)
#     cos = mxc.ones_like(freqs) * mxc.cos(freqs)
#     sin = mxc.ones_like(freqs) * mxc.sin(freqs)
#     freq_cis = mxc.stack([cos, sin], axis = -1)
#     return freq_cis


# def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
#     """
#     Applies the rotary embedding to the query and key tensors.
#     """
#     x_ = torch.view_as_complex(torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
#     x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
#     x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
#     x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)
#     print(x_out)
#     return x_out


# def MLXapply_rotary_emb(x: mxc.array, freqs_cis: mxc.array) -> mxc.array:
#     x_transpose = x.transpose(0, 2, 1, 3).astype(mxc.float32) # step 1
#     x_real = x_transpose[:, :, :, :x_transpose.shape[3]//2] # step 2
#     x_imag = x_transpose[:, :, :, x_transpose.shape[3]//2:]
#     x_     = mxc.stack([x_real, x_imag], axis = -1) # step 3 ~ step4

#     x_out_real = x_[:, :, :, :, 0] * freqs_cis[:, :, 0] - x_[:, :, :, :, 1] * freqs_cis[:, :, 1] # step 5
#     x_out_imag = x_[:, :, :, :, 1] * freqs_cis[:, :, 0] + x_[:, :, :, :, 0] * freqs_cis[:, :, 1]
#     x_out = mxc.stack([x_out_real, x_out_imag], axis = -1)

#     # 해결해야 할 부분
#     x_out__real = x_out[:, :, :, :, 0][:, :, :, :, None]
#     x_out__imag = x_out[:, :, :, :, 1][:, :, :, :, None]
#     x_out = mxc.concatenate([x_out__real, x_out__imag], axis = 3)
#     x_out = mxc.reshape(x_out, (x_out.shape[0], x_out.shape[1], x_out.shape[2], -1)).transpose(0, 2, 1, 3)
#     print(x_out)
#     return x_out


if __name__ == '__main__':
    print('')
    # freqs_cis1 = precompute_freqs_cis(256, 16384, 10000)
    # freqs_cis2 = MLXprecompute_freqs_cis(256, 16384, 10000)
    # index = torch.arange(0, 6)
    
    # freqs1 = freqs_cis1.index_select(0, index) # for pytorch
    # freqs2 = freqs_cis2[mxc.array(index.numpy())] # mlx
    # inputs = torch.ones(size=[1, 6, 8, 256])

    # apply_rotary_emb(inputs, freqs1)
    # MLXapply_rotary_emb(mxc.array(inputs.numpy()), freqs2)