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


def convert():
    """
    1. *.safetensor 버전
        safetensor는 q, k, v 행렬이 분리되어 있다. 따라서 하나로 합쳐야 한다.
    2. *.ckpt 버전
        ckpt는 qkv 행렬이 하나로 묶여있다. 따라서 합칠 필요가 없다.
    """
    tensors     = {}
    mlx_tensors = {}
    
    # # Step 1: torch safetensor 로드
    # path = "./model/gemma-1.1-2b-it/"
    # model_name = "model-{}-of-{}.safetensors"
    # model1 = safetensors.safe_open(path+model_name.format("00001", "00002"), framework="pt")
    # model2 = safetensors.safe_open(path+model_name.format("00002", "00002"), framework="pt")
    
    # # Step 2: torch safetensor를 MLX 텐서로 변환 float32-> bfloat16
    # for key in model1.keys() + model2.keys():
    #     if key in model1.keys():
    #         tensors[key] = mxc.array(model1.get_tensor(key).type(torch.float32).numpy()).astype(mxc.bfloat16)
    #     elif key in model2.keys():
    #         tensors[key] = mxc.array(model2.get_tensor(key).type(torch.float32).numpy()).astype(mxc.bfloat16)

    # for idx in range(18):
    #     qkv_weight = mxc.concatenate([
    #         tensors["model.layers.{}.self_attn.q_proj.weight".format(idx)],
    #         tensors["model.layers.{}.self_attn.k_proj.weight".format(idx)],
    #         tensors["model.layers.{}.self_attn.v_proj.weight".format(idx)]],
    #         axis = 0)
    #     tensors["model.layers.{}.self_attn.qkv_proj.weight".format(idx)] = qkv_weight

    # # Step 3: qkv_proj를 제외하고 q, k, v_proj는 버리기
    # for key in tensors.keys():
    #     # q, k, v 행렬은 concat하여 qkv_proj 형태로 저장하므로 continue
    #     if "q_orj" in key or "k_proj" in key or "v_proj" in key and "qkv_proj" not in key:
    #         continue
    #     else:
    #         mlx_tensors[key] = tensors[key]

    # torch ckpt 버전
    torch_weight = torch.load("model/gemma-1.1-2b-it/gemma-1.1-2b-it.ckpt", mmap=True, weights_only=True)["model_state_dict"]
    for key in torch_weight:
        if "complex" in str(torch_weight[key].dtype):
            continue
        tensors[key] = mxc.array(torch_weight[key].type(torch.float32).numpy()).astype(mxc.bfloat16)
    mlx_tensors = tensors
    mxc.save_safetensors("model/gemma-1.1-2b-it/ckpt_mlx_model.safetensors", mlx_tensors)
    
    # 향후 다른 모델 지원 예정


if __name__ == "__main__":
    convert()