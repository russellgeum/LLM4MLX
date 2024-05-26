import copy
import numpy as np
import safetensors
import mlx
import mlx.nn as mx
import mlx.core as mxc
import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint
from source import *
from source.config import *
from source.tokenizer import *
from source.gemma_mlx import *
from source.gemma_torch import *
# from huggingface_hub import notebook_login
# notebook_login()

# gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", cache_dir = "model/tokenizer")
gemma_tokenizer = Tokenizer("model/tokenizer.model")

input_text = "Write me a poem about Machine Learning."
result = gemma_tokenizer.encode(input_text)
print(result)
result = gemma_tokenizer.decode(result)
print(result)


if __name__ == '__main__':
    print('')