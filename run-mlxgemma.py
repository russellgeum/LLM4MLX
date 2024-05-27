import time
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
from source.config import *
from source.tokenizer import *
from source.gemma_mlx import *
from source.gemma_torch import *


def main():
    model = MLXGemmaForCausalLM(get_config_for_2b())
    model.eval()
    result = model.generate("The meaning of life is", output_len=100)
    print(result)

if __name__ == "__main__":
    main()
