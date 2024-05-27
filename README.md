## PyTorch2MLX
이 레포지토리는 허깅페이스의 gemma-1.1-2b-it safetensor를 로드하여.  
파이토치 구현체에서 gemma-2b, gemma-7b 모델을 추론하는 코드를 보여준다.  
향후 방향:  
허깅페이스의 safetensor를 mlx.array로 변환하여, MLX 구현체에서 추론한다.  

## 해야할 작업
1. [완료] gemma torch 구현체에 safetensor 로드하여 추론
2. [예정] gemma mlx 구현체에 safetensor를 변환한 mlx.array로 구현하여 추론

## Requirements
```
bash requirements.sh
```

## Directory
```
PyTorch2MLX
  ㄴmodel
    ㄴgemma-1.1-2b-it
      ㄴmodel-00001-of-00002.safetensors
      ㄴmodel-00002-of-00002.safetensors
      ㄴtokenizer.model
      ㄴ... ...
    ㄴgemma-1.1-7b-it
      ㄴmodel-00001-of-00004.safetensors
      ㄴ... ...
      ㄴmodel-00004-of-00004.safetensors
      ㄴtokenizer.model
      ㄴ... ...
  ㄴsource
    ㄴconfig.py
    ㄴgemma_mlx.py
    ㄴgemma_torch.py
    ㄴtokenizer.py
  requirements.sh
```

## Usage  
```
python run-gemma.py
```

## Reference
- [Google Gemma Official](https://github.com/google/gemma_pytorch)
- [HuggingFace Gemma-1.1-2b-it](https://huggingface.co/google/gemma-1.1-2b-it)
- [HuggingFace Gemma-1.1-7b-it](https://huggingface.co/google/gemma-1.1-7b-it)
- [HuggingFace Gemma-1.1-2b-it-pytorch](https://huggingface.co/google/gemma-1.1-2b-it-pytorch)