## PyTorch2MLX
Objective: Convert the Hugging Face satetensor model into an MLX implementation, and develop a library for inference on Apple Silicon.  
Future Directions:  
1. Training of LLMs based on Hugging Face (limited to Gemme-2B, Llama3-8B).  
2. Develop a library for converting PyTorch models to MLX models.  
3. Understand methods to optimize and accelerate MLX LLM models for inference.

## To Do
~~0. 완료한 것: class MLXGemmaMLP 레벨까지 연산 결과 일치한 것을 확인~~
1. source/gemma/model_mlx의 class MLXGemmaAttention을 MLX 형태로 변환 (현재 이 부분만 PyTorch 코드)
3. safetensor 모듈을 불러와서 torch weight -> MLX weight로 이식 코드 구현
4. run_gemma.py를 MLX 모델을 빌드하고 weight를 심어다가 추론하는 것을 실행하는 코드로 작성

## Requirements
```
bash requirements.sh
```
## Directory
```
PyTorch2MLX
  ㄴmodel
    ㄴgemma-2b-it
      ㄴgemma-2b-it.ckpt
      ㄴmodel-00001-of-00002.safetensors
      ㄴmodel-00002-of-00002.safetensors
      ㄴtokenizer.model
      ㄴ... ...
    ㄴllama3-8b
  ㄴsource
    ㄴgemma
      ㄴconfig.py
      ㄴmodel_mlx.py
      ㄴmodel.py
      ㄴtokenizer.py
  requirements.sh
```
## Usage  
Coming soon
