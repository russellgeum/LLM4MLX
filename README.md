## PyTorch2MLX
Objective: Convert the Hugging Face satetensor model into an MLX implementation, and develop a library for inference on Apple Silicon.  
Future Directions:  
1. Training of LLMs based on Hugging Face (limited to Gemme-2B, Llama3-8B).  
2. Develop a library for converting PyTorch models to MLX models.  
3. Understand methods to optimize and accelerate MLX LLM models for inference.

## To Do
1. GemmaModel, MLXGemmaModel이 같은 것을 확인 (완료)  
   torch safetensor -> MLX safetensor와 torch ckpt -> MLX safetensor의 MLX 모델 추론이 동일 (당연히 같아야 함)
   torch safetensor는 q, k, v가 분리되어 있음 이를 qkv로 묶으면 torch ckpt의 추론이 동일함 (당연히 같아야 함)
2. Sampler와 디코딩 단계 구현하여 붙이기 (진행 중)  
3. Tokenizer 붙이기
4. weight를 옮기는 것은 test_model.py에서 가구현

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

## Reference
- [Google Gemma Official](https://github.com/google/gemma_pytorch)
- [HuggingFace gemma-1.1-2b-it](https://huggingface.co/google/gemma-1.1-2b-it)
- [HuggingFace gemma-1.1-2b-it-pytorch](https://huggingface.co/google/gemma-1.1-2b-it-pytorch)