from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from huggingface_hub import notebook_login
notebook_login()

gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
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


if __name__ == '__main__':
    print('')