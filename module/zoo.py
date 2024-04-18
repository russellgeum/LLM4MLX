import os
import logging
import wandb
import torch
from trl import SFTTrainer
# from trl import DataCollatorForCompletionOnlyLM

from .config import *
from peft import LoraConfig
from peft import PeftModel
# from peft import get_peft_model
from peft import prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments
from huggingface_hub import notebook_login



class LLMClass():
    def __init__(self, 
            cache_dir: str,
            ModelConfig: dict,
            LoRAConfig: dict,
            PrecisionConfig: dict,
            TrainingConfig: dict):
        """
        ModelConfig
        LoRAConfig
        PrecisionConfig
        TrainingConfig
        """
        # 캐시 폴더 생성
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        os.environ['HF_HOME'] = self.cache_dir

        # 허깅페이스 모델, 데이터셋 경로 및 파인튜닝 모델 지정
        self.device_map = ModelConfig["device_map"]
        self.model_name = ModelConfig["model_name"]
        self.finetuned_model_name = f"{self.model_name}-finetuned"
        self.data_name  = ModelConfig["data_name"]
        self.split      = ModelConfig["split"]
        if self.split > 100:
            raise ValueError("The value of split must be 100 or less.")
        
        # LoRA 하이퍼파라미터 설정
        self.config_LoRA = LoraConfig(                
            r = LoRAConfig["r"],
            bias = LoRAConfig["bias"],
            lora_alpha = LoRAConfig["lora_alpha"],
            lora_dropout = LoRAConfig["lora_dropout"],
            task_type = LoRAConfig["task_type"])

        # BitsAndBytesConfig Mixed Precision으로 학습
        self.config_BnB = BitsAndBytesConfig(
            load_in_4bit = PrecisionConfig["load_4bit"],
            bnb_4bit_use_double_quant = PrecisionConfig["use_double_quant"],
            bnb_4bit_quant_type = PrecisionConfig["quant_type"],
            bnb_4bit_compute_dtype = PrecisionConfig["compute_dtype"])

        # 학습 하이퍼파라미터 설정
        self.config_Training = TrainingArguments(
            output_dir=self.finetuned_model_name,
            num_train_epochs=TrainingConfig["epoch"],
            per_device_train_batch_size=TrainingConfig["batch_size"],
            gradient_accumulation_steps=TrainingConfig["accumulation_step"],
            gradient_checkpointing=TrainingConfig["checkpointing"],
            optim=TrainingConfig["optim"],
            logging_steps=TrainingConfig["logging_step"],
            save_strategy=TrainingConfig["save"],
            learning_rate=TrainingConfig["learning_rate"],
            weight_decay=TrainingConfig["weight_decay"],
            max_grad_norm=TrainingConfig["max_grad_norm"],
            warmup_ratio=TrainingConfig["warmup_ratio"],
            group_by_length=TrainingConfig["group_by_length"],
            lr_scheduler_type=TrainingConfig["lr_scheduler_type"],
            disable_tqdm=TrainingConfig["disable_tqdm"],
            report_to=TrainingConfig["report"],
            seed=42)

        # 학습 트레이너 설정을 위한 데이터, 모델, 토크나이저 업로드
        self.load_dataset()
        self.load_tokenizer()
        self.load_model()

        # 학습 트레이너 설정
        self.trainer = SFTTrainer(
            model=self.base_model,
            train_dataset=self.dataset,
            dataset_text_field='text',
            max_seq_length=min(self.tokenizer.model_max_length, 2048),
            tokenizer=self.tokenizer,
            packing=True,
            args=self.config_Training,
            peft_config=self.config_LoRA)

        # 허깅페이스, Wandb 로그인
        self.login_huggingface()
        self.login_wandb()


    def login_huggingface(self):
        notebook_login()


    def login_wandb(self):
        wandb.login()


    def load_dataset(self):
        self.dataset = load_dataset(self.data_name, split = "train[:" + str(self.split) + "%]")


    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code = True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"


    def load_model(self):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config = self.config_BnB,
            use_cache = False,
            device_map = self.device_map)
        self.base_model.config.pretrainig_tp = 1
        self.base_model.gradient_checkpointing_enable()
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        # self.peft_model = get_peft_model(self.base_model, self.config_LoRA)


    def run(self):
        print("----------- Model Training Start")
        self.trainer.train()

    def save(self):
        self.trainer.save_model()



class Llama2_7B(LLMClass):
    def __init__(self,
            cache_dir: str,
            ModelConfig: dict,
            LoRAConfig: dict,
            PrecisionConfig: dict,
            TrainingConfig: dict):
        super(LLMClass, self).__init__()
        # 캐시 폴더 생성
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        os.environ['HF_HOME'] = self.cache_dir

        """
        meta의 Llama2는 권한을 얻어야 한다. 아래 링크에서 권한에 대한 동의를 할 수 있다.
        https://huggingface.co/meta-llama/Llama-2-7b-hf
        """                             
        # 허깅페이스 모델, 데이터셋 경로 및 파인튜닝 모델 지정
        self.device_map = 'auto'
        self.model_name = 'meta-llama/Llama-2-7b-hf'
        self.finetuned_model_name = 'llama2-7b/fintuned'
        self.data_name  = ModelConfig["data_name"]
        self.split      = ModelConfig["split"]
        if self.split > 100:
            raise ValueError("The value of split must be 100 or less.")

        # LoRA 하이퍼파라미터 설정
        self.config_LoRA = LoraConfig(
            r = LoRAConfig["r"],
            lora_alpha = LoRAConfig["lora_alpha"],
            lora_dropout = LoRAConfig["lora_dropout"],
            target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type = LoRAConfig["task_type"])

        # BitsAndBytesConfig Mixed Precision으로 학습
        self.config_BnB = BitsAndBytesConfig(
            load_in_4bit = PrecisionConfig["load_4bit"],
            bnb_4bit_use_double_quant = PrecisionConfig["use_double_quant"],
            bnb_4bit_quant_type = PrecisionConfig["quant_type"],
            bnb_4bit_compute_dtype = PrecisionConfig["compute_dtype"])

        # 학습 하이퍼파라미터 설정
        self.config_Training = TrainingArguments(
            output_dir=self.finetuned_model_name,
            num_train_epochs=TrainingConfig["epoch"],
            per_device_train_batch_size=TrainingConfig["batch_size"],
            gradient_accumulation_steps=TrainingConfig["accumulation_step"],
            gradient_checkpointing=TrainingConfig["checkpointing"],
            optim=TrainingConfig["optim"],
            logging_steps=TrainingConfig["logging_step"],
            save_strategy=TrainingConfig["save"],
            learning_rate=TrainingConfig["learning_rate"],
            weight_decay=TrainingConfig["weight_decay"],
            max_grad_norm=TrainingConfig["max_grad_norm"],
            warmup_ratio=TrainingConfig["warmup_ratio"],
            group_by_length=TrainingConfig["group_by_length"],
            lr_scheduler_type=TrainingConfig["lr_scheduler_type"],
            disable_tqdm=TrainingConfig["disable_tqdm"],
            report_to=TrainingConfig["report"],
            seed=42)

        # 학습 트레이너 설정을 위한 데이터, 모델, 토크나이저 업로드
        self.load_dataset()
        self.load_tokenizer()
        self.load_model()

        # 학습 트레이너 설정
        self.trainer = SFTTrainer(
            model=self.base_model,
            train_dataset=self.dataset,
            dataset_text_field='text',
            max_seq_length=min(self.tokenizer.model_max_length, 2048),
            tokenizer=self.tokenizer,
            packing=True,
            args=self.config_Training,
            peft_config=self.config_LoRA)

        # 허깅페이스, Wandb 로그인
        self.login_huggingface()
        self.login_wandb()



class Gemma_2B(LLMClass):
    def __init__(self,
            cache_dir: str,
            ModelConfig: dict,
            LoRAConfig: dict,
            PrecisionConfig: dict,
            TrainingConfig: dict):
        super(LLMClass, self).__init__()

        """
        Google의 Gemma는 권한을 얻어야 한다. 아래 링크에서 권한에 대한 동의를 할 수 있다.
        https://huggingface.co/google/gemma-2b-it
        """       
        # 캐시 폴더 생성
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        os.environ['HF_HOME'] = self.cache_dir

        # 허깅페이스 모델, 데이터셋 경로 및 파인튜닝 모델 지정
        self.device_map = 'auto'
        self.model_name = "google/gemma-2b-it"
        # self.model_name = "google/gemma-1.1-2b-it"
        self.finetuned_model_name = 'gemma-2b/fintuned'
        self.data_name  = ModelConfig["data_name"]
        self.split      = ModelConfig["split"]
        if self.split > 100:
            raise ValueError("The value of split must be 100 or less.")

        # LoRA 하이퍼파라미터 설정
        """
        r = LoRA의 rank 차원
        load_alpha:     LoRA의 스케일링 팩터
        lora_dropout:   LoRA의 드롭아웃 비율
        target_modules: LoRA를 적용할 레이어 층
        task_type:      LLM 태스크 타입을 결정
        """
        self.config_LoRA = LoraConfig(                                   
            r = Gemma_LoRAConfig["r"],
            lora_alpha = Gemma_LoRAConfig["lora_alpha"],
            lora_dropout = Gemma_LoRAConfig["lora_dropout"],
            target_modules = Gemma_LoRAConfig["target_modules"],
            task_type = Gemma_LoRAConfig["task_type"]
            )

        # BitsAndBytesConfig Mixed Precision으로 학습
        self.config_BnB = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
            )

        # 학습 하이퍼파라미터 설정
        self.config_Training = TrainingArguments(
            output_dir=self.finetuned_model_name,
            num_train_epochs=TrainingConfig["num_train_epochs"],
            per_device_train_batch_size=TrainingConfig["per_device_train_batch_size"],
            gradient_accumulation_steps=TrainingConfig["gradient_accumulation_steps"],
            gradient_checkpointing=TrainingConfig["gradient_checkpointing"],
            optim=TrainingConfig["optim"],
            logging_steps=TrainingConfig["logging_step"],
            save_strategy=TrainingConfig["save_strategy"],
            learning_rate=TrainingConfig["learning_rate"],
            weight_decay=TrainingConfig["weight_decay"],
            max_grad_norm=TrainingConfig["max_grad_norm"],
            warmup_ratio=TrainingConfig["warmup_ratio"],
            group_by_length=TrainingConfig["group_by_length"],
            lr_scheduler_type=TrainingConfig["lr_scheduler_type"],
            disable_tqdm=TrainingConfig["disable_tqdm"],
            report_to=TrainingConfig["report_to"],
            seed=42
            )

        # 학습 트레이너 설정을 위한 데이터, 모델, 토크나이저 업로드
        self.load_dataset()
        self.load_tokenizer()
        self.load_model()

        # 학습 트레이너 설정
        self.trainer = SFTTrainer(
            model=self.base_model,
            train_dataset=self.dataset,
            max_seq_length=512,
            args = self.config_Training,
            peft_config=self.config_LoRA,
            formatting_func = self.convert_prompt
            )

        # 허깅페이스, Wandb 로그인
        self.login_huggingface()
        self.login_wandb()


    # open-korean-instructions 데이터셋을 Gemma에 맞게 시스템프롬프트를 만들어주는 함수
    # def convert_prompt(self, prompt):
    #     converted_prompt = []
    #     for idx, data in enumerate(prompt["text"]):
    #         print(idx, data)
    #         # 입력 문자열에서 <user>와 <bot> 사이의 텍스트를 추출
    #         user_text = data.split('<usr>')[1].split('<bot>')[0].strip()
    #         bot_text  = data.split('<bot>')[1].strip()

    #         # 새로운 형식으로 문자열 구성
    #         new_format = f"<bos><start_of_turn>user {user_text} <end_of_turn> <start_of_turn>model {bot_text} <end_of_turn><eos>"
    #         converted_prompt.append(new_format)
    #     return converted_prompt
    def convert_prompt(self, prompt):
        converted_prompt = []
        for idx, data in enumerate(prompt["text"]):
            # <sys> 태그로 분리하고, <sys> 태그 이후의 내용만 사용
            if '<sys>' in data:
                data = data.split('<sys>')[1]

            # 결과를 저장할 리스트 초기화
            result_parts = []
            # <usr>와 <bot> 태그 사이의 내용을 반복해서 추출
            user_parts = data.split('<usr>')[1:]  # 첫 번째 <usr> 태그 이후의 모든 부분을 나눔
            for part in user_parts:
                if '<bot>' in part:
                    user_text = part.split('<bot>')[0].strip()
                    model_text = part.split('<bot>')[1].strip()

                    # 새로운 형식으로 문자열 구성 및 결과 리스트에 추가
                    new_format = f"<bos><start_of_turn>user {user_text} <end_of_turn> <start_of_turn>model {model_text} <end_of_turn><eos>"
                    result_parts.append(new_format)

            # 모든 변환된 부분을 하나의 문자열로 결합
            converted_prompt.append(' '.join(result_parts))
        return converted_prompt