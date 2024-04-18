import os
import wandb
import torch
from trl import SFTTrainer
from trl import DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from peft import AutoPeftModelForCausalLM
from peft import get_peft_model
from peft import prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments
from huggingface_hub import notebook_login


ModelConfig = {
    "device": "auto",
    "model": "meta-llama/Llama-2-7b-hf",
    "data_name": "heegyu/open-korean-instructions",
    "split": 10
}

LoRAConfig = {
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "r": 64,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

PrecisionConfig = {
    "load_4bit": True,
    "use_double_quant": True,
    "quant_type": "nf4",
    "compute_dtype": "float16",
}

TrainingConfig = {
    "epoch": 3,
    "batch_size": 4,
    "accumulation_step": 2,
    "checkpointing": True,
    "optim": "paged_adamw_32bit",
    "logging_step": 5,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "group_by_length": False,
    "lr_scheduler_type": "cosine",
    "disable_tqdm": True,
    "report": "wandb",
}


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
        self.device_map = ModelConfig["device"]
        self.model_name = ModelConfig["model"]
        self.finetuned_model_name = f"{self.model_name}-finetuned"
        self.data_name  = ModelConfig["data_name"]
        self.split      = ModelConfig["split"]
        if self.split >= 100:
            self.split = 90

        # LoRA 하이퍼파라미터 설정
        self.config_LoRA = LoraConfig(                                   
            lora_alpha = LoRAConfig["lora_alpha"],
            lora_dropout = LoRAConfig["lora_dropout"],
            r = LoRAConfig["r"],
            bias = LoRAConfig["none"],
            task_type = LoRAConfig["CAUSAL_LM"])

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
            save_strategy=TrainingConfig["epoch"],
            learning_rate=TrainingConfig["learning_rate"],
            weight_decay=TrainingConfig["weight_decay"],
            max_grad_norm=TrainingConfig["max_grad_norm"],
            warmup_ratio=TrainingConfig["warmup_ratio"],
            group_by_length=TrainingConfig["group_by_length"],
            lr_scheduler_type=TrainingConfig["lr_scheduler_type"],
            disable_tqdm=TrainingConfig["disable_tqdm"],
            report_to=TrainingConfig["report"],
            seed=42)

        # 허깅페이스, Wandb 로그인
        self.login_huggingface()
        self.login_wandb()


    def login_huggingface(self):
        notebook_login()


    def login_wandb(self):
        wandb.login()


    def load_data(self):
        self.dataset = load_dataset(self.data_name, split = "train[:" + str(self.split) + "%]")


    def load_model(self):
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config = self.config_BnB,
            use_cache = False,
            device_map = self.device_map)
        self.base_model.config.pretrainig_tp = 1
        self.base_model.gradient_checkpointing_enable()
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        self.peft_model = get_peft_model(self.base_model, self.config_LoRA)


    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code = True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"


    def train(self):
        # 데이터셋, 모델, 토크나이저 로드
        self.load_data()
        self.load_model()
        self.load_tokenizer()

        trainer = SFTTrainer(
            model=self.peft_model,
            train_dataset=self.dataset,
            dataset_text_field='text',
            max_seq_length=min(self.tokenizer.model_max_length, 2048),
            tokenizer=self.tokenizer,
            packing=True,
            args=self.config_Training)
        trainer.train()
        trainer.save_model()




class Llama2(LLMClass):
    def __init__(self):
        super(LLMClass, self).__init__()
        # 캐시 폴더 생성
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        os.environ['HF_HOME'] = self.cache_dir

        """
        meta의 Llama2는 권한을 얻어야 한다.
        아래 링크에서 권한 신청에 대한 동의를 할 수 있다.
        https://huggingface.co/meta-llama/Llama-2-7b-hf
        """                             
        # 허깅페이스 모델, 데이터셋 경로 및 파인튜닝 모델 지정
        self.device_map = ModelConfig["device"]
        self.model_name = ModelConfig["model"]
        self.finetuned_model_name = f"{self.model_name}-finetuned"
        self.data_name  = ModelConfig["data_name"]
        self.split      = ModelConfig["split"]
        if self.split >= 100:
            self.split = 90

        # LoRA 하이퍼파라미터 설정
        self.config_LoRA = LoraConfig(                                   
            lora_alpha = LoRAConfig["lora_alpha"],
            lora_dropout = LoRAConfig["lora_dropout"],
            r = LoRAConfig["r"],
            bias = LoRAConfig["none"],
            task_type = LoRAConfig["CAUSAL_LM"])

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
            save_strategy=TrainingConfig["epoch"],
            learning_rate=TrainingConfig["learning_rate"],
            weight_decay=TrainingConfig["weight_decay"],
            max_grad_norm=TrainingConfig["max_grad_norm"],
            warmup_ratio=TrainingConfig["warmup_ratio"],
            group_by_length=TrainingConfig["group_by_length"],
            lr_scheduler_type=TrainingConfig["lr_scheduler_type"],
            disable_tqdm=TrainingConfig["disable_tqdm"],
            report_to=TrainingConfig["report"],
            seed=42)

        # 허깅페이스, Wandb 로그인
        self.login_huggingface()
        self.login_wandb()


    def login_huggingface(self):
        notebook_login()


    def login_wandb(self):
        wandb.login()