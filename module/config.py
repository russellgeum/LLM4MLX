Llama_ModelConfig = {
    "device_map": "auto",
    "model_name": "meta-llama/Llama-2-7b-hf",
    "data_name": "heegyu/open-korean-instructions",
    "split": 100
    }

Llama_LoRAConfig = {
    "r": 64,
    "bias": "none",
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "task_type": "CAUSAL_LM",
    }

Llama_PrecisionConfig = {
    "load_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    }

Llama_TrainingConfig = {
    "save_strategy": "epoch",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_32bit",
    "logging_step": 5,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "group_by_length": False,
    "lr_scheduler_type": "cosine",
    "disable_tqdm": False,
    "report_to": "wandb",
    }


Gemma_ModelConfig = {
    "device_map": "auto",
    "model_name": "google/gemma-2b-it",
    "data_name": "heegyu/open-korean-instructions",
    "split": 100
    }


Gemma_LoRAConfig = {
    "r": 64,
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM"
    }


Gemma_PrecisionConfig = {
    "load_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    }


Gemma_TrainingConfig = {
    "save_strategy": "epoch",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_32bit",
    "logging_step": 500,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "group_by_length": False,
    "lr_scheduler_type": "cosine",
    "disable_tqdm": True,
    "report_to": "wandb",
    }