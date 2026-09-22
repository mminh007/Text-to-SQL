"""
train.py
========
Fine-tune Qwen2.5-7B-Instruct using Unsloth and QLoRA for the Text-to-SQL task.
Reads configuration from configs/training_config.yaml
"""

import os
import yaml
import argparse
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from src.helpers import EpochCheckpointCallback, TrainLogger
from src.training.dataset_loader import prepare_datasets



def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str, log_dir: str = "logs"):
    logger = TrainLogger(name="train", log_dir=log_dir)
    config = load_config(config_path)
    logger.info("Config loaded from: %s", config_path)

    logger.info("🚀 Initializing Unsloth FastLanguageModel...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name_or_path"],
        max_seq_length=config["model"]["max_seq_length"],
        dtype=None,  # auto detect
        load_in_4bit=config["model"]["load_in_4bit"],
    )
    
    # Enable ChatML template for Qwen if not already set by tokenizer
    tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if (loop.last and add_generation_prompt) or not loop.last %}{{ '<|im_end|>' + '\n'}}{% endif %}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    logger.info("🔧 Setting up PEFT (LoRA)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=config["lora"]["target_modules"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=config["training"]["seed"],
    )
    
    logger.info("📂 Loading and formatting datasets...")
    train_ds, val_ds = prepare_datasets(
        train_path=config["data"]["train_path"],
        val_path=config["data"]["val_path"],
        tokenizer=tokenizer
    )
    
    logger.info("   Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    logger.info("⚙️ Configuring TrainingArguments...")
    training_args = TrainingArguments(
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_ratio=config["training"]["warmup_ratio"],
        num_train_epochs=config["training"]["num_train_epochs"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        optim=config["training"]["optim"],
        weight_decay=config["training"]["weight_decay"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"]["save_steps"],
        eval_strategy=config["training"]["eval_strategy"],        
        eval_steps=config["training"]["eval_steps"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        metric_for_best_model=config["training"]["metric_for_best_model"],
        seed=config["training"]["seed"],
        output_dir=config["training"]["output_dir"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="none"  # can change to wandb or mlflow later
    )

    # --- Build optional epoch-based checkpoint callback ---
    callbacks = []
    save_checkpoints = config["training"].get("save_checkpoints", None)
    if save_checkpoints is not None:
        logger.info("📌 Epoch checkpoint callback enabled: saving every %s epoch(s).", save_checkpoints)
        callbacks.append(
            EpochCheckpointCallback(
                model=model,
                tokenizer=tokenizer,
                output_dir=config["training"]["output_dir"],
                save_every_n_epochs=int(save_checkpoints),
            )
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=config["model"]["max_seq_length"],
        dataset_num_proc=2,  # 4 may cause issues on Colab; 2 is safer
        packing=False,  # Can be True for faster training but might truncate
        args=training_args,
        callbacks=callbacks if callbacks else None,
    )

    logger.info("🔥 Starting training...")
    trainer.train()
    
    logger.info("💾 Saving final model...")
    final_output_path = os.path.join(config["training"]["output_dir"], "final_lora")
    model.save_pretrained(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    
    logger.info("✅ Training complete! Model saved to %s", final_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml", help="Path to config file")
    parser.add_argument("--log-dir", default="logs", help="Directory to save log files")
    args = parser.parse_args()
    
    # Handle relative paths properly if run from project root
    main(args.config, args.log_dir)
