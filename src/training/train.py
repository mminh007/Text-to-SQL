"""
src/training/train.py
======================
QLoRA fine-tuning of Qwen2.5-7B-Instruct for Text-to-SQL (Koel Music DB).

Requirements: GPU with ≥16GB VRAM (RTX 4090 24GB or A100 40GB recommended)
Framework: Unsloth + TRL (SFTTrainer) + HuggingFace PEFT

Usage:
    # Full training run:
    python src/training/train.py --config configs/training_config.yaml

    # Quick smoke test (50 samples, 1 epoch):
    python src/training/train.py --config configs/training_config.yaml --smoke-test

    # Merge LoRA adapter into base model (after training):
    python src/training/train.py --config configs/training_config.yaml --merge-only \\
        --adapter-path outputs/qwen25-7b-koel-sql

    # Push merged model to HuggingFace Hub:
    python src/training/train.py --config configs/training_config.yaml --merge-only \\
        --adapter-path outputs/qwen25-7b-koel-sql --push-to-hub

Infrastructure: vast.ai GPU instance
    - RTX 4090 24GB: ~4–5h for 3 epochs on ~4000 samples
    - A100 40GB: ~2–3h for 3 epochs on ~4000 samples
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")

load_dotenv()


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return cfg


# ── Unsloth model loader ───────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    """Load quantized base model + apply LoRA via Unsloth."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error(
            "Unsloth not installed. Run: pip install -r requirements/training.txt"
        )
        sys.exit(1)

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]

    logger.info("Loading base model: %s", model_cfg["name"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg.get("dtype"),
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    logger.info("Applying QLoRA adapter (r=%d, alpha=%d)", lora_cfg["r"], lora_cfg["alpha"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg.get("random_state", 42),
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


# ── Training args builder ──────────────────────────────────────────────────────

def build_training_args(cfg: dict, smoke_test: bool = False):
    """Build TrainingArguments from config."""
    from transformers import TrainingArguments

    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    log_cfg = cfg["logging"]

    return TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=1 if smoke_test else train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        seed=train_cfg["seed"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        # Evaluation & checkpointing
        eval_strategy=eval_cfg["strategy"],
        eval_steps=50 if smoke_test else eval_cfg["eval_steps"],
        save_strategy=eval_cfg["save_strategy"],
        save_steps=50 if smoke_test else eval_cfg["save_steps"],
        save_total_limit=eval_cfg["save_total_limit"],
        load_best_model_at_end=eval_cfg["load_best_model_at_end"],
        metric_for_best_model=eval_cfg["metric_for_best_model"],
        greater_is_better=eval_cfg["greater_is_better"],
        # Logging
        logging_steps=log_cfg["logging_steps"],
        report_to=log_cfg["report_to"],
        run_name=log_cfg["run_name"],
    )


# ── Main training function ─────────────────────────────────────────────────────

def train(cfg: dict, smoke_test: bool = False, resume_from_checkpoint: bool = False):
    """Full QLoRA training pipeline."""
    from trl import SFTTrainer, SFTConfig

    # Add project root to path so we can import src.training.dataset_loader
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.training.dataset_loader import build_dataset, get_tier_distribution

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ── Load datasets ─────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    prompt_file = Path(data_cfg["system_prompt_file"])
    max_samples = 50 if smoke_test else None

    logger.info("Loading training dataset...")
    train_dataset = build_dataset(
        Path(data_cfg["train_file"]),
        tokenizer,
        system_prompt_file=prompt_file,
        max_samples=max_samples,
    )

    logger.info("Loading validation dataset...")
    val_dataset = build_dataset(
        Path(data_cfg["val_file"]),
        tokenizer,
        system_prompt_file=prompt_file,
        max_samples=max_samples,
    )

    # Log tier distribution
    dist = get_tier_distribution(train_dataset)
    logger.info("Train tier distribution: %s", dist)

    # ── Build trainer ─────────────────────────────────────────────────────────
    training_args = build_training_args(cfg, smoke_test=smoke_test)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_num_proc=2,
        packing=data_cfg.get("packing", False),
        args=training_args,
    )

    # ── GPU memory stats ──────────────────────────────────────────────────────
    try:
        import torch
        gpu_stats = torch.cuda.get_device_properties(0)
        reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        logger.info(
            "GPU: %s | VRAM: %d GB | Reserved: %s GB",
            gpu_stats.name, round(gpu_stats.total_memory / 1024**3, 1), reserved,
        )
    except Exception:
        pass

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training (smoke_test=%s, resume=%s)...", smoke_test, resume_from_checkpoint)
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ── Save adapter ──────────────────────────────────────────────────────────
    output_dir = Path(cfg["training"]["output_dir"])
    logger.info("Saving LoRA adapter to %s/adapter/", output_dir)
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")

    logger.info("Training complete. Stats: %s", trainer_stats)
    return trainer_stats


# ── Merge LoRA adapter into base model ────────────────────────────────────────

def merge_adapter(cfg: dict, adapter_path: Path, push_to_hub: bool = False):
    """
    Merge LoRA weights into the base model and save as full model.

    Usage:
        python src/training/train.py --config configs/training_config.yaml \\
            --merge-only --adapter-path outputs/qwen25-7b-koel-sql
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth not installed.")
        sys.exit(1)

    model_cfg = cfg["model"]
    export_cfg = cfg["export"]
    save_method = export_cfg.get("save_method", "merged_16bit")
    output_dir = Path(cfg["training"]["output_dir"]) / "merged"

    logger.info("Loading adapter from %s for merging...", adapter_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path / "adapter"),
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg.get("dtype"),
        load_in_4bit=False,  # load full precision for merge
    )

    logger.info("Saving merged model (%s) to %s", save_method, output_dir)
    model.save_pretrained_merged(output_dir, tokenizer, save_method=save_method)

    if push_to_hub:
        hub_model_id = export_cfg.get("hub_model_id", "")
        if not hub_model_id:
            logger.error("Set export.hub_model_id in config to push to Hub.")
            sys.exit(1)
        # --push-to-hub requires a WRITE token.
        # HF_TOKEN in .env is read-only (used for model downloads).
        # Set HF_TOKEN_WRITE separately at: https://huggingface.co/settings/tokens
        hf_token_write = os.getenv("HF_TOKEN_WRITE", "")
        if not hf_token_write:
            logger.error(
                "HF_TOKEN_WRITE not set. Push-to-Hub requires a write-permission token.\n"
                "  1. Go to https://huggingface.co/settings/tokens\n"
                "  2. Create a token with 'write' permission\n"
                "  3. Add HF_TOKEN_WRITE=hf_xxx to your .env file"
            )
            sys.exit(1)
        logger.info("Pushing to HuggingFace Hub: %s", hub_model_id)
        model.push_to_hub_merged(hub_model_id, tokenizer, save_method=save_method, token=hf_token_write)

    logger.info("Merge complete. Output: %s", output_dir)


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen2.5-7B-Instruct for Text-to-SQL"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training_config.yaml"),
        help="Path to training_config.yaml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in output_dir (useful for Colab)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick smoke test: 50 samples, 1 epoch",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip training, only merge adapter into base model",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Path to saved adapter (for --merge-only)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub (requires HF_TOKEN_WRITE with write permission + hub_model_id in config)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.merge_only:
        adapter_path = args.adapter_path or Path(cfg["training"]["output_dir"])
        merge_adapter(cfg, adapter_path=adapter_path, push_to_hub=args.push_to_hub)
    else:
        train(cfg, smoke_test=args.smoke_test, resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
