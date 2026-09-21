# Text-to-SQL Fine-tune — Koel Music DB

> Fine-tune **Qwen2.5-7B-Instruct** for Vietnamese natural language → T-SQL translation  
> on the **Koel Music Streaming** database schema.

---

## 🗺️ Project Status

See [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md) for full progress tracker.

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Generation & Validation | ✅ Done |
| 2 | QLoRA Fine-tuning (Unsloth) | 🔵 Ready |
| 3 | Evaluation (base vs fine-tuned) | ⚪ Planned |
| 4 | Serving (FastAPI + inference) | ⚪ Planned |
| 5 | MLOps Pipeline | ⚪ Planned |

---

## 📁 Project Structure

```
text-to-sql/
├── PROJECT_AUDIT.md          ← progress tracker
├── README.md                 ← this file
│
├── src/                      ← all Python source
│   ├── data/
│   │   ├── generate_data.py  ← Phase 1: Groq-powered dataset generator
│   │   └── validate.py       ← Phase 1: data quality validator
│   ├── training/
│   │   ├── train.py          ← Phase 2: QLoRA training (Unsloth)
│   │   └── dataset_loader.py ← Phase 2: dataset loading + chat template
│   ├── evaluation/
│   │   └── evaluate.py       ← Phase 3: multi-metric evaluation
│   ├── inference/
│   │   └── pipeline.py       ← Phase 4: end-to-end pipeline (placeholder)
│   └── api/
│       └── main.py           ← Phase 4: FastAPI endpoint (placeholder)
│
├── configs/
│   └── training_config.yaml  ← QLoRA hyperparameters
│
├── prompts/
│   └── system_prompt.txt     ← system prompt template ({schema} injected at runtime)
│
├── data/                     ← generated dataset (gitignored)
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
│
├── results/                  ← evaluation outputs
├── notebooks/                ← EDA, analysis notebooks
├── docs/                     ← planning documents
├── requirements/
│   ├── base.txt              ← shared deps
│   ├── training.txt          ← Phase 2 deps (Unsloth, TRL, etc.)
│   └── serving.txt           ← Phase 4 deps (FastAPI, vLLM, etc.)
│
└── .github/workflows/ci.yml  ← CI/CD
```

---

## 🚀 Quick Start

### Phase 1 — Data (already done)

```bash
# Generate dataset
python src/data/generate_data.py --all

# Validate output
python src/data/validate.py --input data/train.jsonl
```

### Phase 2 — Training

```bash
# On GPU instance (vast.ai):
pip install -r requirements/training.txt

# Smoke test (50 samples, verify pipeline works):
python src/training/train.py --config configs/training_config.yaml --smoke-test

# Full training run:
python src/training/train.py --config configs/training_config.yaml

# Merge LoRA adapter into base model:
python src/training/train.py --config configs/training_config.yaml --merge-only
```

### Phase 3 — Evaluation

```bash
pip install -r requirements/serving.txt

# Evaluate fine-tuned vs base:
python src/evaluation/evaluate.py --compare \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --finetuned-model outputs/qwen25-7b-koel-sql/merged \
    --test-data data/test.jsonl
```

---

## ⚙️ Configuration

All training hyperparameters are in [`configs/training_config.yaml`](configs/training_config.yaml).

Key config:
```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  max_seq_length: 4096
  load_in_4bit: true      # QLoRA

lora:
  r: 16
  alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  num_train_epochs: 3
  learning_rate: 2e-4
  lr_scheduler_type: cosine
```

---

## 📄 Documentation

- [`docs/finetune_plan.md`](docs/finetune_plan.md) — schema analysis, dataset strategy
- [`docs/text_to_sql_brief.md`](docs/text_to_sql_brief.md) — project overview
- [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md) — progress tracker + decision log

---

## 🏗️ Infrastructure

- **Training:** vast.ai — RTX 4090 24GB (~$0.35/hr) or A100 40GB (~$1.5/hr)
- **Estimated time:** 3–5h for 3 epochs on ~4000 samples
- **Framework:** [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) + HuggingFace PEFT
