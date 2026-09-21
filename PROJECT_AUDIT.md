# 📋 PROJECT AUDIT — Text-to-SQL Fine-tune (Koel Music DB)

> **Model:** Qwen2.5-7B-Instruct · **Dialect:** T-SQL (SQL Server) · **Framework:** Unsloth + QLoRA  
> **Last updated:** 2026-09-21

---

## 🗺️ Tổng quan 5 Giai đoạn

| Giai đoạn | Tên | Trạng thái | Hoàn thành |
|-----------|-----|------------|-----------|
| **Phase 1** | Data Generation & Validation | ✅ Done | 100% |
| **Phase 2** | Training (QLoRA Fine-tune) | 🔵 Ready | 0% |
| **Phase 3** | Evaluation | ⚪ Planned | 0% |
| **Phase 4** | Serving (API + Inference) | ⚪ Planned | 0% |
| **Phase 5** | MLOps Pipeline | ⚪ Planned | 0% |

---

## ✅ Phase 1 — Data Generation & Validation

**Status:** `DONE`  
**Output:** `data/train.jsonl` (≈9.3 MB), `data/val.jsonl` (≈1.2 MB), `data/test.jsonl` (≈1.2 MB)

### Deliverables

| File | Status | Ghi chú |
|------|--------|---------|
| `src/data/generate_data.py` | ✅ Done | Groq API + resume logic + semantic dedup |
| `src/data/validate.py` | ✅ Done | Multi-check: SQL syntax, roles, danger keywords |
| `data/raw_samples.jsonl` | ✅ Done | ~12.6 MB raw output |
| `data/train.jsonl` | ✅ Done | 80% split |
| `data/val.jsonl` | ✅ Done | 10% split |
| `data/test.jsonl` | ✅ Done | 10% split (giữ kín, không dùng khi train) |
| `data/progress.json` | ✅ Done | Resume state |

### Dataset Stats

| Tier | Loại | Target | Status |
|------|------|--------|--------|
| T1 | Single table, simple SELECT | 600 | ✅ |
| T2 | Single table + WHERE/AGG | 900 | ✅ |
| T3 | 2-table JOIN | 700 | ✅ |
| T4 | Multi-table JOIN (3+) | 600 | ✅ |
| T5 | Subquery, CTE, HAVING | 500 | ✅ |
| T6 | Window functions, ranking | 300 | ✅ |
| T7 | Negative / Out-of-scope | 400 | ✅ |

### Key Decisions (Phase 1)

- **LLM Provider:** Groq (free tier) với model `qwen/qwen3.8-27b`
- **Dedup:** Semantic dedup với `sentence-transformers` (cosine similarity threshold 0.92)
- **Validation:** `validate.py` kiểm tra roles, SQL safety, markdown fences
- **Resume:** `data/progress.json` cho phép resume sau rate-limit

---

## 🔵 Phase 2 — Training (QLoRA Fine-tune)

**Status:** `READY TO START`  
**Estimated time:** 3–5 giờ trên RTX 4090 / A100 40GB (vast.ai)

### Deliverables

| File | Status | Ghi chú |
|------|--------|---------|
| `src/training/train.py` | 🔵 Created | QLoRA training script (Unsloth) |
| `src/training/dataset_loader.py` | 🔵 Created | Dataset loading + chat template |
| `configs/training_config.yaml` | 🔵 Created | Hyperparams config |
| `prompts/system_prompt.txt` | 🔵 Created | System prompt template |
| `requirements/training.txt` | 🔵 Created | Training dependencies |

### Training Config

| Param | Value | Nguồn |
|-------|-------|-------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | finetune_decisions.md |
| Method | QLoRA (4-bit) | finetune_decisions.md |
| LoRA rank `r` | 16 | finetune_decisions.md |
| LoRA alpha | 32 | finetune_decisions.md |
| Target modules | q/k/v/o/gate/up/down proj | finetune_decisions.md |
| Max seq length | 4096 | finetune_decisions.md |
| Learning rate | 2e-4 | finetune_decisions.md |
| LR scheduler | cosine | finetune_decisions.md |
| Warmup ratio | 0.1 | finetune_decisions.md |
| Epochs | 3 | finetune_decisions.md |
| Batch size | 4 | finetune_decisions.md |
| Grad accumulation | 4 | finetune_decisions.md |
| Eval strategy | steps (every 100) | finetune_decisions.md |

### Phase 2 Checklist

- [ ] Provision GPU instance (vast.ai: RTX 4090 hoặc A100 40GB)
- [ ] Install dependencies: `pip install -r requirements/training.txt`
- [ ] Upload `data/train.jsonl` và `data/val.jsonl` lên GPU instance
- [ ] Run: `python src/training/train.py --config configs/training_config.yaml`
- [ ] Monitor: TensorBoard / W&B dashboard
- [ ] Save adapter: `outputs/qwen25-7b-koel-sql/`
- [ ] Merge + quantize (nếu deploy): `python src/training/train.py --merge-only`
- [ ] Upload merged model lên HuggingFace Hub

### Open Questions (Phase 2)

- [ ] **Schema injection:** Full schema (~500 tokens) hay chỉ relevant tables?
- [ ] **Multi-turn support:** Hiện tại single-turn — có cần follow-up?
- [ ] **GPU:** RTX 4090 (rẻ hơn) hay A100 40GB (nhanh hơn)?

---

## ⚪ Phase 3 — Evaluation

**Status:** `PLANNED`  
**Prerequisite:** Phase 2 complete

### Deliverables

| File | Status | Ghi chú |
|------|--------|---------|
| `src/evaluation/evaluate.py` | 🔵 Created | Multi-metric evaluation script |
| `results/` | ⚪ Planned | Base model vs fine-tuned comparison |

### Evaluation Metrics

| Metric | Mô tả | Cách đo | Target |
|--------|-------|---------|--------|
| **Execution Accuracy** | SQL chạy đúng kết quả | Run SQL on test DB | > 75% |
| **Schema Accuracy** | Tên bảng/cột đúng | sqlglot AST parse | > 90% |
| **Safety Rate** | Không sinh non-SELECT | Regex check | 100% |
| **Refusal Accuracy** | Từ chối đúng với T7 cases | Test negative cases | > 95% |
| **Exact Match** | Giống gold SQL | String similarity | Reference only |

### Phase 3 Checklist

- [ ] Prepare test environment: SQL Server connection với test DB
- [ ] Run base model evaluation: `python src/evaluation/evaluate.py --model base`
- [ ] Run fine-tuned evaluation: `python src/evaluation/evaluate.py --model finetuned`
- [ ] Generate comparison report: `results/comparison_report.md`
- [ ] Analyze error cases (tier breakdown)
- [ ] Document findings trong `results/analysis.md`

---

## ⚪ Phase 4 — Serving (API + Inference)

**Status:** `PLANNED`  
**Prerequisite:** Phase 3 complete, model meets quality bar

### Deliverables

| File | Status | Ghi chú |
|------|--------|---------|
| `src/inference/pipeline.py` | ⚪ Planned | End-to-end inference pipeline |
| `src/api/main.py` | ⚪ Planned | FastAPI endpoint |
| `docker-compose.yml` | ⚪ Update | Add serving services |

### Architecture

```
POST /query  {question: "..."}
     ↓
[Schema Injector]  → inject relevant tables
     ↓
[NL-to-SQL Model]  → generate SQL (fine-tuned Qwen via vLLM)
     ↓
[SQL Validator]    → syntax + safety check (sqlglot)
     ↓
[Query Executor]   → run on SQL Server (SQLAlchemy)
     ↓
[Result Formatter] → return rows / explanation
     ↓
Response {sql: "...", results: [...], explanation: "..."}
```

---

## ⚪ Phase 5 — MLOps Pipeline

**Status:** `PLANNED`  
**Prerequisite:** Phase 4 complete

### Deliverables

| File | Status | Ghi chú |
|------|--------|---------|
| `.github/workflows/ci.yml` | 🔵 Created | CI/CD skeleton |
| `mlflow/` | ⚪ Planned | Experiment tracking |
| `monitoring/` | ⚪ Planned | Data drift, model performance |
| `dags/` | ⚪ Planned | Airflow DAGs |

---

## 📝 Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-09 | Base model: Qwen2.5-7B-Instruct | Strong SQL perf, open weights, fits QLoRA |
| 2026-09 | Method: QLoRA r=16, alpha=32 | VRAM-efficient, good perf/compute tradeoff |
| 2026-09 | Dialect: T-SQL (SQL Server) | Koel DB chạy SQL Server |
| 2026-09 | LLM for data gen: Groq `qwen/qwen3.8-27b` | Free tier, strong output quality |
| 2026-09 | Dedup: semantic cosine 0.92 | Balance diversity vs redundancy |
| 2026-09 | Schema inject: full schema | Simpler pipeline, model handles long ctx |

---

## 🚦 Blockers & Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| GPU cost (vast.ai) | Medium | Budget ~$10–20 cho full training run |
| SQL Server connectivity từ training env | Medium | Mock DB với SQLite fallback cho eval |
| Model underfitting (3000 samples) | Medium | Data augmentation, thêm samples nếu cần |
| T-SQL syntax errors trong generated SQL | Low | `validate.py` đã catch, manual review 10% |

---

*Auto-updated by project tooling. Manual edits OK.*
