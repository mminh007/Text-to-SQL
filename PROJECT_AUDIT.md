# Project Audit: Koel Text-to-SQL Finetuning

## 🎯 Mục tiêu dự án
Fine-tune mô hình **Qwen2.5-7B-Instruct** để dịch câu hỏi tiếng Việt sang câu truy vấn **T-SQL** dựa trên schema của hệ thống Koel Music Streaming.

## 📈 Tiến độ các giai đoạn (Phases)

### Giai đoạn 1: Data Generation & Validation (✅ Hoàn thành)
- [x] Thiết kế compact schema (25 bảng)
- [x] Lên danh sách các Tiers (1-7) độ phức tạp
- [x] Tạo `generate_data.py` pipeline (có rate limit handling, semantic deduplication, SQL execution validation)
- [x] Tạo `validate.py` để check dataset quality
- [x] Sinh 5000+ samples
- [x] Split train/val/test set (80/10/10)

### Giai đoạn 2: Training (⏳ Đang tiến hành)
- [x] Quyết định cấu hình Training (Unsloth, QLoRA, r=16, alpha=32, seq_len=4096)
- [x] Refactor project structure, đưa code vào thư mục `src/training/`
- [ ] Implement `dataset_loader.py` (chuyển đổi JSONL thành format HuggingFace / Unsloth)
- [ ] Implement `train.py` (sử dụng SFTTrainer, cấu hình training args, PEFT)
- [ ] Chạy smoke test training (với subset 10-50 mẫu)
- [ ] Fine-tune full dataset (trên GPU/Vast.ai)
- [ ] Lưu & xuất model adapter

### Giai đoạn 3: Evaluation (📝 Đã lên kế hoạch)
- [ ] Viết `evaluate.py`
- [ ] Implement Execution Accuracy metric (đánh giá qua kết nối ODBC T-SQL)
- [ ] Implement Schema Accuracy (dùng `sqlglot`)
- [ ] Đánh giá Base model vs Fine-tuned model trên Test set
- [ ] Tổng hợp report kết quả vào `results/`

### Giai đoạn 4: Serving & Inference (Thì tương lai)
- [ ] Tạo API backend (FastAPI) trong `src/api/`
- [ ] Tạo module `pipeline.py` cho NL-to-SQL logic, RAG (schema selection)
- [ ] Validate đầu ra trước khi query vào DB

### Giai đoạn 5: MLOps Pipeline (Thì tương lai)
- [ ] CI/CD (GitHub Actions)
- [ ] MLflow experiment tracking
- [ ] Monitoring cho data drift & model performance

---

## 📝 Nhật ký quyết định (Decisions Log)
- **2026-09-17:** Quyết định dùng Unsloth QLoRA thay vì full finetune để tối ưu chi phí và tránh catastrophic forgetting. Tăng context length lên 4096 do schema lớn. Tham khảo thêm tại `docs/finetune_decisions.md`.
- **2026-09-21:** Tiến hành refactor cấu trúc thư mục, chuyển từ mô hình script đơn lẻ sang structure bài bản.
