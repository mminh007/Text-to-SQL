# Text-to-SQL Fine-Tuning (Koel Music DB)

Dự án này tập trung vào việc fine-tune mô hình LLM (cụ thể là **Qwen2.5-7B-Instruct**) để hiểu và dịch ngôn ngữ tự nhiên (tiếng Việt) thành truy vấn **T-SQL (SQL Server)** phục vụ cho schema của Koel Music Streaming.

## Cấu trúc thư mục

- `configs/`: File cấu hình YAML cho việc training và evaluation
- `data/`: Dữ liệu sinh ra (raw, train, val, test)
- `docs/`: Tài liệu dự án (planning, decisions, briefs)
- `notebooks/`: Các Jupyter notebooks phục vụ EDA và testing
- `prompts/`: Chứa các prompt cố định, VD: system_prompt
- `requirements/`: Các dependency files tách biệt theo môi trường
- `results/`: Output đánh giá, metrics report
- `src/`: Mã nguồn chính của dự án
  - `data/`: Mã nguồn sinh và kiểm tra dữ liệu
  - `training/`: Mã nguồn phục vụ fine-tuning mô hình (Unsloth, QLoRA)
  - `evaluation/`: Mã nguồn đánh giá mô hình với Base vs Fine-tuned
  - `api/` & `inference/`: Mã nguồn phục vụ serving

## Các giai đoạn dự án
Vui lòng tham khảo file `PROJECT_AUDIT.md` để xem tiến độ chi tiết.
1. **Giai đoạn 1**: Data Generation & Validation (Hoàn thành)
2. **Giai đoạn 2**: Training (Đang thực hiện)
3. **Giai đoạn 3**: Evaluation
4. **Giai đoạn 4**: Serving API & Inference Pipeline
5. **Giai đoạn 5**: MLOps & CI/CD

## Hướng dẫn cài đặt
Sử dụng các file trong thư mục `requirements/` để cài đặt môi trường tương ứng:
```bash
pip install -r requirements/base.txt
pip install -r requirements/training.txt
```
