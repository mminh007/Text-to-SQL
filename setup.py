"""
setup.py
========
Cho phép cài package ở chế độ editable:
    pip install -e .

Sau khi cài, Python sẽ nhận ra `src.*` là package hợp lệ
dù chạy từ bất kỳ thư mục nào.

Ví dụ chạy training sau khi cài:
    python -m src.training.train --config configs/training_config.yaml
"""

from setuptools import setup, find_packages

setup(
    name="text-to-sql-finetune",
    version="0.1.0",
    packages=find_packages(),   # tự tìm tất cả thư mục có __init__.py
    python_requires=">=3.10",
)
