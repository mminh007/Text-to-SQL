"""
fix_v2_notebook.py
==================
Rebuilds colab_training_v2.ipynb with all missing cells fixed.
Run from project root: python scripts/fix_v2_notebook.py
"""

import json
import os


def md(source_lines):
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines}


def code(id_, source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": id_,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(md([
    "# Text-to-SQL Fine-tuning — Colab Training\n",
    "**Model:** Qwen2.5-7B-Instruct | **Method:** QLoRA via Unsloth\n\n",
    "Run the cells **top-to-bottom**. Read each section header before running.\n\n",
    "> Requires Colab Pro (A100 / L4). T4 may OOM with Qwen2.5-7B even in 4-bit.",
]))

# ── STEP 1: Check GPU ────────────────────────────────────────────────────────
cells.append(md(["## Step 1 — Check GPU"]))
cells.append(code("check_gpu", [
    "import subprocess, torch\n",
    "\n",
    "print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)\n",
    "print('PyTorch version :', torch.__version__)\n",
    "print('CUDA available  :', torch.cuda.is_available())\n",
    "if torch.cuda.is_available():\n",
    "    p = torch.cuda.get_device_properties(0)\n",
    "    print('GPU             :', p.name)\n",
    "    print('VRAM            : %.1f GB' % (p.total_memory / 1024**3))\n",
    "else:\n",
    "    raise RuntimeError('No GPU! Go to Runtime -> Change runtime type -> GPU')",
]))

# ── STEP 2: Install deps ─────────────────────────────────────────────────────
cells.append(md([
    "## Step 2 — Install Dependencies\n\n",
    "**Version notes:**\n",
    "- `unsloth[colab-new]` from git: auto-detects CUDA/torch, installs matching xformers\n",
    "- `trl < 0.10`: avoids `SFTTrainer` breaking API change in 0.10+\n",
    "- `bitsandbytes >= 0.43`: required for `adamw_8bit` and 4-bit quantization\n",
    "- Do **not** install `xformers` separately (unsloth handles it)\n",
    "- Do **not** install `pyodbc` (Windows-only, not available on Colab Linux)",
]))
cells.append(code("install_deps", [
    "# Install unsloth first — it resolves compatible xformers/triton versions\n",
    '!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q\n',
    "\n",
    "# Install the rest of the training stack with safe version constraints\n",
    "!pip install \\\n",
    "    \"numpy<2.0.0\" \\\n",
    '    "transformers>=4.50.3" \\\n',
    '    "trl>=0.8.0,<0.10.0" \\\n',
    '    "peft>=0.11.0" \\\n',
    '    "accelerate>=0.28.0" \\\n',
    '    "bitsandbytes>=0.43.0" \\\n',
    '    "datasets>=2.18.0" \\\n',
    '    "sqlglot>=23.0.0" \\\n',
    '    "pyyaml" \\\n',
    "    -q\n",
    "\n",
    "print('All dependencies installed!')",
]))
cells.append(code("verify_deps", [
    "import importlib\n",
    "pkgs = ['unsloth','transformers','trl','peft','accelerate','bitsandbytes','datasets','sqlglot']\n",
    "for pkg in pkgs:\n",
    "    try:\n",
    "        m = importlib.import_module(pkg)\n",
    "        print('OK   %-20s %s' % (pkg, getattr(m, '__version__', '?')))\n",
    "    except ImportError:\n",
    "        print('MISS', pkg)",
]))

# ── STEP 3: Mount Drive ──────────────────────────────────────────────────────
cells.append(md([
    "## Step 3 — Mount Google Drive\n\n",
    "Checkpoints and the final model are saved to Drive so they survive session resets.",
]))
cells.append(code("mount_drive", [
    "from google.colab import drive\n",
    "import os\n",
    "\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "# ── CONFIGURE: change this path if you want a different folder on Drive ──\n",
    "DRIVE_ROOT = '/content/drive/MyDrive/text-to-sql-finetune'\n",
    "# ─────────────────────────────────────────────────────────────────────────\n",
    "\n",
    "os.makedirs(DRIVE_ROOT, exist_ok=True)\n",
    "print('Drive root:', DRIVE_ROOT)",
]))

# ── STEP 4: Source code ──────────────────────────────────────────────────────
cells.append(md([
    "## Step 4 — Setup Source Code\n\n",
    "Choose **Option A** (clone from GitHub) or **Option B** (copy from Drive).",
]))
cells.append(code("setup_source", [
    "import os, sys\n",
    "\n",
    "# ── Option A: Clone from GitHub ────────────────────────────────────────\n",
    "# from google.colab import userdata\n",
    "# TOKEN = userdata.get('GITHUB_TOKEN')  # add via Colab Secrets (key icon)\n",
    "# REPO  = 'YOUR_USERNAME/text-to-sql-finetune'\n",
    "# !git clone https://{TOKEN}@github.com/{REPO}.git /content/text-to-sql-finetune\n",
    "\n",
    "# ── Option B: Copy from Google Drive ────────────────────────────────────\n",
    "# !cp -r \"{DRIVE_ROOT}/src\"     /content/text-to-sql-finetune/\n",
    "# !cp -r \"{DRIVE_ROOT}/configs\" /content/text-to-sql-finetune/\n",
    "# !cp -r \"{DRIVE_ROOT}/data\"    /content/text-to-sql-finetune/\n",
    "# !cp -r \"{DRIVE_ROOT}/prompts\" /content/text-to-sql-finetune/\n",
    "\n",
    "# ── Set PROJECT_DIR after the source is in place ────────────────────────\n",
    "PROJECT_DIR = '/content/text-to-sql-finetune'  # adjust if using Option B\n",
    "\n",
    "os.chdir(PROJECT_DIR)\n",
    "# Add project root to sys.path so 'src.*' package imports work\n",
    "if PROJECT_DIR not in sys.path:\n",
    "    sys.path.insert(0, PROJECT_DIR)\n",
    "\n",
    "# Install project as editable package (makes src.helpers, src.training, ... importable)\n",
    "!pip install -e . -q\n",
    "\n",
    "print('CWD      :', os.getcwd())\n",
    "print('Contents :', os.listdir('.'))",
]))

# ── STEP 5: train() helper (ORIGINAL cell, id preserved) ────────────────────
cells.append(md(["## Step 5 — Config & Build Training Command"]))
cells.append(code("5d296f74", [
    "def train(config):\n",
    '    """Build a CLI argument string from a config dict."""\n',
    '    args = ""\n',
    "    for k, v in config.items():\n",
    '        if k.startswith("_"):\n',
    '            args += f\'"{v}" \'\n',
    "        elif isinstance(v, str):\n",
    "            args += f'--{k}=\"{v}\" '\n",
    "        elif isinstance(v, bool) and v:\n",
    '            args += f"--{k} "\n',
    "        elif isinstance(v, float) and not isinstance(v, bool):\n",
    '            args += f"--{k}={v} "\n',
    "        elif isinstance(v, int) and not isinstance(v, bool):\n",
    '            args += f"--{k}={v} "\n',
    "    return args",
]))

# ── cfg_file (ORIGINAL cell, id preserved) ──────────────────────────────────
cells.append(code("4231cf8a", [
    "cfg_file = '/content/text-to-sql-finetune/configs/training_config.yaml'",
]))

# ── Override config (NEW) ────────────────────────────────────────────────────
cells.append(md([
    "### (Optional) Override training config values\n\n",
    "Edit the overrides below to point outputs at Drive and tune batch sizes.",
]))
cells.append(code("override_config", [
    "import yaml\n",
    "\n",
    "with open(cfg_file, 'r') as f:\n",
    "    config_data = yaml.safe_load(f)\n",
    "\n",
    "# Redirect outputs to Drive so they survive session resets\n",
    "config_data['training']['output_dir'] = DRIVE_ROOT + '/outputs/qwen2.5-7b-text2sql'\n",
    "\n",
    "# save_checkpoints: save a LoRA checkpoint every N epochs.\n",
    "# Set to None to disable. Use 1 on Colab for safest recovery.\n",
    "config_data['training']['save_checkpoints'] = 1\n",
    "\n",
    "# Reduce batch size to fit Colab VRAM (effective batch = 1 x 16 = 16)\n",
    "config_data['training']['per_device_train_batch_size'] = 1\n",
    "config_data['training']['gradient_accumulation_steps'] = 16\n",
    "\n",
    "# Write the patched config back; train.py will read it via --config flag\n",
    "with open(cfg_file, 'w') as f:\n",
    "    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)\n",
    "\n",
    "print('Config updated:')\n",
    "print('  output_dir       :', config_data['training']['output_dir'])\n",
    "print('  save_checkpoints :', config_data['training']['save_checkpoints'])\n",
    "print('  batch (eff.)     :',\n",
    "      config_data['training']['per_device_train_batch_size']\n",
    "      * config_data['training']['gradient_accumulation_steps'])",
]))

# ── Resume checkpoint (NEW) ──────────────────────────────────────────────────
cells.append(md([
    "### (Optional) Resume from a previous checkpoint\n\n",
    "If the session was interrupted, uncomment the resume line and re-run Step 6.",
]))
cells.append(code("check_resume", [
    "import glob, os\n",
    "\n",
    "OUTPUT_DIR = config_data['training']['output_dir']\n",
    "epoch_ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'checkpoint-epoch-*')))\n",
    "print('Epoch checkpoints found:', epoch_ckpts or '(none)')\n",
    "\n",
    "# None  -> train from scratch\n",
    "# Uncomment the line below to resume from the latest saved epoch checkpoint:\n",
    "# RESUME_CHECKPOINT = epoch_ckpts[-1] if epoch_ckpts else None\n",
    "RESUME_CHECKPOINT = None\n",
    "print('Resume from:', RESUME_CHECKPOINT)",
]))

# ── STEP 6: train_config (ORIGINAL cell id) — FIXED missing train_args call ─
cells.append(md(["## Step 6 — Build & Run Training"]))
cells.append(code("cb98ae3d", [
    "train_config = {\n",
    '    "config": cfg_file\n',
    "}\n",
    "\n",
    "# Build the CLI argument string (was missing in the original notebook)\n",
    "train_args = train(train_config)\n",
    "print('train_args:', train_args)",
]))

# ── Build script (ORIGINAL cell id) — FIXED path + PYTHONPATH ────────────────
cells.append(code("9edd7570", [
    "# FIXED: correct path is /content/text-to-sql-finetune/... (not /text-to-sql-finetune/)\n",
    "# FIXED: set PYTHONPATH so 'src.*' package imports work inside the subprocess\n",
    "script = (\n",
    "    'PYTHONPATH=/content/text-to-sql-finetune '\n",
    "    'python /content/text-to-sql-finetune/src/training/train.py '\n",
    "    + train_args\n",
    ")\n",
    "print('Command to run:')\n",
    "print(script)",
]))

# ── Run script (ORIGINAL cell id preserved) ──────────────────────────────────
cells.append(code("e3ed7fcb", [
    "!{script}",
]))

# ── STEP 7: Verify outputs (replaces original empty cell) ────────────────────
cells.append(md(["## Step 7 — Verify Outputs"]))
cells.append(code("c1567356", [
    "import glob, os\n",
    "\n",
    "OUTPUT_DIR = config_data['training']['output_dir']\n",
    "print('Output dir:', OUTPUT_DIR, '\\n')\n",
    "\n",
    "epoch_ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'checkpoint-epoch-*')))\n",
    "print('Epoch checkpoints:')\n",
    "for c in epoch_ckpts:\n",
    "    files = os.listdir(c)\n",
    "    size  = sum(os.path.getsize(os.path.join(c, f))\n",
    "                for f in files if os.path.isfile(os.path.join(c, f)))\n",
    "    print('  %-35s  %.1f MB' % (os.path.basename(c), size / 1e6))\n",
    "\n",
    "final = os.path.join(OUTPUT_DIR, 'final_lora')\n",
    "if os.path.exists(final):\n",
    "    print('\\nFinal model:', os.listdir(final))\n",
    "else:\n",
    "    print('\\nFinal model: (not saved yet)')",
]))

# ── Resume guide ─────────────────────────────────────────────────────────────
cells.append(md([
    "---\n",
    "## Resume Guide — When Colab GPU Quota Expires\n\n",
    "1. Re-run **Steps 1–5** (GPU check, install, Drive mount, source setup, config)\n",
    "2. In the **Optional Resume** cell, uncomment:\n",
    "   ```python\n",
    "   RESUME_CHECKPOINT = epoch_ckpts[-1] if epoch_ckpts else None\n",
    "   ```\n",
    "3. Re-run **Step 6** — the trainer loads LoRA weights from Drive and continues\n\n",
    "| `save_checkpoints` | Max progress lost |\n",
    "|---|---|\n",
    "| `1` | 1 epoch |\n",
    "| `2` | 2 epochs |\n",
    "| `None` | Everything (not recommended on Colab) |",
]))

# =============================================================================
# Write notebook
# =============================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "colab_training_v2.ipynb")
out = os.path.normpath(out)

with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Written :", out)
print("Size    :", round(os.path.getsize(out) / 1024, 1), "KB")
print("Cells   :", len(cells))
