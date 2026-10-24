"""Run one model on a Colab GPU instead of Kaya.

Upload your own data/preprocessed.jsonl to /content/Right-Drug-Right-Eye/data/
first (it is not in the repo). Gated models (Llama) also need
`huggingface-cli login` with an account that has accepted the licence.
"""
import os
import subprocess
import sys

REPO = "https://github.com/DharunSomalingam/Right-Drug-Right-Eye.git"
DIR = "/content/Right-Drug-Right-Eye"
MODEL_INDEX = "3"  # 0 Llama-3.1-8B, 1 Qwen2.5-7B, 2 Qwen2.5-14B, 3 Ophtimus-8B

if not os.path.exists(DIR):
    subprocess.run(["git", "clone", REPO, DIR], check=True)
os.chdir(DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
               check=True)
subprocess.run([sys.executable, "src/run_re.py", "--backend", "hf",
                "--model-index", MODEL_INDEX,
                "--input", "data/preprocessed.jsonl",
                "--outdir", "results/h1", "--max-new-tokens", "1024"], check=True)
