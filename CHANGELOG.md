# Changelog

Changes to the code and prompts. Any change to the prompt text or few-shot
examples also bumps `PROMPT_VERSION` in `src/prompt_h1.py`.

## [Unreleased]

- (add changes here as you make them)

## [v1.0-thesis] - 2026-10-24

Code and prompts reported in the Honours thesis.

- Two-stage pipeline: Label Studio loader with offset checks, then preprocessing
  (line split, `<Drug id="Dn">` tagging), relation extraction and scoring.
- Zero-shot prompt (Appendix B.1.1–B.1.2) and few-shot prompt with five solved
  examples from two reserved donor notes (Appendix B.1.3). `PROMPT_VERSION = "1.0-thesis"`.
- `run_re.py`: Hugging Face Transformers backend (fp16, greedy, 1,024 new
  tokens, no repetition penalty), Ollama backend, CPU stub. Per-note
  checkpointing, GPU timing around `generate()`, aborts on CPU offload.
- `score_h1.py`: binding table (accuracy, binding P/R/F1, evidence grounding),
  timing table, evidence report, per-drug detail.
- Models: Llama-3.1-8B-Instruct, Qwen2.5-7B/14B-Instruct, Ophtimus-Instruct-8B.
