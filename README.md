# Right Drug, Right Eye?

Code for the UWA Honours thesis **"Right Drug, Right Eye? Evaluating LLMs for
Medication–Eye Relation Extraction in Ophthalmology Clinical Notes for
Resource-Constrained Healthcare Settings"** (Dharun Somalingam, 2026).
Supervisors: Dr Vasanth Muthusamy, Prof. Andrew Turpin, Prof. Amitava Datta,
Dr Naeha Sharif.

> **Status: active development** at the Lions Eye Institute (pmc091 group, UWA).
> The code on `main` changes frequently. The exact version used for the
> Honours thesis is tagged **`v1.0-thesis`**; check out that tag to reproduce
> the thesis results. See [CHANGELOG.md](CHANGELOG.md) for what changed since.

The task: given an ophthalmology clinical note where each medication is already
marked, decide **which eye each medication is used in**: OD (right), OS
(left), OU (both), Not Stated, or Ambiguous. The model also quotes the
evidence span from the note. This is relation extraction (RE) only. Gold drug
spans come from the clinician's annotation, so the evaluation measures binding
a drug to its eye, not finding the drug.

Four locally deployable open-weight models are compared under two prompting
conditions (zero-shot, and few-shot with five solved examples):

| Index | Model | Hugging Face checkpoint |
|---:|---|---|
| 0 | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| 1 | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| 2 | Qwen2.5-14B-Instruct | `Qwen/Qwen2.5-14B-Instruct` |
| 3 | Ophtimus-Instruct-8B (ophthalmology) | `BaekSeungJu/Ophtimus-Instruct-8B` |

## Pipeline

```
Label Studio export ──► ls_loader.py ──► preprocess.py ──► run_re.py ──► score_h1.py
 (clinician gold)      gold records     tagged notes      predictions    tables
                       offsets checked  <Drug id="Dn">    one model,     binding,
                                                          checkpointed   timing,
                                                                         evidence
```

| File | What it does |
|---|---|
| `src/ls_loader.py` | Reads a Label Studio JSON export into gold records and checks every span's offsets against the note text |
| `src/preprocess.py` | Stage 1: splits notes into lines, wraps each gold drug as `<Drug id="Dn">name</Drug>`, and cleans the text outside the tags |
| `src/prompt_h1.py` | Zero-shot prompt (thesis Appendix B.1.1–B.1.2) plus a strict JSON output parser |
| `src/prompt_h1_grounded.py` | Few-shot prompt: the same prompt plus five solved examples (Appendix B.1.3) |
| `src/run_re.py` | Stage 2: runs one model over all notes, timed and resumable |
| `src/score_h1.py` | Binding table, timing table, evidence report and per-drug detail |
| `run_h1.sh` | SLURM array job: one task per model (V100-32GB, UWA Kaya) |
| `colab_run.py` | Runs one model on Google Colab |

## Data (not included)

**No notes, annotations or model outputs are in this repository.**
`data/`, `results/` and `logs/` are git-ignored.

- Notes: the de-identified [OHSU Glaucoma Medication Dataset](https://github.com/jche253/Glaucoma_Med_Dataset)
  (Chen et al., 2022).
- Annotations: the clinician's medication–eye annotations are not released.
  To use your own, annotate in Label Studio with:
  - a span label `MEDICATION` (or `DRUG`) on each drug
  - a per-span choice `reference_laterality` set to one of `OD`, `OS`, `OU`,
    `Not Stated` or `Ambiguous`
  - optional relations from the drug to its laterality/evidence span
- Each drug is tagged once per note, on the mention that carries the eye.

## Setup

```bash
pip install -r requirements.txt          # pick the torch build for your GPU
huggingface-cli login                    # Llama is a gated model
```

On a cluster without internet on compute nodes, download the checkpoints to
`$HF_HOME` on a login node first. `run_h1.sh` sets `HF_HUB_OFFLINE=1`.

## Usage

**1. Preprocess** (CPU)

```bash
python src/preprocess.py --export data/ls_export.json --out data/preprocessed.jsonl --show
```

You can pass several exports to `--export`; note ids must be unique.

**2. Dry run** (CPU, no model, checks the whole pipeline)

```bash
python src/run_re.py --model-index 0 --stub --input data/preprocessed.jsonl --outdir results/h1_dry
python src/score_h1.py --gold data/preprocessed.jsonl --results results/h1_dry --include-stub
```

**3. Run the models** (GPU)

```bash
sbatch run_h1.sh                     # zero-shot, all four models
PROMPT=grounded sbatch run_h1.sh     # few-shot, all four models
# or one model directly:
python src/run_re.py --model-index 2 --prompt zero --input data/preprocessed.jsonl --outdir results/h1
```

If you use `--prompt grounded`, the two donor notes the examples come from are
dropped automatically. Add `--exclude-donors` to a zero-shot run so both
conditions are scored on the same notes. To use an Ollama server instead of
Transformers, add `--backend ollama --host http://<server>:11434`.

A job that dies can be resubmitted. It resumes from `predictions.jsonl`, and
`runs.jsonl` gets one line per launch.

**4. Score** (CPU)

```bash
python src/score_h1.py --gold data/preprocessed.jsonl --results results/h1
```

Options: `--collapse-ambiguous` (treat Ambiguous as Not Stated),
`--exclude-inferred` (drop gold labels inferred from relations rather than
set explicitly), `--evidence-all` (list correct decisions in the evidence
report too).

## Outputs

`run_re.py` writes `results/<run>/<model>/`:
- `predictions.jsonl`: one line per note, holding the raw output, parsed labels, evidence, parse errors, token counts and generation time
- `runs.jsonl`: one line per launch, holding the settings, GPU, load time, job time and peak VRAM

`score_h1.py` writes to `results/<run>/`:
- `binding_table.md/.csv`, `timing_table.md/.csv`, `evidence_report.md`, `detail.csv`

Each tagged medication is scored as one of:

| Category | Gold | Prediction |
|---|---|---|
| Correct binding | OD/OS/OU | same eye |
| Correct Not Stated | Not Stated | Not Stated |
| Incorrect binding | OD/OS/OU | a different eye |
| Omission | OD/OS/OU | Not Stated / Ambiguous |
| Fabrication | Not Stated | an eye |
| No output | any | drug missing from the output, or the JSON did not parse |

Binding precision, recall and F1 treat a correct eye as the positive class
(TP = correct binding; FP = incorrect + fabrication; FN = gold-eye drugs not
correctly bound).

## Versioning and reproducibility

- `v1.0-thesis` marks the code and prompts reported in the thesis. `main` moves on from there.
- `PROMPT_VERSION` in `src/prompt_h1.py` names the prompt. Bump it whenever
  the system prompt, user template or few-shot examples change.
- Every launch of `run_re.py` writes `prompt_version` and `code_commit` (git
  hash, `+dirty` if there are uncommitted edits) to `runs.jsonl`. A result
  can always be traced back to the code and prompt that produced it.
- Write each experiment to its own `--outdir` (e.g. `results/2026-11_prompt-1.1`)
  rather than reusing `results/h1`. A resumed run silently mixes outputs if the
  code changed in between.

## Roadmap

- [x] H1: drug to laterality binding on gold drug spans, zero-shot and few-shot (thesis)
- [ ] End-to-end NER + RE (the model finds the drugs itself, no drug lexicon)
- [ ] Dynamic few-shot: retrieval-based selection of examples per note
- [ ] Evaluation across the easy / medium / hard note tiers
- [ ] Qwen2.5-14B on MI210 (ROCm) as well as V100
- [ ] Larger annotated test set

## Inference settings

These are the settings from Section 3.6 of the thesis, and the defaults in `run_re.py`:
- Transformers, fp16 with no quantisation (V100 has no bf16)
- the model's own chat template, with system and user as separate messages
- greedy decoding, max 1,024 new tokens, repetition penalty 1.0, seed 42
- one request per note covering all of its medications

Timing is measured around `model.generate()` only, with
`torch.cuda.synchronize()` on both sides. The run aborts if any layer is
offloaded to CPU or disk.
