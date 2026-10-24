# Working on this repo

## Never commit clinical data
Notes, Label Studio exports, `preprocessed.jsonl`, model outputs and
`detail.csv` all contain note text or clinician labels. They live in `data/`,
`results/` and `logs/`, which are git-ignored. Before every commit, run
`git status` and check that nothing from those folders is staged.

## Branches
- `main`: always runs (`--stub` dry run passes).
- `feature/<name>` (e.g. `feature/dynamic-fewshot`, `feature/ner-e2e`): new
  work, merged into `main` once the dry run passes.
- Tags such as `v1.0-thesis` mark versions that results were reported from. Never move a tag.

## Before merging
```bash
python src/run_re.py --model-index 0 --stub --input data/preprocessed.jsonl --outdir results/dry
python src/run_re.py --model-index 0 --stub --prompt grounded --input data/preprocessed.jsonl --outdir results/dry
python src/score_h1.py --gold data/preprocessed.jsonl --results results/dry --include-stub
```

## Changing a prompt
1. Edit `src/prompt_h1.py` / `src/prompt_h1_grounded.py`.
2. Bump `PROMPT_VERSION` (e.g. `1.0-thesis` → `1.1`).
3. Add a line to `CHANGELOG.md` under *Unreleased*.
4. Run into a new `--outdir`. Never resume an old run with a new prompt.
