"""Score every model under --results against the gold standard.

Each tagged medication (drug-note pair) gets one outcome:
    correct_bound  gold OD/OS/OU, predicted the same eye
    correct_ns     gold Not Stated, predicted Not Stated
    correct_amb    gold Ambiguous, predicted Ambiguous
    incorrect      gold eye, predicted a different eye
    omission       gold eye, predicted Not Stated / Ambiguous
    fabrication    gold Not Stated / Ambiguous, predicted an eye
    amb_mismatch   Not Stated <-> Ambiguous
    no_output      drug missing from the output, or the output did not parse

Binding P/R/F1 treat an eye on a gold-eye drug as the positive class:
    TP = correct_bound, FP = incorrect + fabrication,
    FN = every gold-eye drug that was not correct_bound.

Writes to --results:
    binding_table.md/.csv   outcome counts + accuracy, P, R, F1, evidence rates
    timing_table.md/.csv    load, per-note and per-drug generation time
    evidence_report.md      each (non-correct) decision with its evidence
    detail.csv              one row per drug per model

Usage:
    python src/score_h1.py --gold data/preprocessed.jsonl --results results/h1
        [--collapse-ambiguous] [--exclude-inferred] [--include-stub] [--evidence-all]
"""
import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

EYES = {"OD", "OS", "OU"}
ROWS = [
    ("correct_bound", "Correct binding (drug-laterality)"),
    ("correct_ns", "Correct Not Stated"),
    ("correct_amb", "Correct Ambiguous"),
    ("incorrect", "Incorrect binding (misclassification)"),
    ("omission", "Omission (missing laterality relation)"),
    ("fabrication", "Fabrication (laterality for Not Stated)"),
    ("amb_mismatch", "Ambiguous mismatch"),
    ("no_output", "No output (parse fail / drug missing)"),
]
DISPLAY = {  # result dir name (from run_re.safe_name) -> column header
    "Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "Qwen2.5-7B-Instruct": "Qwen 2.5 7B",
    "Qwen2.5-14B-Instruct": "Qwen 2.5 14B",
    "Ophtimus-Instruct-8B": "Ophtimus 8B",
}
ORDER = list(DISPLAY)  # column order; anything else (few-shot, stub) goes after

# Keys tried (in order) when a prediction is a dict rather than a bare label.
LABEL_KEYS = ("laterality", "label", "eye", "pred", "value")
EVIDENCE_KEYS = ("evidence", "quote", "span", "rationale", "reason", "justification")
# Keys tried on the gold note for the full note text (used for grounding check).
NOTE_TEXT_KEYS = ("text", "note_text", "note", "content")


def categorise(gold, pred):
    """Outcome category for one drug (see module docstring)."""
    if pred is None:
        return "no_output"
    if gold in EYES:
        if pred == gold:
            return "correct_bound"
        return "incorrect" if pred in EYES else "omission"
    if gold == "Not Stated":
        if pred == "Not Stated":
            return "correct_ns"
        return "fabrication" if pred in EYES else "amb_mismatch"
    if gold == "Ambiguous":
        if pred == "Ambiguous":
            return "correct_amb"
        return "fabrication" if pred in EYES else "amb_mismatch"
    raise ValueError(f"bad gold {gold!r}")


def collapse(v):
    """--collapse-ambiguous: treat Ambiguous as Not Stated on both sides."""
    return "Not Stated" if v == "Ambiguous" else v


# ---------------------------------------------------------------- evidence
def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _as_text(ev):
    if ev is None:
        return ""
    if isinstance(ev, (list, tuple)):
        return " | ".join(str(e).strip() for e in ev if str(e).strip())
    return str(ev).strip()


def unpack_pred(rec, drug_id):
    """Return (label, evidence) for one drug.

    Supports:
      preds[drug_id] = "OD"
      preds[drug_id] = {"laterality": "OD", "evidence": "..."}
      evidence[drug_id] = "..."   (separate map on the note record)
    """
    rec = rec or {}
    entry = rec.get("preds", {}).get(drug_id)
    label, ev = entry, None
    if isinstance(entry, dict):
        label = next((entry[k] for k in LABEL_KEYS if entry.get(k)), None)
        ev = next((entry[k] for k in EVIDENCE_KEYS if entry.get(k)), None)
    if not ev:
        for k in EVIDENCE_KEYS:
            m = rec.get(k)
            if isinstance(m, dict) and m.get(drug_id):
                ev = m[drug_id]
                break
    return label, _as_text(ev)


def grounded(evidence, note_text, sentence):
    """'yes' if every evidence snippet appears verbatim (case/space-insensitive)
    in the note (or the drug's sentence if no note text), 'no' otherwise,
    '' if no evidence was given."""
    if not evidence:
        return ""
    hay = _norm(note_text or sentence or "")
    parts = [p for p in (s.strip() for s in evidence.split("|")) if p]
    return "yes" if all(_norm(p) in hay for p in parts) else "no"


# ---------------------------------------------------------------- scoring
def score_model(gold_notes, pred_path, args):
    """Return (counts, metrics, detail rows) for one predictions.jsonl."""
    preds = {}
    for l in open(pred_path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            preds[r["note_id"]] = r

    counts = {k: 0 for k, _ in ROWS}
    detail, missing_notes = [], []
    for n in gold_notes:
        rec = preds.get(n["note_id"])
        if rec is None:
            missing_notes.append(n["note_id"])
        note_text = next((n[k] for k in NOTE_TEXT_KEYS if n.get(k)), "")
        for d in n["drugs"]:
            if d["gold"] is None:
                continue
            if args.exclude_inferred and d["gold_source"] == "inferred":
                continue
            g = d["gold"]
            p, ev = unpack_pred(rec, d["id"])
            if args.collapse_ambiguous:
                g, p = collapse(g), (collapse(p) if p else p)
            cat = categorise(g, p)
            counts[cat] += 1
            detail.append({"note_id": n["note_id"], "drug_id": d["id"],
                           "drug": d["text"], "gold": g,
                           "gold_source": d["gold_source"],
                           "pred": p if p else "-", "category": cat,
                           "evidence": ev,
                           "evidence_grounded": grounded(ev, note_text, d["sentence"]),
                           "sentence": d["sentence"]})

    n = sum(counts.values())
    tp = counts["correct_bound"]
    fp = counts["incorrect"] + counts["fabrication"]
    fn = sum(1 for r in detail if r["gold"] in EYES and r["category"] != "correct_bound")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0
    correct = counts["correct_bound"] + counts["correct_ns"] + counts["correct_amb"]
    answered = [r for r in detail if r["category"] != "no_output"]
    with_ev = [r for r in answered if r["evidence"]]
    metrics = {"n": n, "accuracy": correct / n if n else 0.0,
               "precision": prec, "recall": rec_, "f1": f1,
               "n_gold_eye": sum(1 for r in detail if r["gold"] in EYES),
               "n_gold_ns": sum(1 for r in detail if r["gold"] == "Not Stated"),
               "evidence_rate": len(with_ev) / len(answered) if answered else 0.0,
               "grounded_rate": (sum(r["evidence_grounded"] == "yes" for r in with_ev)
                                 / len(with_ev)) if with_ev else 0.0,
               "missing_notes": missing_notes}
    return counts, metrics, detail


def timing(model_dir):
    """Timing summary from predictions.jsonl (per note) and runs.jsonl (per launch)."""
    notes = [json.loads(l) for l in open(model_dir / "predictions.jsonl") if l.strip()]
    runs = [json.loads(l) for l in open(model_dir / "runs.jsonl")] \
        if (model_dir / "runs.jsonl").exists() else []
    gs = [r["gen_seconds"] for r in notes if r.get("gen_seconds") is not None]
    drugs = sum(r.get("n_drugs", 0) for r in notes)
    out_tok = sum(r.get("output_tokens") or 0 for r in notes)
    return {
        "gpu": ", ".join(sorted({r.get("gpu", "?") for r in runs})) or "?",
        "load_s": sum(r.get("load_seconds", 0) for r in runs),
        "notes": len(notes),
        "total_gen_s": sum(gs),
        "mean_s_note": st.mean(gs) if gs else 0,
        "median_s_note": st.median(gs) if gs else 0,
        "s_per_drug": sum(gs) / drugs if drugs else 0,
        "out_tok_s": out_tok / sum(gs) if gs and sum(gs) else 0,
        "mean_in_tok": st.mean([r.get("input_tokens") or 0 for r in notes]) if notes else 0,
        "job_s": sum(r.get("job_seconds", 0) for r in runs),
        "peak_vram_gb": max([r.get("peak_vram_gb", 0) or 0 for r in runs] or [0]),
        "truncated": sum(1 for r in notes if r.get("truncated")),
        "notes_with_errors": sum(1 for r in notes if r.get("errors")),
        "launches": len(runs),
    }


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def md_cell(s, limit=None):
    s = re.sub(r"\s+", " ", str(s or "")).replace("|", "\\|")
    return (s[:limit] + "…") if limit and len(s) > limit else s


def evidence_report(alld, all_rows):
    """Markdown listing each decision with the model's evidence."""
    rows = [r for r in alld if all_rows or not r["category"].startswith("correct")]
    out = ["# Evidence for model decisions"
           + ("" if all_rows else " (errors only)"), ""]
    for model in dict.fromkeys(r["model"] for r in rows):
        mr = [r for r in rows if r["model"] == model]
        out += [f"## {DISPLAY.get(model, model)} ({len(mr)})", "",
                "| Note | Drug | Gold | Pred | Category | Evidence | Grounded | Sentence |",
                "|---|---|---|---|---|---|---|---|"]
        out += [f"| {r['note_id']} | {md_cell(r['drug'])} | {r['gold']} | {r['pred']} "
                f"| {r['category']} | {md_cell(r['evidence']) or '_none_'} "
                f"| {r['evidence_grounded'] or '-'} | {md_cell(r['sentence'], 160)} |"
                for r in mr]
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="preprocessed JSONL (has gold)")
    ap.add_argument("--results", required=True, help="dir with one subdir per model")
    ap.add_argument("--collapse-ambiguous", action="store_true")
    ap.add_argument("--exclude-inferred", action="store_true")
    ap.add_argument("--include-stub", action="store_true")
    ap.add_argument("--evidence-all", action="store_true",
                    help="evidence report/printout covers correct decisions too")
    args = ap.parse_args()

    gold = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]
    res = Path(args.results)
    dirs = [p for p in res.iterdir() if (p / "predictions.jsonl").exists()
            and (args.include_stub or not p.name.endswith("_STUB"))]
    dirs.sort(key=lambda p: (ORDER.index(p.name) if p.name in ORDER else 99, p.name))
    if not dirs:
        raise SystemExit(f"no model results under {res}")
    names = [DISPLAY.get(p.name, p.name) for p in dirs]

    allc, allm, allt, alld = [], [], [], []
    for p in dirs:
        c, m, d = score_model(gold, p / "predictions.jsonl", args)
        allc.append(c); allm.append(m); allt.append(timing(p))
        alld += [dict(r, model=p.name) for r in d]
        if m["missing_notes"]:
            print(f"WARN {p.name}: {len(m['missing_notes'])} gold notes have no "
                  f"prediction: {m['missing_notes']}")
        if m["evidence_rate"] == 0:
            print(f"WARN {p.name}: no evidence found in predictions -- check that the "
                  f"inference prompt/output saves it (keys tried: {EVIDENCE_KEYS})")

    # ---- binding table
    rows = []
    for key, label in ROWS:
        if all(c[key] == 0 for c in allc) and key in ("correct_amb",):
            continue
        rows.append([label] + [f"{c[key]} ({c[key] / m['n']:.0%})" if m["n"] else "0"
                               for c, m in zip(allc, allm)])
    rows.append(["**Total drug mentions**"] + [m["n"] for m in allm])
    rows.append(["Accuracy"] + [f"{m['accuracy']:.3f}" for m in allm])
    rows.append(["Binding precision"] + [f"{m['precision']:.3f}" for m in allm])
    rows.append(["Binding recall"] + [f"{m['recall']:.3f}" for m in allm])
    rows.append(["Binding F1"] + [f"{m['f1']:.3f}" for m in allm])
    rows.append(["Evidence provided"] + [f"{m['evidence_rate']:.0%}" for m in allm])
    rows.append(["Evidence grounded in note"] + [f"{m['grounded_rate']:.0%}" for m in allm])
    m0 = allm[0]
    flags = []
    if args.collapse_ambiguous: flags.append("Ambiguous collapsed to Not Stated")
    if args.exclude_inferred: flags.append("inferred gold excluded")
    head = (f"Gold: {m0['n']} drug mentions -- {m0['n_gold_eye']} with an eye, "
            f"{m0['n_gold_ns']} Not Stated."
            + (f" ({'; '.join(flags)})" if flags else ""))
    bt = head + "\n\n" + md_table(["Binding analysis"] + names, rows)

    # ---- timing table
    trow = lambda lab, f: [lab] + [f(t) for t in allt]
    trows = [
        trow("GPU", lambda t: t["gpu"]),
        trow("Model load (s)", lambda t: f"{t['load_s']:.1f}"),
        trow("Notes", lambda t: t["notes"]),
        trow("Total inference (s)", lambda t: f"{t['total_gen_s']:.1f}"),
        trow("Mean per note (s)", lambda t: f"{t['mean_s_note']:.2f}"),
        trow("Median per note (s)", lambda t: f"{t['median_s_note']:.2f}"),
        trow("Per drug mention (s)", lambda t: f"{t['s_per_drug']:.2f}"),
        trow("Output tokens / s", lambda t: f"{t['out_tok_s']:.1f}"),
        trow("Mean input tokens", lambda t: f"{t['mean_in_tok']:.0f}"),
        trow("Peak VRAM (GB)", lambda t: t["peak_vram_gb"]),
        trow("Job wall time incl. load (s)", lambda t: f"{t['job_s']:.1f}"),
        trow("Truncated outputs", lambda t: t["truncated"]),
        trow("Notes with parse errors", lambda t: t["notes_with_errors"]),
    ]
    tt = md_table(["Run time"] + names, trows)
    gpus = {t["gpu"] for t in allt}
    if len(gpus) > 1:
        tt += f"\n\nWARNING: models ran on different GPUs {gpus} -- times are not comparable."
    if any(t["launches"] > 1 for t in allt):
        tt += "\n\nNote: some models were resumed across several launches; load and job time are summed."

    # ---- write
    (res / "binding_table.md").write_text(bt + "\n")
    (res / "timing_table.md").write_text(tt + "\n")
    (res / "evidence_report.md").write_text(evidence_report(alld, args.evidence_all) + "\n",
                                            encoding="utf-8")
    with open(res / "binding_table.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric"] + names)
        for key, label in ROWS:
            w.writerow([label] + [c[key] for c in allc])
        for k in ("n", "accuracy", "precision", "recall", "f1",
                  "evidence_rate", "grounded_rate"):
            w.writerow([k] + [round(m[k], 4) for m in allm])
    with open(res / "timing_table.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric"] + names)
        for k in allt[0]:
            w.writerow([k] + [t[k] for t in allt])
    with open(res / "detail.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["model", "note_id", "drug_id", "drug", "gold", "gold_source",
                "pred", "category", "evidence", "evidence_grounded", "sentence"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        w.writerows(alld)

    print(bt + "\n\n" + tt)
    title = "all decisions" if args.evidence_all else "non-correct only"
    print(f"\n-- decisions with evidence ({title}) --")
    for r in alld:
        if args.evidence_all or not r["category"].startswith("correct"):
            print(f"  {DISPLAY.get(r['model'], r['model']):<15} {r['note_id']:<18} "
                  f"{r['drug_id']:<4}{r['drug'][:28]:<30} gold={r['gold']:<11}"
                  f"pred={r['pred']:<11}{r['category']}")
            ev = r["evidence"] or "(none)"
            tag = f" [grounded={r['evidence_grounded']}]" if r["evidence"] else ""
            print(f"      evidence: {ev[:200]}{tag}")
    print(f"\n-> {res}/binding_table.md, timing_table.md, evidence_report.md, detail.csv")


if __name__ == "__main__":
    main()