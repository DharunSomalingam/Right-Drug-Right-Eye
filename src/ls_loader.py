"""Read a Label Studio JSON export into gold records.

One record per annotated note:
    {"note_id", "task_id", "text",
     "drugs": [{"ls_id", "start", "end", "text", "gold", "gold_source", "evidence"}]}

gold        OD / OS / OU / Not Stated / Ambiguous, from the
            reference_laterality choice on the drug span
gold_source "choice"   -> set explicitly by the annotator
            "inferred" -> no choice set; taken from the single OD/OS/OU token
                          in the spans the drug is related to (warned, and can
                          be dropped at scoring time with --exclude-inferred)

Every span's character offsets are checked against the note text; a span
whose text does not match its offsets is dropped with a warning.

Usage:  python src/ls_loader.py export.json   (prints the drugs per note)
"""
import json
import re
from pathlib import Path

# Label Studio choice value -> canonical label
CANON = {
    "OD": "OD", "OS": "OS", "OU": "OU",
    "NOT_STATED": "Not Stated", "NOT STATED": "Not Stated",
    "AMBIGUOUS": "Ambiguous",
}
DRUG_LABELS = {"MEDICATION", "DRUG"}  # span labels that count as a drug
LAT_TOKEN_RE = re.compile(r"\b(OD|OS|OU)\b")


def note_id_from_upload(name):
    """Label Studio prefixes uploads with a hash: 'a1b2c3d4-x.txt' -> 'x'."""
    stem = Path(name).stem
    return stem.split("-", 1)[1] if "-" in stem else stem


def load_export(path):
    """Return (records, warnings) for one export file."""
    tasks = json.load(open(path, encoding="utf-8"))
    records, warnings = [], []

    for t in tasks:
        text = t["data"]["text"]
        nid = note_id_from_upload(t.get("file_upload") or str(t["id"]))
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            warnings.append(f"{nid}: no annotation, skipped")
            continue
        if len(anns) > 1:
            warnings.append(f"{nid}: {len(anns)} annotations, using the latest")
        ann = max(anns, key=lambda a: a.get("updated_at") or "")

        # spans: id -> span; choices: span id -> laterality choice; rels: arrows
        spans, choices, rels = {}, {}, []
        for r in ann["result"]:
            if r["type"] == "relation":
                rels.append(r)
                continue
            v = r["value"]
            if r["type"] == "labels":
                s, e = v["start"], v["end"]
                if text[s:e] != v["text"]:
                    warnings.append(f"{nid}/{r['id']}: offset drift "
                                    f"{v['text']!r} != {text[s:e]!r}, dropped")
                    continue
                spans[r["id"]] = {"start": s, "end": e, "text": v["text"],
                                  "label": v["labels"][0]}
            elif r["type"] == "choices" and r["from_name"] == "reference_laterality":
                choices[r["id"]] = v["choices"][0]

        # one entry per drug span, with its gold laterality
        drugs = []
        for sid, sp in spans.items():
            if sp["label"] not in DRUG_LABELS:
                continue
            evidence = [dict(spans[r["to_id"]], rel=(r.get("labels") or [None])[0])
                        for r in rels if r["from_id"] == sid and r["to_id"] in spans]

            gold, source = None, None
            if sid in choices:
                gold = CANON.get(choices[sid].upper())
                source = "choice"
                if gold is None:
                    warnings.append(f"{nid}/{sp['text']}: unknown choice "
                                    f"{choices[sid]!r}")
            else:
                toks = {m.group(1) for ev in evidence
                        for m in LAT_TOKEN_RE.finditer(ev["text"])}
                if len(toks) == 1:
                    gold, source = toks.pop(), "inferred"
                    warnings.append(
                        f"{nid}/{sp['text']}@{sp['start']}: no reference_laterality "
                        f"choice; gold={gold} INFERRED from relation "
                        f"{[e['rel'] for e in evidence]} -> "
                        f"{[e['text'] for e in evidence]}. Set the choice in "
                        f"Label Studio to make this explicit.")
                else:
                    warnings.append(f"{nid}/{sp['text']}@{sp['start']}: no gold "
                                    f"laterality, dropped from scoring")

            drugs.append({"ls_id": sid, **{k: sp[k] for k in ("start", "end", "text")},
                          "gold": gold, "gold_source": source, "evidence": evidence})

        drugs.sort(key=lambda d: d["start"])
        for a, b in zip(drugs, drugs[1:]):
            if b["start"] < a["end"]:
                warnings.append(f"{nid}: overlapping drug spans {a['text']!r} / "
                                f"{b['text']!r}")
        records.append({"note_id": nid, "task_id": t["id"], "text": text,
                        "drugs": drugs})

    return records, warnings


if __name__ == "__main__":
    import sys
    recs, warns = load_export(sys.argv[1])
    for r in recs:
        print(f"{r['note_id']}: {len(r['drugs'])} drugs")
        for d in r["drugs"]:
            print(f"   {d['start']:>5}  {d['text']:<34} {str(d['gold']):<11} {d['gold_source']}")
    for w in warns:
        print("WARN", w)
