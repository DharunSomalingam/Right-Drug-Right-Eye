"""Stage 1: turn gold records into the tagged text the model sees.

For each note:
  1. split the raw text into sentences/lines (never inside a drug span)
  2. wrap each gold drug as <Drug id="Dn">name</Drug>, numbered D1.. in
     order of appearance
  3. clean the text OUTSIDE the tags: de-identification masks (*, **, */*)
     become "...", odd symbols are removed, whitespace is collapsed
  4. drop lines that have no letters/digits and no drug

The drug span text itself is never changed, so the tag always holds exactly
what the annotator selected.

Output: one JSON line per note
    {"note_id", "tagged_text", "sentences", "drugs": [{"id", "text", "gold", ...}]}

Usage:
    python src/preprocess.py --export data/ls_export.json \
        --out data/preprocessed.jsonl [--show]
"""
import re

# Sentence/line boundaries: 2+ spaces (line breaks in the export), bullets,
# or . ! ? followed by a capital letter
BOUNDARY_RE = re.compile(r"\s{2,}|[\u2022\u00b7\u25aa\u25cf]|(?<=[.!?])\s+(?=[A-Z])")
# De-identification masks: *, **, */* ...
MASK_RE = re.compile(r"\*[\*/]*")
# Bullets, block characters and markup symbols
SPECIAL_RE = re.compile(r"[\u2022\u00b7\u25aa\u25cf\u2588#~|_\[\]{}<>\\^`]")
# Repeated "..." left after masking
DOTS_RE = re.compile(r"(?:\.\.\.\s*){2,}")
# Any run of whitespace
WS_RE = re.compile(r"\s+")
# A line is kept only if it contains a letter or digit
HAS_WORD_RE = re.compile(r"[A-Za-z0-9]")


def split_sentences(text, protected):
    """Return (start, end) offsets of each sentence. A boundary that falls
    inside a protected (drug) span is ignored, so a drug is never split."""
    def inside(pos):
        return any(s < pos < e for s, e in protected)

    cuts = [0]
    for m in BOUNDARY_RE.finditer(text):
        if not inside(m.start()) and not inside(m.end()):
            cuts.extend([m.start(), m.end()])
    cuts.append(len(text))

    out = []
    for s, e in zip(cuts[::2], cuts[1::2]):
        if s < e and text[s:e].strip():
            out.append((s, e))
    return out


def clean(seg):
    """Normalise a piece of text that lies outside any drug tag."""
    seg = seg.replace("\u00a0", " ")
    seg = MASK_RE.sub(" ... ", seg)
    seg = SPECIAL_RE.sub(" ", seg)
    seg = DOTS_RE.sub("... ", seg)
    return WS_RE.sub(" ", seg)


def preprocess_note(rec):
    """Gold record from ls_loader -> tagged note record."""
    text, drugs = rec["text"], rec["drugs"]

    # ids follow reading order (ls_loader sorts drugs by start offset)
    for i, d in enumerate(drugs, 1):
        d["id"] = f"D{i}"

    sents = split_sentences(text, [(d["start"], d["end"]) for d in drugs])

    lines, sent_meta = [], []
    for si, (s, e) in enumerate(sents):
        in_sent = [d for d in drugs if s <= d["start"] and d["end"] <= e]
        parts, pos = [], s
        for d in in_sent:
            parts.append(clean(text[pos:d["start"]]))
            parts.append(f' <Drug id="{d["id"]}">{d["text"]}</Drug> ')
            pos = d["end"]
        parts.append(clean(text[pos:e]))
        line = WS_RE.sub(" ", "".join(parts)).strip()

        if not in_sent and not HAS_WORD_RE.search(line):
            continue
        for d in in_sent:
            d["sentence_idx"] = len(lines)
            d["sentence"] = line
        sent_meta.append({"idx": len(lines), "start": s, "end": e,
                          "drug_ids": [d["id"] for d in in_sent]})
        lines.append(line)

    # every gold drug must have landed in exactly one sentence
    missing = [d["text"] for d in drugs if "sentence_idx" not in d]
    if missing:
        raise ValueError(f"{rec['note_id']}: drugs not mapped to a sentence: {missing}")

    return {"note_id": rec["note_id"], "tagged_text": "\n".join(lines),
            "sentences": sent_meta,
            "drugs": [{k: d[k] for k in ("id", "text", "start", "end", "gold",
                                         "gold_source", "sentence_idx", "sentence")}
                      for d in drugs]}


if __name__ == "__main__":
    import argparse, json, sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ls_loader import load_export

    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, nargs="+",
                    help="one or more Label Studio JSON exports")
    ap.add_argument("--out", required=True, help="preprocessed JSONL")
    ap.add_argument("--show", action="store_true", help="print tagged notes")
    args = ap.parse_args()

    recs, warns = [], []
    for p in args.export:
        r, w = load_export(p)
        recs += r; warns += w
    ids = [r["note_id"] for r in recs]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        sys.exit(f"duplicate note_ids across exports: {dup}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            p = preprocess_note(r)
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            n_by = {}
            for d in p["drugs"]:
                n_by[d["gold"]] = n_by.get(d["gold"], 0) + 1
            print(f"{p['note_id']:<20} {len(p['sentences']):>3} sentences  "
                  f"{len(p['drugs'])} drugs  gold={n_by}")
            if args.show:
                print("-" * 70 + "\n" + p["tagged_text"] + "\n" + "-" * 70)
    for w in warns:
        print("WARN", w)
    print(f"-> {args.out}")
