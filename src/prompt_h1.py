"""Zero-shot prompt for H1 (drug -> laterality binding) and its output parser.

The model gets one whole preprocessed note, where every gold medication is
wrapped as <Drug id="Dn">name</Drug>, and must return one JSON object per id:

    [{"id": "D1", "laterality": "OU", "evidence": "BID OU"}, ...]

SYSTEM and USER at PROMPT_VERSION "1.0-thesis" are the exact templates in
Appendix B.1.1 / B.1.2 of the Honours thesis (git tag v1.0-thesis). The prompt
is part of the experimental condition: whenever you change SYSTEM, USER or the
few-shot examples, bump PROMPT_VERSION and add a line to CHANGELOG.md.
run_re.py writes the version into every runs.jsonl line, so each result can
be traced back to the prompt that produced it.

parse_output() turns the raw model text back into {id: label} and
{id: evidence}, and records every problem it had to work around (bad JSON,
unknown ids, evidence that is not in the note, ...) so the scorer can report
it instead of silently hiding it.
"""
import json
import re

PROMPT_VERSION = "1.0-thesis"

LABELS = ("OD", "OS", "OU", "Not Stated", "Ambiguous")

SYSTEM = (
    "You are a clinical NLP annotator for ophthalmology notes. "
    "You link each tagged medication to the eye it is applied to, using only "
    "what the note says. You never guess, you never invent medication IDs, and "
    "you never quote text that is not in the note."
)

# Literal braces in the JSON shape are doubled ({{ }}) because the template is
# filled with str.format(note=..., n=..., ids=...).
USER = r"""### Task:
The clinical note below has been split into sentences, one per line.
Each ophthalmic medication to label is tagged in the form <Drug id="Dn">name</Drug>.
The ids you must label are listed after the note.
For EVERY listed id, read the whole note, decide which eye the note says that
medication is used in, and quote the words that told you.

### Labels:
- OD: right eye
- OS: left eye
- OU: both eyes
- Not Stated: the note gives no eye for this medication
- Ambiguous: the note refers to an eye for this medication, but it cannot be
  resolved to exactly one of OD, OS or OU (for example "the operative eye"
  when no operated eye is stated)

### Rules:
1. Label only the listed ids. Ignore every medication that is not tagged.
2. Use the whole note. An eye may be stated with the medication, in another
   statement about the same medication, or in a statement that covers several
   medications. Abbreviated and written-out forms of an eye both count.
3. An eye may be written near the medication or far from it, including as a
   section heading (for example "OD" or "OS" at the start of a block). For
   every eye you find, read the surrounding text closely and judge whether
   that eye really belongs to this medication before you use it. Being near
   the medication does not make it the right eye, and being far away does
   not make it wrong.
4. Do not take an eye from a different medication, or from a diagnosis, exam
   finding or test result that does not concern this medication.
5. If no statement gives an eye for the medication, but clinical information in
   the note makes the eye clear (for example, the medication is given for a
   procedure stated for one eye), use that eye and quote the clinical
   statement as evidence.
6. If the note gives no eye for this medication, answer Not Stated. Do not guess.
7. "evidence" is a short continuous span of the note that contains the words
   giving the eye.
   - Copy it character for character from the note, as ONE unbroken piece.
   - Do not paraphrase, and do not shorten to an abbreviation the note does
     not use.
   - Do not include the <Drug id="Dn"> or </Drug> tags; copy only the words.
   - For Not Stated, use an empty string "".
8. Check before answering: if your evidence names an eye, the label must be
   that eye. Use Not Stated only when your evidence is empty.
9. Return exactly one object per listed id, in the listed order. Never add,
   skip, repeat or renumber ids.

### Output:
Return ONLY a JSON array. Start with [ and end with ]. No markdown fences and
no explanation before or after it.
Each object has exactly three keys, in this order: "id", "laterality", "evidence".
Inside a string value, write a double quote as \" and a backslash as \\.
Shape:
[{{"id": "D1", "laterality": "<one label>", "evidence": "<exact span or empty string>"}},
 {{"id": "D2", "laterality": "<one label>", "evidence": "<exact span or empty string>"}}]

### Note:
{note}

### Drug ids to label ({n} total, in this order):
{ids}

### Output:"""


_TAG_ID_RE = re.compile(r'<Drug\s+id="([^"]+)"')


def build_user(tagged_text, drug_ids=None):
    """Fill the user template for one note.

    drug_ids defaults to the tag ids found in the note, in order of appearance.
    """
    if drug_ids is None:
        drug_ids = list(dict.fromkeys(_TAG_ID_RE.findall(tagged_text)))
    return USER.format(note=tagged_text, n=len(drug_ids), ids=", ".join(drug_ids))


# ------------------------------------------------------------------ parsing

# Greedy: from the first "[" to the last "]", so text before/after is ignored.
_JSON_RE = re.compile(r"\[.*\]", re.DOTALL)
# "evidence" is the last key, so its value runs until a quote followed by
# } and then , or ]. Used only to repair unescaped quotes inside evidence.
_EVIDENCE_VAL_RE = re.compile(r'("evidence"\s*:\s*")(.*?)("\s*\}\s*(?:,|\]))', re.DOTALL)

# Label variants models actually produce -> canonical label.
_ALIASES = {
    "od": "OD", "right": "OD", "right eye": "OD", "re": "OD",
    "os": "OS", "left": "OS", "left eye": "OS", "le": "OS",
    "ou": "OU", "both": "OU", "both eyes": "OU", "bilateral": "OU",
    "not stated": "Not Stated", "not_stated": "Not Stated", "none": "Not Stated",
    "null": "Not Stated", "n/a": "Not Stated", "": "Not Stated",
    "ambiguous": "Ambiguous", "unclear": "Ambiguous",
}


def normalise_label(v):
    """Model's laterality value -> canonical label, or None if unrecognised."""
    if v is None:
        return "Not Stated"
    k = re.sub(r"\s+", " ", str(v).strip().lower()).strip(" .")
    return _ALIASES.get(k)


def _escape_evidence_quotes(text):
    """Escape bare double quotes inside evidence values (e.g. an inch mark
    copied from the note) so the array becomes valid JSON."""
    def fix(m):
        return m.group(1) + re.sub(r'(?<!\\)"', r'\\"', m.group(2)) + m.group(3)
    return _EVIDENCE_VAL_RE.sub(fix, text)


def _norm_ws(s: str) -> str:
    """Lower-case and collapse whitespace, for the evidence-in-note check."""
    return " ".join(s.split()).lower()


def parse_output(raw: str, ids: list[str], text: str | None = None):
    """Parse one model response.

    raw   model output text
    ids   the drug ids that were asked for, e.g. ["D1", "D2"]
    text  the note with tags stripped; if given, every evidence span is
          checked to appear in it (case- and whitespace-insensitive)

    Returns (preds, evidence, errors):
      preds    {id: canonical label}  -- ids missing here are scored "No output"
      evidence {id: evidence string}
      errors   list of human-readable problems (empty list = clean output)
    """
    preds: dict[str, str] = {}
    evidence: dict[str, str] = {}
    errors: list[str] = []

    if not raw or not raw.strip():
        return preds, evidence, ["empty output"]

    m = _JSON_RE.search(raw)
    if not m:
        return preds, evidence, ["no JSON array found"]
    blob = m.group(0)

    try:
        rows = json.loads(blob)
    except ValueError as e:
        # one repair attempt only: unescaped quotes inside evidence
        try:
            rows = json.loads(_escape_evidence_quotes(blob))
            errors.append(f"repaired JSON: {e}")
        except ValueError:
            return preds, evidence, [f"JSON parse error: {e}"]

    if not isinstance(rows, list):
        return preds, evidence, ["top-level JSON is not an array"]

    id_set = set(ids)
    norm_text = _norm_ws(text) if text else None

    for row in rows:
        if not isinstance(row, dict):
            errors.append(f"non-object row {row!r}")
            continue

        did = row.get("id") or row.get("drug_id")
        if not isinstance(did, str) or did not in id_set:
            errors.append(f"unknown id {did!r}")      # invented / renumbered id
            continue
        if did in preds:
            errors.append(f"duplicate id {did}")      # first answer wins
            continue

        lab = normalise_label(row.get("laterality"))
        if lab is None:
            errors.append(f"{did}: unrecognised laterality {row.get('laterality')!r}")
            continue
        preds[did] = lab

        ev = row.get("evidence")
        if ev is None:
            ev = row.get("span") or row.get("quote") or ""
        ev = str(ev).strip()
        evidence[did] = ev

        if lab in ("OD", "OS", "OU") and not ev:
            errors.append(f"{did}: {lab} with no evidence")
        elif ev and norm_text is not None and _norm_ws(ev) not in norm_text:
            errors.append(f"{did}: evidence not found in note")

    if len(rows) != len(ids):
        errors.append(f"row count {len(rows)} != {len(ids)} ids")
    for did in ids:
        if did not in preds:
            errors.append(f"missing from output: {did}")

    return preds, evidence, errors
