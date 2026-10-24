"""Few-shot condition for the H1 prompt (Appendix B.1.3 of the thesis).

Same SYSTEM, labels, rules and output format as prompt_h1. The only change is
an "### Examples:" block of five solved medication-eye relations, inserted
just before the target note.

Each example is one tagged medication shown in a short excerpt of a note, with
its reference label and evidence span. The excerpts come from two annotated
notes that are reserved as exemplars (DONOR_NOTES) and are never part of the
test set, so no evaluated note appears inside the prompt.

  Example 1  shorthand "Drug freq EYE - time"                     -> OU
  Example 2  explicit sig, with a BOTH-eyes line for another drug  -> OD
  Example 3  "Instill in eye." with no side, other drug says left  -> Not Stated
  Example 4  long run-on plan line, eye right after the frequency  -> OS
  Example 5  eye stated once for a group of post-op drugs          -> OS

For the thesis run (PROMPT_VERSION 1.0-thesis) the examples and wording were
fixed before the test run and not tuned. Changing them is a prompt change:
bump prompt_h1.PROMPT_VERSION.
"""
import prompt_h1
from prompt_h1 import SYSTEM, USER  # noqa: F401  (re-exported for run_re)

# Notes the examples were taken from. run_re.py drops these from the input
# whenever the few-shot prompt is used (or --exclude-donors is passed).
DONOR_NOTES = {"980bbb3945", "81e0e34baf"}

# Braces are doubled because the full template still goes through str.format.
EXAMPLES = r"""### Examples:
The following are five solved examples, each a tagged medication in a short
piece of a clinical note, with the correct output. They are not part of the
note you must label. Apply the same reasoning to every listed id in the real
note, which is longer and contains several medications. Reply with only the
JSON array described in ### Output.

Example 1
### Note:
Eye medications:
<Drug id="D1">Cosopt</Drug> BID OU - 7:00AM, 11:30AM
### Drug ids to label (1 total, in this order):
D1 (Cosopt)
### Output:
[{{"id": "D1", "laterality": "OU", "evidence": "BID OU"}}]

Example 2
### Note:
<Drug id="D1">NETARSUDIL</Drug> 0.02 % EYE DROPS Instill 1 drop into the right eye once daily at bedtime.
Indications: wide-angle glaucoma
...... MED RX NONCONTROLLED (SEE ADMIN INSTRUCT F) Instill 1 drop into BOTH eyes six times daily while awake.
### Drug ids to label (1 total, in this order):
D1 (NETARSUDIL)
### Output:
[{{"id": "D1", "laterality": "OD", "evidence": "into the right eye"}}]

Example 3
### Note:
<Drug id="D1">CARBOXYMETHYLCELLULOSE SODIUM</Drug> 1 % EYE LIQUID GEL DROPS Instill in eye.
CHOLECALCIFEROL (VITAMIN D3) 1,000 UNIT CAPSULE Take 1,000 Units by mouth once daily.
DORZOLAMIDE 22.3 MG-TIMOLOL 6.8 MG/ML EYE DROPS Instill 1 drop into the left eye two times daily.
### Drug ids to label (1 total, in this order):
D1 (CARBOXYMETHYLCELLULOSE SODIUM)
### Output:
[{{"id": "D1", "laterality": "Not Stated", "evidence": ""}}]

Example 4
### Note:
IOP OD 11 OS 13 - <Drug id="D1">Timolol</Drug> qam OS - Latanoprost QHS OS - Patanol prn - Ofloxacin prn - PF AT PRN - Tube formed, well closed - IOP good day 1 post op - Start Prednisolone q2h, and Moxifloxacin QID, Atropine daily OS - Hold Timolol and - Latanoprost for now
### Drug ids to label (1 total, in this order):
D1 (Timolol)
### Output:
[{{"id": "D1", "laterality": "OS", "evidence": "qam OS"}}]

Example 5
### Note:
Tube formed, well closed - IOP good day 1 post op - Start Prednisolone q2h, and <Drug id="D1">Moxifloxacin</Drug> QID, Atropine daily OS - Hold Timolol and - Latanoprost for now
### Drug ids to label (1 total, in this order):
D1 (Moxifloxacin)
### Output:
[{{"id": "D1", "laterality": "OS", "evidence": "Moxifloxacin QID, Atropine daily OS"}}]

"""

# Insert the examples in front of the target note. The "### Note:\n{note}"
# marker only exists once in USER (the examples' own headers are not
# followed by "{note}"), so exactly one insertion happens.
GROUNDED_USER = USER.replace("### Note:\n{note}", EXAMPLES + "### Note:\n{note}", 1)
assert GROUNDED_USER != USER, "examples were not inserted"


def build_user(tagged_text, drug_ids=None):
    """Same as prompt_h1.build_user, but with the examples block."""
    if drug_ids is None:
        drug_ids = list(dict.fromkeys(prompt_h1._TAG_ID_RE.findall(tagged_text)))
    return GROUNDED_USER.format(note=tagged_text, n=len(drug_ids), ids=", ".join(drug_ids))
