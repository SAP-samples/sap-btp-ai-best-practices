"""Pick two random persona photos with first names that match each photo.

The bundled library (assets/personas/) holds headshot cut-outs taken from the
AI4U one-pager deck. manifest.json tags each photo with the gender it
presents, and names.json holds a pool of generic first names per gender. The
picker draws two different photos, then a name for each from the matching
pool, never repeating a name. Roles are not picked here; they come from the
use case.

Examples:
    # Random pair, printed as JSON to paste into content.json
    python scripts/pick_personas.py

    # Reproducible pair
    python scripts/pick_personas.py --seed 7

Output (stdout):
    {"left_persona": {"name": "Maya", "photo": "p04.webp"},
     "right_persona": {"name": "Jordan", "photo": "p03.webp"}}
"""

import argparse
import json
import random
from pathlib import Path

PERSONAS = Path(__file__).resolve().parent.parent / "assets" / "personas"


def pick(seed=None):
    """Return a dict with left_persona and right_persona {name, photo}.

    Args:
        seed: optional int for a reproducible pick.

    Returns:
        dict ready to merge into the content JSON.
    """
    rng = random.Random(seed)
    photos = json.loads((PERSONAS / "manifest.json").read_text())["personas"]
    names = json.loads((PERSONAS / "names.json").read_text())
    left, right = rng.sample(photos, 2)
    first = rng.choice(names[left["gender"]])
    second = rng.choice([n for n in names[right["gender"]] if n != first])
    return {
        "left_persona": {"name": first, "photo": left["file"]},
        "right_persona": {"name": second, "photo": right["file"]},
    }


def _self_check():
    """Assert over many seeds: distinct photos, distinct names, gender-matched names."""
    photos = {p["file"]: p["gender"] for p in json.loads((PERSONAS / "manifest.json").read_text())["personas"]}
    names = json.loads((PERSONAS / "names.json").read_text())
    for seed in range(200):
        r = pick(seed)
        a, b = r["left_persona"], r["right_persona"]
        assert a["photo"] != b["photo"] and a["name"] != b["name"]
        for p in (a, b):
            assert (PERSONAS / p["photo"]).is_file()
            assert p["name"] in names[photos[p["photo"]]]
    print("pick_personas self-check passed (200 seeds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pick two random, gender-consistent personas.")
    ap.add_argument("--seed", type=int, help="seed for a reproducible pick")
    ap.add_argument("--check", action="store_true", help="run the self-check instead")
    args = ap.parse_args()
    if args.check:
        _self_check()
    else:
        print(json.dumps(pick(args.seed), indent=2))
