"""
Check the 50 register variants against their base essays.

Checks three things per essay:
  1. word count drift from the base (target: within 5%)
  2. Indian English feature count in each version
  3. content overlap with the base (did the rewrite change the substance?)

Run from anywhere:
    python scripts/check_variants.py

Reads:   data/base/<essay_id>.txt
         data/variants/<essay_id>__urban.txt
         data/variants/<essay_id>__tier2.txt
Writes:  metadata/variant_checks.csv
"""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "data" / "base"
VAR_DIR = ROOT / "data" / "variants"
TOPICS_PATH = ROOT / "data" / "topics.csv"
OUT_PATH = ROOT / "metadata" / "variant_checks.csv"

WORD_DRIFT_LIMIT = 0.05      # 5%
TIER2_FEATURE_MIN = 3
TIER2_FEATURE_MAX = 8
OVERLAP_MIN = 0.55           # flag below this; judge by eye, not by rule

# Regex patterns for features that can be spotted automatically.
# Article dropping cannot be caught this way, so it is measured separately
# as article density, and checked by eye.
FEATURE_PATTERNS = {
    "progressive_state_verb": r"\b(am|is|are|was|were)\s+(knowing|understanding|having|wanting|needing|liking|believing|seeing|hearing|feeling)\b",
    "focus_only": r"\bonly\s*[.,]|\bonly\s+then\b|\bthat\s+is\s+why\b[^.]*\bonly\b",
    "emphatic_itself": r"\bitself\b",
    "plural_uncountable": r"\b(equipments|feedbacks|advices|informations|furnitures|softwares|homeworks|luggages|staffs|researches|evidences)\b",
    "extra_preposition": r"\b(discuss|discussed|discussing)\s+about\b|\bcope\s+up\s+with\b|\bstress\s+on\b",
    "the_same_pronoun": r"\b(submitted|completed|attempted|returned|sent|solved)\s+the\s+same\b",
    "redundant_particle": r"\b(returned|revert|reverted|reply|replied|repeat|repeated)\s+back\b|\brepeat\s+again\b",
    "doubled_word": r"\b(\w+)\s+\1\b",
    "general_extender": r"\band\s+all\b",
}

STOPWORDS = set("""
a an the and or but if then than that this these those of to in on at for with
by from as is am are was were be been being do does did have has had i me my
myself we our it its he she they them their you your not no so such very can
could would should will shall may might must there here what which who whom
when where why how all any both each few more most other some only own same
too s t just now also into out up down over under again further once
""".split())


def words(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def content_words(text):
    return {w for w in words(text) if w not in STOPWORDS and len(w) > 2}


def overlap(a, b):
    """Share of the base essay's content words that survive in the rewrite."""
    ca, cb = content_words(a), content_words(b)
    if not ca:
        return 0.0
    return round(len(ca & cb) / len(ca), 2)


def count_features(text):
    low = text.lower()
    counts = {}
    for name, pat in FEATURE_PATTERNS.items():
        counts[name] = len(re.findall(pat, low))
    return counts


def article_density(text):
    w = words(text)
    if not w:
        return 0.0
    n = sum(1 for x in w if x in {"the", "a", "an"})
    return round(100 * n / len(w), 1)


def main():
    essays = [r["essay_id"] for r in csv.DictReader(open(TOPICS_PATH))]
    rows = []

    for eid in essays:
        base_p = BASE_DIR / f"{eid}.txt"
        if not base_p.exists():
            print(f"MISSING base: {eid}")
            continue
        base = base_p.read_text()
        base_wc = len(base.split())

        for reg in ["urban", "tier2"]:
            var_p = VAR_DIR / f"{eid}__{reg}.txt"
            if not var_p.exists():
                print(f"MISSING variant: {eid} {reg}")
                continue
            var = var_p.read_text()
            wc = len(var.split())
            drift = round((wc - base_wc) / base_wc, 3)
            feats = count_features(var)
            total_feats = sum(feats.values())
            ov = overlap(base, var)

            flags = []
            if abs(drift) > WORD_DRIFT_LIMIT:
                flags.append("word_count")
            if ov < OVERLAP_MIN:
                flags.append("content_changed")
            if reg == "tier2" and not (TIER2_FEATURE_MIN <= total_feats <= TIER2_FEATURE_MAX):
                flags.append("feature_count")
            if reg == "urban" and total_feats > 0:
                flags.append("features_in_urban")

            row = {
                "essay_id": eid, "register": reg,
                "base_words": base_wc, "words": wc, "drift": drift,
                "content_overlap": ov,
                "total_features": total_feats,
                "article_density": article_density(var),
                "base_article_density": article_density(base),
                "flags": "|".join(flags) or "ok",
            }
            row.update(feats)
            rows.append(row)

            print(f"{eid:<26} {reg:<6} words={wc:<4} drift={drift:+.1%} "
                  f"overlap={ov} feats={total_feats} {row['flags']}")

    if not rows:
        print("Nothing checked.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if r["flags"] != "ok"]
    print(f"\nWrote {OUT_PATH}")
    print(f"{len(rows)} variants checked, {len(bad)} flagged.")
    for r in bad:
        print(f"  {r['essay_id']} {r['register']}: {r['flags']}")


if __name__ == "__main__":
    main()