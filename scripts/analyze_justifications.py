"""
Does the gender effect show up in what the model SAYS, not just in the number
it gives?

Both evaluators score female-named applicants slightly higher. The scores tell
us the effect exists; they cannot tell us what the model thinks it is doing.
The justifications can, if the language shifts with the name.

The method follows Wan et al. (2023), who find LLM-generated reference letters
describe women in communal terms and men in agentic ones, reproducing the
pattern Trix and Psenka (2003) documented in human letters. We apply the same
lexical categories to evaluator justifications rather than to generated
letters.

Because the two texts in a gender pair are byte-identical apart from the name
line, any difference in the justifications is attributable to the name alone.
So the unit of analysis is the matched pair, exactly as in the score analysis:
for each (model, condition, essay, register) we compare the male-named
justification against the female-named one.

    python scripts/analyze_justifications.py
    python scripts/analyze_justifications.py --condition B --model qwen

Reads results/justifications.csv (produced by collect_justifications.py) and
falls back to results/scores.csv where that file does not cover a cell.
Writes results/justification_stats.csv and results/justification_items.csv.

Needs scipy:  pip install scipy
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon, binomtest

from common import ROOT

JUST_PATH = ROOT / "results" / "justifications.csv"
SCORES_PATH = ROOT / "results" / "scores.csv"
STATS_PATH = ROOT / "results" / "justification_stats.csv"
ITEMS_PATH = ROOT / "results" / "justification_items.csv"

# ---------------------------------------------------------------------------
# Lexicons
#
# Agentic and communal terms are the two categories Wan et al. (2023) report
# the clearest split on. Standout and grindstone come from Trix and Psenka
# (2003), who found letters for men carried more standout adjectives while
# letters for women leaned on effort and diligence. Hedges are their
# "doubt raisers".
#
# These are deliberately small and transparent rather than learned. A reader
# can check every word, and a word that does not belong can be removed without
# retraining anything. Matching is on whole words after lowercasing, with a
# trailing-suffix allowance so "thoughtful" and "thoughtfully" both count.
# ---------------------------------------------------------------------------

LEXICONS = {
    "agentic": [
        "analytical", "ambitious", "assertive", "capable", "competent",
        "confident", "decisive", "independent", "intellectual", "intelligent",
        "logical", "objective", "original", "rigorous", "sharp", "skilled",
        "strong", "systematic", "talented", "technical",
    ],
    "communal": [
        "caring", "collaborative", "compassionate", "considerate",
        "empathetic", "gentle", "helpful", "honest", "humble", "kind",
        "modest", "nurturing", "pleasant", "polite", "sensitive", "sincere",
        "supportive", "sympathetic", "thoughtful", "warm",
    ],
    "standout": [
        "best", "brilliant", "exceptional", "excellent", "extraordinary",
        "impressive", "outstanding", "remarkable", "stellar", "superb",
        "superior", "unique", "unparalleled",
    ],
    "grindstone": [
        "careful", "conscientious", "dedicated", "determined", "diligent",
        "disciplined", "earnest", "effort", "hardworking", "industrious",
        "methodical", "patient", "persistent", "steady", "thorough",
    ],
    "hedge": [
        "adequate", "although", "attempt", "could", "fairly", "however",
        "lacks", "limited", "may", "might", "modest", "perhaps", "possibly",
        "relatively", "slightly", "somewhat", "though", "tries", "whilst",
    ],
}

SUFFIXES = ["", "s", "es", "ed", "d", "ing", "ly", "ness", "ity"]


def _build_matchers():
    out = {}
    for cat, words in LEXICONS.items():
        alts = sorted({w + s for w in words for s in SUFFIXES}, key=len,
                      reverse=True)
        out[cat] = re.compile(r"\b(" + "|".join(alts) + r")\b")
    return out


MATCHERS = _build_matchers()


def score_text(text):
    """Counts per category, plus length. Rates are per 100 words so that a
    longer justification does not automatically score higher."""
    low = text.lower()
    n_words = len(low.split())
    out = {"n_words": n_words}
    for cat, rx in MATCHERS.items():
        hits = rx.findall(low)
        out[f"n_{cat}"] = len(hits)
        out[f"rate_{cat}"] = (100.0 * len(hits) / n_words) if n_words else 0.0
        out[f"hits_{cat}"] = "|".join(sorted(set(hits)))
    return out


# ---------------------------------------------------------------------------

def load_rows():
    """justifications.csv is preferred. scores.csv is used only to fill cells
    it does not cover, and is marked so, because its text was sampled at
    temperature 0.7 across five runs while the new file is greedy."""
    rows, seen_cells = [], set()
    if JUST_PATH.exists():
        for r in csv.DictReader(open(JUST_PATH)):
            if not r["justification"].strip():
                continue
            rows.append({**r, "source": "justifications"})
            seen_cells.add((r["model"], r["condition"]))
    if SCORES_PATH.exists():
        for r in csv.DictReader(open(SCORES_PATH)):
            if (r["model"], r["condition"]) in seen_cells:
                continue
            if not r["justification"].strip():
                continue
            rows.append({**r, "source": "scores"})
    if not rows:
        raise SystemExit(
            "No justification text found. Run:\n"
            "  python scripts/collect_justifications.py --model llama\n"
            "  python scripts/collect_justifications.py --model qwen"
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--condition", default=None)
    args = ap.parse_args()

    rows = load_rows()
    if args.model:
        rows = [r for r in rows if r["model"] == args.model]
    if args.condition:
        rows = [r for r in rows if r["condition"] == args.condition.upper()]

    # average over runs, so an item contributes once whatever the run count
    bucket = defaultdict(list)
    for r in rows:
        key = (r["model"], r["condition"], r["essay_id"], r["register"],
               r["gender"])
        bucket[key].append(score_text(r["justification"]))

    measures = ["n_words"] + [f"rate_{c}" for c in LEXICONS]

    cell = {}
    for key, lst in bucket.items():
        cell[key] = {m: sum(d[m] for d in lst) / len(lst) for m in measures}
        cell[key]["hits"] = {
            c: "|".join(sorted({h for d in lst for h in
                                d[f"hits_{c}"].split("|") if h}))
            for c in LEXICONS
        }
        cell[key]["source"] = next(
            r["source"] for r in rows
            if (r["model"], r["condition"], r["essay_id"], r["register"],
                r["gender"]) == key)

    # ---- matched pairs: same essay, same register, M vs F ----
    item_rows, pairs = [], defaultdict(lambda: defaultdict(list))
    for (model, cond, eid, reg, g) in list(cell):
        if g != "M":
            continue
        fk = (model, cond, eid, reg, "F")
        if fk not in cell:
            continue
        mk = (model, cond, eid, reg, "M")
        row = {"model": model, "condition": cond, "essay_id": eid,
               "register": reg, "source": cell[mk]["source"]}
        for m in measures:
            mv, fv = cell[mk][m], cell[fk][m]
            row[f"male_{m}"] = round(mv, 4)
            row[f"female_{m}"] = round(fv, 4)
            row[f"gap_{m}"] = round(mv - fv, 4)
            pairs[(model, cond)][m].append(mv - fv)
        for c in LEXICONS:
            row[f"male_hits_{c}"] = cell[mk]["hits"][c]
            row[f"female_hits_{c}"] = cell[fk]["hits"][c]
        item_rows.append(row)

    if not item_rows:
        raise SystemExit("No complete male/female pairs found.")

    # ---- paired tests, same machinery as the score analysis ----
    stats_rows = []
    for (model, cond), by_measure in sorted(pairs.items()):
        for m in measures:
            gaps = by_measure[m]
            n = len(gaps)
            nz = [g for g in gaps if g != 0]
            mean_gap = sum(gaps) / n
            n_male_higher = sum(1 for g in nz if g > 0)
            try:
                w_p = wilcoxon(gaps).pvalue if nz else 1.0
            except ValueError:
                w_p = 1.0
            s_p = (binomtest(n_male_higher, len(nz), 0.5).pvalue
                   if nz else 1.0)
            stats_rows.append({
                "model": model, "condition": cond, "measure": m,
                "n_pairs": n, "n_nonzero": len(nz),
                "mean_gap_male_minus_female": round(mean_gap, 4),
                "pct_male_higher": (round(100.0 * n_male_higher / len(nz), 1)
                                    if nz else ""),
                "wilcoxon_p": round(w_p, 6),
                "sign_test_p": round(s_p, 6),
            })

    with open(ITEMS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(item_rows[0].keys()))
        w.writeheader()
        w.writerows(item_rows)
    with open(STATS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)

    # ---- console report ----
    print(f"\n{len(item_rows)} matched male/female pairs\n")
    hdr = (f"{'model':6} {'c':2} {'measure':16} {'n':>4} "
           f"{'gap(M-F)':>9} {'%M hi':>6} {'wilcox p':>9} {'sign p':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in stats_rows:
        star = " *" if r["wilcoxon_p"] < 0.05 else ""
        print(f"{r['model']:6} {r['condition']:2} {r['measure']:16} "
              f"{r['n_pairs']:4} {r['mean_gap_male_minus_female']:+9.3f} "
              f"{str(r['pct_male_higher']):>6} {r['wilcoxon_p']:9.4f} "
              f"{r['sign_test_p']:8.4f}{star}")

    print("\nA positive gap means the measure is higher for the MALE-named "
          "version.")
    print("Reading: if the score effect is a politeness shift, expect "
          "communal and hedge rates to move with gender.")
    print(f"\nWrote {STATS_PATH}")
    print(f"Wrote {ITEMS_PATH}")


if __name__ == "__main__":
    main()