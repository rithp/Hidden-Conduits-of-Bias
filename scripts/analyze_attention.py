"""
Turn the per-essay rollout output into the test the paper actually needs.

attention_rollout.py produces one row per item with a dialect-to-content
relevance ratio. On its own a ratio of 1.3 means nothing: it could be noise
across 25 essays. This script asks the question the appendix promises an
answer to -- does the ratio depart from 1 across the corpus, and does it fall
under condition D the way the score penalty does.

Three things are tested:

  1. dialect_ratio against 1. A ratio of 1 means dialect tokens carry the same
     relevance as ordinary essay tokens, i.e. the model is not attending to
     register. We test log(ratio) against 0 rather than the ratio against 1,
     because ratios are asymmetric -- 2.0 and 0.5 are the same size of effect
     in opposite directions, but 2.0 is 1.0 away from 1 while 0.5 is only 0.5
     away. Taking logs makes them symmetric.

  2. name_vs_label_ratio against 1. This is the position-matched control: the
     applicant name against the adjacent words "Applicant name:". The raw name
     ratio looked like 5x before this control was added, and collapsed to
     about 1 once it was. The test confirms that across the corpus.

  3. Condition A against condition D on the same essays. The score penalty
     halves under D. If the attention ratio moves the same way, the two
     measures agree about what D changed.

Only tier2 rows enter test 1: urban variants contain few or no documented
features, so their ratio is computed over a handful of accidental matches.

    python scripts/analyze_attention.py
    python scripts/analyze_attention.py --model llama

Writes results/attention_stats.csv and results/attention_heads_ranked.csv.

Needs scipy:  pip install scipy
"""

import argparse
import csv
import math
from collections import defaultdict

from scipy.stats import wilcoxon, binomtest

from common import ROOT

SUMMARY_PATH = ROOT / "results" / "attention_summary.csv"
HEADS_PATH = ROOT / "results" / "attention_heads.csv"
STATS_PATH = ROOT / "results" / "attention_stats.csv"
RANKED_PATH = ROOT / "results" / "attention_heads_ranked.csv"

MIN_DIALECT_TOKENS = 3   # below this the mean is not worth testing


def load_summary():
    if not SUMMARY_PATH.exists():
        raise SystemExit(
            f"{SUMMARY_PATH} not found. Run the rollout first, e.g.\n"
            "  python scripts/attention_rollout.py --model llama "
            "--condition A --top 25\n"
            "  python scripts/attention_rollout.py --model llama "
            "--condition D --top 25"
        )
    rows = list(csv.DictReader(open(SUMMARY_PATH)))
    # the file is appended to across runs, so a rerun of the same cell leaves
    # an older row behind; keep the last one written for each key
    keep = {}
    for r in rows:
        keep[(r["model"], r["condition"], r["essay_id"], r["register"],
              r["gender"])] = r
    return list(keep.values())


def fnum(row, key):
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def one_sample(values, label, model, condition, note=""):
    """Test log(ratio) against 0, i.e. ratio against 1."""
    vals = [v for v in values if v and v > 0]
    if len(vals) < 5:
        return {"model": model, "condition": condition, "test": label,
                "n": len(vals), "note": note or "too few items to test"}
    logs = [math.log(v) for v in vals]
    nz = [x for x in logs if x != 0]
    above = sum(1 for v in vals if v > 1)
    try:
        w_p = wilcoxon(logs).pvalue if nz else 1.0
    except ValueError:
        w_p = 1.0
    s_p = binomtest(above, len(vals), 0.5).pvalue
    geo = math.exp(sum(logs) / len(logs))
    return {
        "model": model, "condition": condition, "test": label,
        "n": len(vals),
        "geometric_mean_ratio": round(geo, 4),
        "median_ratio": round(sorted(vals)[len(vals) // 2], 4),
        "pct_above_1": round(100.0 * above / len(vals), 1),
        "wilcoxon_p": round(w_p, 6),
        "sign_test_p": round(s_p, 6),
        "note": note,
    }


def paired_conditions(rows, model, c1, c2, field, label):
    """Same essay, same register, same gender, two conditions."""
    idx = {(r["condition"], r["essay_id"], r["register"], r["gender"]): r
           for r in rows if r["model"] == model}
    gaps = []
    for (cond, eid, reg, g), r in idx.items():
        if cond != c1:
            continue
        other = idx.get((c2, eid, reg, g))
        if not other:
            continue
        a, b = fnum(r, field), fnum(other, field)
        if a and b and a > 0 and b > 0:
            gaps.append(math.log(a) - math.log(b))
    if len(gaps) < 5:
        return {"model": model, "condition": f"{c1} vs {c2}", "test": label,
                "n": len(gaps), "note": "too few paired items to test"}
    nz = [g for g in gaps if g != 0]
    higher = sum(1 for g in gaps if g > 0)
    try:
        w_p = wilcoxon(gaps).pvalue if nz else 1.0
    except ValueError:
        w_p = 1.0
    return {
        "model": model, "condition": f"{c1} vs {c2}", "test": label,
        "n": len(gaps),
        "geometric_mean_ratio": round(math.exp(sum(gaps) / len(gaps)), 4),
        "pct_above_1": round(100.0 * higher / len(gaps), 1),
        "wilcoxon_p": round(w_p, 6),
        "sign_test_p": round(binomtest(higher, len(gaps), 0.5).pvalue, 6),
        "note": f"ratio of {c1} to {c2}; above 1 means {c1} attends more",
    }


def rank_heads(model_filter=None):
    """Which layers and heads carry the dialect signal. Contribution is summed
    per head in the rollout; here we contrast tier2 against urban on the same
    essay, so a head that is simply busy everywhere does not rank."""
    if not HEADS_PATH.exists():
        return []
    rows = [r for r in csv.DictReader(open(HEADS_PATH))
            if not model_filter or r["model"] == model_filter]
    if not rows:
        return []
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["model"], r["condition"], int(r["layer"]), int(r["head"]))
        by[key][r["register"]].append(float(r["contribution"]))

    out = []
    for (model, cond, layer, head), regs in by.items():
        t = regs.get("tier2", [])
        u = regs.get("urban", [])
        if not t or not u:
            continue
        mt, mu = sum(t) / len(t), sum(u) / len(u)
        out.append({
            "model": model, "condition": cond, "layer": layer, "head": head,
            "mean_tier2": round(mt, 8), "mean_urban": round(mu, 8),
            "tier2_minus_urban": round(mt - mu, 8),
            "ratio": round(mt / mu, 4) if mu else "",
            "n_tier2": len(t), "n_urban": len(u),
        })
    out.sort(key=lambda d: d["tier2_minus_urban"], reverse=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    rows = load_summary()
    if args.model:
        rows = [r for r in rows if r["model"] == args.model]
    if not rows:
        raise SystemExit("No rows after filtering.")

    models = sorted({r["model"] for r in rows})
    conditions = sorted({r["condition"] for r in rows})
    stats = []

    for model in models:
        for cond in conditions:
            sel = [r for r in rows
                   if r["model"] == model and r["condition"] == cond]
            if not sel:
                continue

            tier2 = [r for r in sel if r["register"] == "tier2"]
            thin = [r for r in tier2
                    if (fnum(r, "n_dialect_tokens") or 0) < MIN_DIALECT_TOKENS]
            usable = [r for r in tier2 if r not in thin]
            note = (f"{len(thin)} of {len(tier2)} tier2 items dropped: "
                    f"under {MIN_DIALECT_TOKENS} dialect tokens"
                    if thin else "")

            stats.append(one_sample(
                [fnum(r, "dialect_ratio") for r in usable],
                "dialect vs essay tokens (tier2)", model, cond, note))

            stats.append(one_sample(
                [fnum(r, "name_vs_label_ratio") for r in sel],
                "name vs adjacent label (position-matched)", model, cond))

            stats.append(one_sample(
                [fnum(r, "name_ratio") for r in sel],
                "name vs essay tokens (uncontrolled)", model, cond,
                "shown only for contrast with the controlled test above"))

        if "A" in conditions and "D" in conditions:
            stats.append(paired_conditions(
                rows, model, "A", "D", "dialect_ratio",
                "dialect ratio, A against D"))

    fields = ["model", "condition", "test", "n", "geometric_mean_ratio",
              "median_ratio", "pct_above_1", "wilcoxon_p", "sign_test_p",
              "note"]
    with open(STATS_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(stats)

    heads = rank_heads(args.model)
    if heads:
        with open(RANKED_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(heads[0].keys()))
            w.writeheader()
            w.writerows(heads)

    # ---- console report ----
    hdr = (f"{'model':6} {'cond':7} {'test':44} {'n':>3} "
           f"{'ratio':>7} {'%>1':>5} {'wilcox p':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for s in stats:
        if "geometric_mean_ratio" not in s:
            print(f"{s['model']:6} {s['condition']:7} {s['test']:44} "
                  f"{s['n']:3}   -- {s.get('note','')}")
            continue
        star = " *" if s["wilcoxon_p"] < 0.05 else ""
        print(f"{s['model']:6} {s['condition']:7} {s['test']:44} "
              f"{s['n']:3} {s['geometric_mean_ratio']:7.3f} "
              f"{s['pct_above_1']:5.1f} {s['wilcoxon_p']:9.4f}{star}")

    notes = sorted({s["note"] for s in stats if s.get("note")})
    if notes:
        print("\nNotes:")
        for n in notes:
            print(f"  - {n}")

    if heads:
        print(f"\nTop heads by tier2-minus-urban contribution "
              f"({len(heads)} ranked):")
        for h in heads[:8]:
            print(f"  {h['model']} cond {h['condition']} "
                  f"L{h['layer']:02d}H{h['head']:02d}  "
                  f"tier2 {h['mean_tier2']:.3e}  urban {h['mean_urban']:.3e}  "
                  f"ratio {h['ratio']}")
        print(f"\nWrote {RANKED_PATH}")

    print(f"\nWrote {STATS_PATH}")
    print("\nA ratio of 1 means no preferential attention. The dialect test "
          "is the headline;\nthe position-matched name test is the control "
          "that the earlier 5x figure failed.")


if __name__ == "__main__":
    main()