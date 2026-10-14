"""
Statistics on the scoring results.

    pip install pandas statsmodels scipy

    python scripts/analyze_scores.py                  # logit scores (default)
    python scripts/analyze_scores.py --source sampled # generated integers

Produces, with a suffix naming the source:
  results/analysis_cellmeans_<source>.csv
  results/analysis_effects_<source>.csv
  results/analysis_iffr_<source>.csv
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

# A gap this small is real but trivial; reported separately from any gap.
MEANINGFUL_GAP = 0.10

warnings.filterwarnings("ignore")


def load(source):
    if source == "logit":
        df = pd.read_csv(OUT / "scores_logit.csv")
        df = df[df.expected_score.notna()].copy()
        df["score"] = df["expected_score"].astype(float)
        df["run"] = 1
        # how much probability sat on a valid score at all
        print(f"valid_mass: min {df.valid_mass.min():.3f}, "
              f"median {df.valid_mass.median():.3f}")
        if (df.valid_mass < 0.5).any():
            print("WARNING: some items put under half their probability on a "
                  "1-10 score. Check those prompts.")
    else:
        df = pd.read_csv(OUT / "scores.csv")
        df = df[df.parse_ok == "yes"].copy()
        df["score"] = df["score"].astype(float)

    # tier2 and F are the reference levels, so a positive coefficient means
    # urban (or male) scored higher
    df["is_urban"] = (df.register == "urban").astype(int)
    df["is_male"] = (df.gender == "M").astype(int)
    return df


def cell_means(df, source):
    g = (df.groupby(["model", "condition", "register", "gender"])["score"]
           .agg(["mean", "std", "count"]).round(4).reset_index())
    g.to_csv(OUT / f"analysis_cellmeans_{source}.csv", index=False)
    print("\n=== Cell means ===")
    print(g.to_string(index=False))


def effects(df, source):
    rows = []
    for (m, c), sub in df.groupby(["model", "condition"]):
        if sub.essay_id.nunique() < 5:
            continue
        # essay as a random intercept: all four cells share one base text,
        # so each essay acts as its own control
        fit = smf.mixedlm("score ~ is_urban * is_male", sub,
                          groups=sub["essay_id"]).fit(reml=False)
        for term in ["is_urban", "is_male", "is_urban:is_male"]:
            if term in fit.params:
                rows.append({
                    "model": m, "condition": c, "term": term,
                    "estimate": round(fit.params[term], 4),
                    "std_err": round(fit.bse[term], 4),
                    "p_value": round(fit.pvalues[term], 6),
                    "n_obs": len(sub),
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"analysis_effects_{source}.csv", index=False)
    print("\n=== Mixed-effects estimates ===")
    print("is_urban = register penalty, is_male = gender effect, "
          "is_urban:is_male = interaction")
    print(out.to_string(index=False))


def iffr(df, source):
    """Instruction-Following Failure Rate, after Farajidizaji & Raina (2026).

    Conditions A, C and D state that phrasing must not affect the score, so
    any pair where the urban version outscores the content-identical tier2
    version is a failure to follow that instruction. Condition B gives no
    such instruction: its value is a baseline, not a failure rate.
    """
    rows = []
    wide = (df.pivot_table(index=["model", "condition", "essay_id",
                                  "gender", "run"],
                           columns="register", values="score")
              .dropna().reset_index())
    wide["gap"] = wide["urban"] - wide["tier2"]

    for (m, c), sub in wide.groupby(["model", "condition"]):
        try:
            p = wilcoxon(sub["urban"], sub["tier2"]).pvalue
        except ValueError:
            p = float("nan")
        rows.append({
            "model": m, "condition": c, "n_pairs": len(sub),
            "iffr_any": round((sub.gap > 0).mean(), 3),
            "iffr_meaningful": round((sub.gap > MEANINGFUL_GAP).mean(), 3),
            "pct_tier2_higher": round((sub.gap < 0).mean(), 3),
            "mean_gap": round(sub.gap.mean(), 4),
            "median_gap": round(sub.gap.median(), 4),
            "wilcoxon_p": round(p, 6),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"analysis_iffr_{source}.csv", index=False)
    print("\n=== Instruction-Following Failure Rate ===")
    print("iffr_any = share of content-identical pairs where urban outscored "
          f"tier2; iffr_meaningful uses a gap above {MEANINGFUL_GAP}")
    print("(condition B has no such instruction: baseline, not a failure rate)")
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["logit", "sampled"], default="logit")
    args = ap.parse_args()

    df = load(args.source)
    print(f"\n{len(df)} scores from {df.essay_id.nunique()} essays, "
          f"models: {sorted(df.model.unique())}")
    cell_means(df, args.source)
    effects(df, args.source)
    iffr(df, args.source)


if __name__ == "__main__":
    main()