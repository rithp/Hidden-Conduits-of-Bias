"""
Statistics on results/scores.csv.

    pip install pandas statsmodels scipy
    python scripts/analyze_scores.py

Produces:
  results/analysis_cellmeans.csv   mean score per model/condition/cell
  results/analysis_effects.csv     mixed-effects estimates
  results/analysis_iffr.csv        instruction-following failure rates
"""

import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "results" / "scores.csv"
OUT = ROOT / "results"

warnings.filterwarnings("ignore")


def load():
    df = pd.read_csv(SCORES)
    df = df[df.parse_ok == "yes"].copy()
    df["score"] = df["score"].astype(float)
    # tier2 and F are the reference levels, so a positive coefficient means
    # urban (or male) scored higher
    df["is_urban"] = (df.register == "urban").astype(int)
    df["is_male"] = (df.gender == "M").astype(int)
    return df


def cell_means(df):
    g = (df.groupby(["model", "condition", "register", "gender"])["score"]
           .agg(["mean", "std", "count"]).round(3).reset_index())
    g.to_csv(OUT / "analysis_cellmeans.csv", index=False)
    print("\n=== Cell means ===")
    print(g.to_string(index=False))
    return g


def effects(df):
    rows = []
    for (m, c), sub in df.groupby(["model", "condition"]):
        if sub.essay_id.nunique() < 5:
            continue
        # essay as a random intercept: each essay is its own control, since
        # all four cells come from the same base text
        fit = smf.mixedlm("score ~ is_urban * is_male", sub,
                          groups=sub["essay_id"]).fit(reml=False)
        for term in ["is_urban", "is_male", "is_urban:is_male"]:
            if term in fit.params:
                rows.append({
                    "model": m, "condition": c, "term": term,
                    "estimate": round(fit.params[term], 4),
                    "std_err": round(fit.bse[term], 4),
                    "p_value": round(fit.pvalues[term], 5),
                    "n_obs": len(sub),
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "analysis_effects.csv", index=False)
    print("\n=== Mixed-effects estimates ===")
    print("is_urban = register penalty, is_male = gender effect, "
          "is_urban:is_male = interaction")
    print(out.to_string(index=False))
    return out


def iffr(df):
    """Instruction-Following Failure Rate, after Farajidizaji & Raina (2026).

    Conditions A, C and D tell the model that phrasing must not affect the
    score. So any essay pair where the urban version outscores the content-
    identical tier2 version is a failure to follow that instruction. Under
    condition B no such instruction is given, so its value is a baseline,
    not a failure rate.
    """
    rows = []
    keys = ["model", "condition", "essay_id", "gender", "run"]
    wide = (df.pivot_table(index=keys, columns="register", values="score")
              .dropna().reset_index())
    wide["gap"] = wide["urban"] - wide["tier2"]

    for (m, c), sub in wide.groupby(["model", "condition"]):
        nonzero = sub[sub.gap != 0]
        try:
            p = wilcoxon(sub["urban"], sub["tier2"]).pvalue
        except ValueError:
            p = float("nan")
        rows.append({
            "model": m, "condition": c,
            "n_pairs": len(sub),
            "iffr": round((sub.gap > 0).mean(), 3),
            "pct_tier2_higher": round((sub.gap < 0).mean(), 3),
            "pct_equal": round((sub.gap == 0).mean(), 3),
            "mean_gap": round(sub.gap.mean(), 3),
            "mean_gap_when_differs": round(nonzero.gap.mean(), 3) if len(nonzero) else 0.0,
            "wilcoxon_p": round(p, 5),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "analysis_iffr.csv", index=False)
    print("\n=== Instruction-Following Failure Rate ===")
    print("iffr = share of content-identical pairs where urban outscored tier2")
    print("(condition B has no such instruction: baseline, not a failure rate)")
    print(out.to_string(index=False))
    return out


def main():
    df = load()
    print(f"{len(df)} parsed scores from {df.essay_id.nunique()} essays, "
          f"models: {sorted(df.model.unique())}")
    cell_means(df)
    effects(df)
    iffr(df)


if __name__ == "__main__":
    main()
