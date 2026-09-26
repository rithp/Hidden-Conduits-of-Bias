"""
Summarise the name pretest and pick the final 5 male + 5 female names.

Run after name_pretest.py has finished for BOTH models:
    python scripts/select_names.py

Writes:
    metadata/name_pretest_summary.csv   one row per name, all metrics
    metadata/name_pool.csv              the final 10 names

The selection is a starting point. You may override it by hand, but write
the reason in the reason_selected column of name_pool.csv.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "metadata" / "name_pretest_model.csv"
SUMMARY_PATH = ROOT / "metadata" / "name_pretest_summary.csv"
POOL_PATH = ROOT / "metadata" / "name_pool.csv"

POOL_SIZE = 5                  # per gender
MIN_GENDER_ACCURACY = 0.9      # must hold for EACH model separately
BALANCE_WARN = 0.2             # warn if pools differ by more than this
NEUTRAL_FIELDS = ["region_of_india", "urban_or_rural"]  # income dropped: anchors showed it carries no signal
GENDER_WORD = {"M": "male", "F": "female"}


def clean(value):
    """Turn a messy answer into one value.
    'male / unsure'          -> 'male'    (only one real option named)
    'male / female / unsure' -> 'unsure'  (did not commit)
    'middle / low / unsure'  -> 'unsure'
    """
    parts = [p.strip() for p in value.split("/") if p.strip()]
    committed = [p for p in parts if p != "unsure"]
    if len(committed) == 1:
        return committed[0]
    return "unsure"


def load_rows():
    rows = list(csv.DictReader(open(IN_PATH)))
    rows = [r for r in rows if r["parse_ok"] == "yes"]
    for r in rows:
        for fld in ["gender", "region_of_india", "urban_or_rural",
                    "family_income", "religion"]:
            r[fld] = clean(r.get(fld, "") or "")
    return rows


def summarise(rows):
    by_name = defaultdict(list)
    for r in rows:
        by_name[r["name"]].append(r)

    summary = []
    for name, rs in by_name.items():
        intended = rs[0]["intended_gender"]
        s = {"name": name, "intended_gender": intended, "type": rs[0]["type"],
             "n_answers": len(rs)}

        # gender accuracy, per model; keep the worse of the two
        accs = []
        for m in sorted({r["model"] for r in rs}):
            mr = [r for r in rs if r["model"] == m]
            acc = sum(r["gender"] == GENDER_WORD[intended] for r in mr) / len(mr)
            s[f"gender_acc_{m}"] = round(acc, 2)
            accs.append(acc)
        s["gender_acc_min"] = round(min(accs), 2)

        # how often each model said "unsure" on the fields we want neutral
        unsure = []
        for fld in NEUTRAL_FIELDS:
            rate = sum(r[fld] == "unsure" for r in rs) / len(rs)
            s[f"unsure_{fld}"] = round(rate, 2)
            s[f"modal_{fld}"] = Counter(r[fld] for r in rs).most_common(1)[0][0]
            unsure.append(rate)
        s["neutrality_score"] = round(sum(unsure) / len(unsure), 2)

        s["share_high_income"] = round(
            sum(r["family_income"] == "high" for r in rs) / len(rs), 2)
        s["share_urban"] = round(
            sum(r["urban_or_rural"] == "urban" for r in rs) / len(rs), 2)
        summary.append(s)
    return summary


def check_anchors(summary):
    """If the models say 'unsure' even for strongly coded names,
    then 'unsure' on the candidates tells us nothing."""
    print("\n--- Anchor check ---")
    anchors = [s for s in summary if s["type"] == "anchor"]
    for a in anchors:
        print(f"{a['name']:<10} region={a['modal_region_of_india']:<10} "
              f"urban={a['modal_urban_or_rural']:<8} "
              f"neutrality={a['neutrality_score']}")
    if anchors:
        avg = sum(a["neutrality_score"] for a in anchors) / len(anchors)
        if avg > 0.6:
            print("WARNING: models mostly answer 'unsure' even for anchors.")
            print("'Unsure' on candidates is then weak evidence of neutrality.")
            print("Consider rewording the prompt to force a best guess.")


def pick(summary, gender):
    pool = [s for s in summary
            if s["type"] == "candidate"
            and s["intended_gender"] == gender
            and s["gender_acc_min"] >= MIN_GENDER_ACCURACY]
    pool.sort(key=lambda s: s["neutrality_score"], reverse=True)
    if len(pool) < POOL_SIZE:
        print(f"WARNING: only {len(pool)} {gender} names passed the "
              f"gender-accuracy bar. Add more candidates and rerun.")
    return pool[:POOL_SIZE]


def balance_check(male, female):
    print("\n--- Balance check (male pool vs female pool) ---")
    for key in ["share_high_income", "share_urban", "neutrality_score"]:
        m = sum(s[key] for s in male) / max(len(male), 1)
        f = sum(s[key] for s in female) / max(len(female), 1)
        flag = "  <-- WARNING" if abs(m - f) > BALANCE_WARN else ""
        print(f"{key:<20} male={m:.2f} female={f:.2f}{flag}")


def main():
    rows = load_rows()
    summary = summarise(rows)

    with open(SUMMARY_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Wrote {SUMMARY_PATH}")

    check_anchors(summary)
    male = pick(summary, "M")
    female = pick(summary, "F")
    balance_check(male, female)

    with open(POOL_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "gender", "gender_acc_min",
                    "neutrality_score", "reason_selected"])
        for s in male + female:
            w.writerow([s["name"], s["intended_gender"], s["gender_acc_min"],
                        s["neutrality_score"],
                        "auto: passed gender bar, top neutrality score"])
    print(f"\nWrote {POOL_PATH}")
    print("Final pool:", [s["name"] for s in male], [s["name"] for s in female])


if __name__ == "__main__":
    main()