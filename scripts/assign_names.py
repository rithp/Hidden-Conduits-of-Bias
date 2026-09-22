"""
Assign one male and one female name to each of the 25 base essays.

Each name is used exactly 5 times. The same name is used for both
register versions of an essay (that happens later, not here).

Run from anywhere:
    python scripts/assign_names.py

Reads:   metadata/name_pool.csv, data/topics.csv
Writes:  metadata/name_assignment.csv
"""

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = ROOT / "metadata" / "name_pool.csv"
TOPICS_PATH = ROOT / "data" / "topics.csv"
OUT_PATH = ROOT / "metadata" / "name_assignment.csv"

SEED = 42
USES_PER_NAME = 5


def main():
    random.seed(SEED)

    pool = list(csv.DictReader(open(POOL_PATH)))
    male = [r["name"] for r in pool if r["gender"] == "M"]
    female = [r["name"] for r in pool if r["gender"] == "F"]
    essays = [r["essay_id"] for r in csv.DictReader(open(TOPICS_PATH))]

    assert len(male) == 5, f"expected 5 male names, got {len(male)}"
    assert len(female) == 5, f"expected 5 female names, got {len(female)}"
    assert len(essays) == 25, f"expected 25 essays, got {len(essays)}"

    male_slots = male * USES_PER_NAME
    female_slots = female * USES_PER_NAME
    random.shuffle(male_slots)
    random.shuffle(female_slots)

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["essay_id", "male_name", "female_name", "seed"])
        for e, m, fe in zip(essays, male_slots, female_slots):
            w.writerow([e, m, fe, SEED])

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()