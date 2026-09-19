import csv, os, time, re
from pathlib import Path
import anthropic

MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0.3
MAX_ATTEMPTS = 5

BASE_DIR = Path("data/base")
LOG_PATH = Path("metadata/generation_log.csv")
PROMPT_PATH = Path("prompts/01_base_generation.md")

client = anthropic.Anthropic()

BANNED = [
    "cbse", "icse", "state board", "jee", "neet", "bitsat", "cet",
    "kota", "allen", "fiitjee", "aakash", "coaching institute",
    "english medium", "english-medium", "regional medium",
    "vernacular", "mother tongue",
    "rupee", "rupees", "lakh", "crore", "₹",
    "brother", "sister", "mother", "father", "son", "daughter",
    "himself", "herself",
]
BANNED_PRONOUNS = [r"\bhe\b", r"\bshe\b", r"\bhis\b", r"\bher\b", r"\bhim\b"]


def check_leaks(text):
    """Return list of problems found. Empty list = clean."""
    low = text.lower()
    found = [w for w in BANNED if w in low]
    found += [p for p in BANNED_PRONOUNS if re.search(p, low)]
    return found


def word_count(text):
    return len(text.split())


def generate(topic_desc, template):
    prompt = template.replace("[topic description]", topic_desc)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        temperature=TEMPERATURE,
        top_p=1.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def main():
    template = PROMPT_PATH.read_text()
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    topics = list(csv.DictReader(open("data/topics.csv")))
    log_rows = []

    for row in topics:
        essay_id = row["essay_id"]
        outfile = BASE_DIR / f"{essay_id}.txt"

        if outfile.exists():
            print(f"skip {essay_id} (exists)")
            continue

        for attempt in range(1, MAX_ATTEMPTS + 1):
            text = generate(row["topic_description"], template)
            wc = word_count(text)
            leaks = check_leaks(text)

            ok_wc = 470 <= wc <= 510
            ok_leaks = len(leaks) == 0

            print(f"{essay_id} attempt {attempt}: {wc} words, "
                  f"leaks={leaks if leaks else 'none'}")

            if ok_wc and ok_leaks:
                outfile.write_text(text)
                log_rows.append({
                    "essay_id": essay_id,
                    "theme": row["theme"],
                    "generator_model": MODEL,
                    "temperature": TEMPERATURE,
                    "top_p": 1.0,
                    "date": time.strftime("%Y-%m-%d"),
                    "attempt_number": attempt,
                    "word_count": wc,
                    "passed_leak_check": "yes",
                    "notes": "",
                })
                break
            time.sleep(1)
        else:
            print(f"FAILED {essay_id} after {MAX_ATTEMPTS} attempts")
            log_rows.append({
                "essay_id": essay_id, "theme": row["theme"],
                "generator_model": MODEL, "temperature": TEMPERATURE,
                "top_p": 1.0, "date": time.strftime("%Y-%m-%d"),
                "attempt_number": MAX_ATTEMPTS, "word_count": wc,
                "passed_leak_check": "no",
                "notes": f"unresolved: {leaks}",
            })

    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        if write_header:
            w.writeheader()
        w.writerows(log_rows)


if __name__ == "__main__":
    main()