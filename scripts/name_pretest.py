"""
Name pretest: ask each evaluator model what it infers from a first name.

Designed for Google Colab (free T4 GPU). Run ONE model per session,
then restart the runtime before running the other, to free GPU memory.

Usage (from the repo root):
    python scripts/name_pretest.py --model llama
    python scripts/name_pretest.py --model qwen

Output is appended to metadata/name_pretest_model.csv.
If Colab disconnects, just rerun: finished (model, name, run) rows are skipped.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Settings. Record these in the README. Use the SAME models and precision
# later for scoring and for the attention analysis.
# ---------------------------------------------------------------------------
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}
N_RUNS = 5            # samples per name, to see how stable the guesses are
TEMPERATURE = 0.7     # >0 so the 5 runs can differ; 0 would repeat one answer
TOP_P = 1.0
MAX_NEW_TOKENS = 150
SEED = 42

PROMPT_PATH = Path("prompts/02_name_pretest.md")
CANDIDATES_PATH = Path("metadata/name_candidates.csv")
OUT_PATH = Path("metadata/name_pretest_model.csv")

FIELDS = [
    "model", "model_id", "name", "intended_gender", "type", "run",
    "gender", "region_of_india", "religion", "urban_or_rural",
    "family_income", "confidence", "parse_ok", "raw",
]


def load_model(model_id):
    # 4-bit quantisation so an 8B model fits on a free 15 GB T4.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    return tok, model


def ask(tok, model, prompt_text, run):
    torch.manual_seed(SEED + run)
    messages = [{"role": "user", "content": prompt_text}]
    input_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()


def parse(text):
    """Pull the first {...} block out of the reply and read it as JSON."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return {k: str(v).strip().lower() for k, v in data.items()}


def already_done():
    if not OUT_PATH.exists():
        return set()
    with open(OUT_PATH) as f:
        return {(r["model"], r["name"], r["run"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS.keys(), required=True)
    args = ap.parse_args()

    model_id = MODELS[args.model]
    template = PROMPT_PATH.read_text()
    candidates = list(csv.DictReader(open(CANDIDATES_PATH)))
    done = already_done()

    print(f"Loading {model_id} ...")
    tok, model = load_model(model_id)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_PATH.exists()

    with open(OUT_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()

        for c in candidates:
            prompt_text = template.replace("{NAME}", c["name"])
            for run in range(1, N_RUNS + 1):
                if (args.model, c["name"], str(run)) in done:
                    continue

                raw = ask(tok, model, prompt_text, run)
                parsed = parse(raw)

                row = {
                    "model": args.model,
                    "model_id": model_id,
                    "name": c["name"],
                    "intended_gender": c["intended_gender"],
                    "type": c["type"],
                    "run": run,
                    "parse_ok": "yes" if parsed else "no",
                    "raw": raw.replace("\n", " "),
                }
                if parsed:
                    for k in ["gender", "region_of_india", "religion",
                              "urban_or_rural", "family_income", "confidence"]:
                        row[k] = parsed.get(k, "")
                w.writerow(row)
                f.flush()  # write each row immediately in case Colab dies

                print(f"{args.model} | {c['name']:<10} run {run} | "
                      f"gender={row.get('gender', '?')} "
                      f"income={row.get('family_income', '?')} "
                      f"parse={row['parse_ok']}")

    print("Done.")


if __name__ == "__main__":
    main()
