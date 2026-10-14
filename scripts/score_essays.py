"""
Score every essay variant with one evaluator model.

Run one model per Colab session, restarting the runtime in between:
    python scripts/score_essays.py --model llama
    python scripts/score_essays.py --model qwen

Options:
    --runs N          samples per item per condition (default 5)
    --temperature T   default 0.7; use 0.0 for a single deterministic pass
    --conditions A,C  subset of conditions, in the order they are run

Results append to results/scores.csv after every single call, and finished
(model, essay, register, gender, condition, run, temperature) rows are
skipped, so a disconnected session is resumed by rerunning the same command.

Scale: 100 items x 4 conditions x 5 runs = 2000 calls per model. On a free
T4 that is roughly 3-6 hours per model. Conditions run in the order given,
so A (the primary test) completes first and partial results stay usable.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import MODELS, CONDITIONS, SCORES_PATH, build_prompt, iter_items, parse_score

MAX_NEW_TOKENS = 160
TOP_P = 1.0
SEED = 42

FIELDS = [
    "model", "model_id", "essay_id", "register", "gender", "name",
    "condition", "run", "temperature", "score", "justification",
    "parse_ok", "raw",
]


def load_model(model_id):
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


def ask(tok, model, prompt, temperature, run):
    torch.manual_seed(SEED + run)
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=TOP_P if temperature > 0 else None,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][input_len:], skip_special_tokens=True).strip()


def already_done():
    if not SCORES_PATH.exists():
        return set()
    with open(SCORES_PATH) as f:
        return {
            (r["model"], r["essay_id"], r["register"], r["gender"],
             r["condition"], r["run"], r["temperature"])
            for r in csv.DictReader(f)
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS.keys(), required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--conditions", default="A,C,B,D")
    args = ap.parse_args()

    conditions = [c.strip().upper() for c in args.conditions.split(",")]
    for c in conditions:
        assert c in CONDITIONS, f"unknown condition {c}"

    model_id = MODELS[args.model]
    temp_s = f"{args.temperature:.1f}"
    done = already_done()

    items = list(iter_items())
    total = len(items) * len(conditions) * args.runs
    print(f"{len(items)} items x {len(conditions)} conditions x "
          f"{args.runs} runs = {total} calls ({len(done)} already on file)")

    print(f"Loading {model_id} ...")
    tok, model = load_model(model_id)

    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SCORES_PATH.exists()
    n = 0
    t0 = time.time()

    with open(SCORES_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()

        for cond in conditions:
            for item in items:
                for run in range(1, args.runs + 1):
                    key = (args.model, item["essay_id"], item["register"],
                           item["gender"], cond, str(run), temp_s)
                    if key in done:
                        continue

                    prompt = build_prompt(cond, item["name"], item["text"])
                    raw = ask(tok, model, prompt, args.temperature, run)
                    score, just = parse_score(raw)

                    w.writerow({
                        "model": args.model, "model_id": model_id,
                        "essay_id": item["essay_id"],
                        "register": item["register"],
                        "gender": item["gender"], "name": item["name"],
                        "condition": cond, "run": run,
                        "temperature": temp_s,
                        "score": score if score is not None else "",
                        "justification": just.replace("\n", " "),
                        "parse_ok": "yes" if score is not None else "no",
                        "raw": raw.replace("\n", " ")[:600],
                    })
                    f.flush()

                    n += 1
                    if n % 20 == 0:
                        rate = n / (time.time() - t0)
                        print(f"[{cond}] {n} done, {rate:.2f}/s, "
                              f"last: {item['essay_id']} {item['register']} "
                              f"{item['gender']} -> {score}")

    print(f"Done. {n} new calls in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
