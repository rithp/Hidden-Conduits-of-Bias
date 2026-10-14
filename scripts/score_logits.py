"""
Score every item by reading the model's probability distribution over the
score token, instead of sampling one integer.

The assistant turn is forced to begin '{"score": ', so the next token the
model predicts is the score itself. Taking the probability of each digit
and computing the expected value gives a continuous score that responds to
small shifts -- which sampled integers cannot do.

    python scripts/score_logits.py --model llama
    python scripts/score_logits.py --model qwen

One forward pass per item, no generation, deterministic. 100 items x 4
conditions = 400 passes per model, roughly 10-20 minutes on a free T4.

Resumable: rows are written immediately and finished ones are skipped.
"""

import argparse
import csv
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import ANSWER_PREFIX, CONDITIONS, MODELS, ROOT, build_prompt, iter_items

OUT_PATH = ROOT / "results" / "scores_logit.csv"

FIELDS = [
    "model", "model_id", "essay_id", "register", "gender", "name",
    "condition", "expected_score", "argmax_score", "valid_mass",
    "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
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


def digit_token_ids(tok, digit):
    """Ids that would render this digit here. Both the bare form and the
    space-prefixed form are collected, since tokenizers differ on which one
    follows a space."""
    ids = []
    for form in (digit, " " + digit):
        t = tok.encode(form, add_special_tokens=False)
        if len(t) == 1:
            ids.append(t[0])
    return ids or [tok.encode(digit, add_special_tokens=False)[0]]


def build_digit_map(tok):
    return {d: digit_token_ids(tok, str(d)) for d in range(1, 10)}


@torch.no_grad()
def score_distribution(tok, model, prompt, digit_map, zero_ids):
    """Returns (expected_score, argmax_score, valid_mass, per-score probs).

    '10' is two tokens, so the mass on token '1' is split: the model is run
    once more with '1' appended to see how much of it continues into '0'.
    """
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    ) + ANSWER_PREFIX

    ids = tok(rendered, add_special_tokens=False,
              return_tensors="pt")["input_ids"].to(model.device)
    probs = torch.softmax(model(ids).logits[0, -1].float(), dim=-1)

    raw = {d: sum(probs[i].item() for i in toks)
           for d, toks in digit_map.items()}

    # split the '1' mass between 1 and 10
    p_ten = 0.0
    if raw[1] > 1e-6:
        ids2 = tok(rendered + "1", add_special_tokens=False,
                   return_tensors="pt")["input_ids"].to(model.device)
        probs2 = torch.softmax(model(ids2).logits[0, -1].float(), dim=-1)
        p_zero = sum(probs2[i].item() for i in zero_ids)
        p_ten = raw[1] * p_zero
        raw[1] = raw[1] * (1 - p_zero)

    dist = {**raw, 10: p_ten}
    valid_mass = sum(dist.values())
    if valid_mass <= 0:
        return None, None, 0.0, dist

    norm = {k: v / valid_mass for k, v in dist.items()}
    expected = sum(k * v for k, v in norm.items())
    argmax = max(norm, key=norm.get)
    return expected, argmax, valid_mass, norm


def already_done():
    if not OUT_PATH.exists():
        return set()
    with open(OUT_PATH) as f:
        return {(r["model"], r["essay_id"], r["register"], r["gender"],
                 r["condition"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS.keys(), required=True)
    ap.add_argument("--conditions", default="A,C,B,D")
    args = ap.parse_args()

    conditions = [c.strip().upper() for c in args.conditions.split(",")]
    model_id = MODELS[args.model]
    items = list(iter_items())
    done = already_done()

    print(f"{len(items)} items x {len(conditions)} conditions = "
          f"{len(items) * len(conditions)} passes "
          f"({len(done)} already on file)")
    print(f"Loading {model_id} ...")
    tok, model = load_model(model_id)
    digit_map = build_digit_map(tok)
    zero_ids = digit_token_ids(tok, "0")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_PATH.exists()
    n, t0 = 0, time.time()

    with open(OUT_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()

        for cond in conditions:
            assert cond in CONDITIONS, f"unknown condition {cond}"
            for item in items:
                key = (args.model, item["essay_id"], item["register"],
                       item["gender"], cond)
                if key in done:
                    continue

                prompt = build_prompt(cond, item["name"], item["text"])
                exp, am, mass, dist = score_distribution(
                    tok, model, prompt, digit_map, zero_ids)

                row = {
                    "model": args.model, "model_id": model_id,
                    "essay_id": item["essay_id"],
                    "register": item["register"], "gender": item["gender"],
                    "name": item["name"], "condition": cond,
                    "expected_score": round(exp, 6) if exp else "",
                    "argmax_score": am if am else "",
                    "valid_mass": round(mass, 6),
                }
                for d in range(1, 11):
                    row[f"p{d}"] = round(dist.get(d, 0.0), 6)
                w.writerow(row)
                f.flush()

                n += 1
                if n % 25 == 0:
                    print(f"[{cond}] {n} done, "
                          f"{n / (time.time() - t0):.2f}/s, last "
                          f"{item['essay_id']} {item['register']} "
                          f"{item['gender']} -> {exp:.3f}")

    print(f"Done. {n} passes in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()