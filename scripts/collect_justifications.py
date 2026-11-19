"""
Collect the one-sentence justifications the evaluators write alongside each
score, for every item and both models.

Why this script exists separately from score_logits.py: the distributional
measure never generates any text. It reads the probability distribution over
the score token and stops. That is the right way to measure the score, but it
leaves us with no justifications to analyse. results/scores.csv does contain
justifications, but only for LLaMA, and only for conditions A and C in full --
there is no Qwen text at all, and no condition D. Since the gender effect we
want to explain is largest for Qwen under condition B, we have to generate the
text we are missing.

    python scripts/collect_justifications.py --model llama
    python scripts/collect_justifications.py --model qwen

Greedy decoding by default, one pass per item, so the output is deterministic
and matches the scoring measure's logic: no sampling, no averaging over runs.
Pass --runs 5 --temperature 0.7 if you want sampled text instead.

100 items x 4 conditions = 400 generations per model. Slower than the logit
pass because tokens are actually produced: roughly 30-45 minutes on a free T4.

Resumable: rows are written as they are produced and finished ones skipped.
"""

import argparse
import csv
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import (ANSWER_PREFIX, CONDITIONS, MODELS, ROOT, build_prompt,
                    iter_items, parse_score)

OUT_PATH = ROOT / "results" / "justifications.csv"

FIELDS = [
    "model", "model_id", "essay_id", "register", "gender", "name",
    "condition", "run", "temperature", "score", "justification",
    "n_words", "parse_ok", "raw",
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


def generate(tok, model, prompt, temperature, max_new_tokens=220):
    """Force the reply to start with '{"score": ' exactly as the scoring pass
    does, so the justification we analyse belongs to the same forward context
    that produced the measured score."""
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    ) + ANSWER_PREFIX

    enc = tok(rendered, add_special_tokens=False, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[-1]

    kw = dict(max_new_tokens=max_new_tokens,
              pad_token_id=tok.eos_token_id)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        kw.update(do_sample=False)

    with torch.no_grad():
        out = model.generate(**enc, **kw)

    new = out[0][input_len:]
    # the prefix was fed in, so paste it back to get parseable JSON
    return ANSWER_PREFIX + tok.decode(new, skip_special_tokens=True)


def already_done():
    if not OUT_PATH.exists():
        return set()
    return {(r["model"], r["essay_id"], r["register"], r["gender"],
             r["condition"], r["run"])
            for r in csv.DictReader(open(OUT_PATH))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS.keys(), required=True)
    ap.add_argument("--conditions", default="A,B,C,D")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 means greedy and deterministic (default)")
    args = ap.parse_args()

    conditions = [c.strip().upper() for c in args.conditions.split(",")]
    for c in conditions:
        assert c in CONDITIONS, f"unknown condition {c}"

    model_id = MODELS[args.model]
    items = list(iter_items())
    done = already_done()
    total = len(items) * len(conditions) * args.runs

    print(f"{len(items)} items x {len(conditions)} conditions x "
          f"{args.runs} run(s) = {total} generations "
          f"({len(done)} already on file)")
    print(f"Loading {model_id} ...")
    tok, model = load_model(model_id)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_PATH.exists()
    n, t0 = 0, time.time()

    with open(OUT_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()

        for cond in conditions:
            for item in items:
                for run in range(1, args.runs + 1):
                    key = (args.model, item["essay_id"], item["register"],
                           item["gender"], cond, str(run))
                    if key in done:
                        continue

                    prompt = build_prompt(cond, item["name"], item["text"])
                    raw = generate(tok, model, prompt, args.temperature)
                    score, just = parse_score(raw)

                    w.writerow({
                        "model": args.model, "model_id": model_id,
                        "essay_id": item["essay_id"],
                        "register": item["register"],
                        "gender": item["gender"], "name": item["name"],
                        "condition": cond, "run": run,
                        "temperature": args.temperature,
                        "score": score if score else "",
                        "justification": just,
                        "n_words": len(just.split()),
                        "parse_ok": "yes" if score else "no",
                        "raw": raw,
                    })
                    f.flush()

                    n += 1
                    if n % 25 == 0:
                        rate = n / (time.time() - t0)
                        print(f"  {n} done, {rate:.2f}/s")

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()