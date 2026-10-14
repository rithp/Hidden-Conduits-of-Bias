"""
Gradient-weighted attention rollout over the scoring forward pass.

What it does, in order:
  1. Reads results/scores.csv and picks the essays with the largest
     urban-minus-tier2 score gap under condition A.
  2. For each one, feeds the model the exact scoring prompt used during
     scoring, with the assistant turn forced to start with '{"score": '.
     The next token the model predicts IS the score, so the logit of that
     token is a clean scalar to differentiate.
  3. Backpropagates from that logit to every attention matrix, combines
     gradients with attention weights (Chefer et al. 2021), and rolls the
     result across layers with an identity term for the residual stream
     (Abnar & Zuidema 2020).
  4. Splits the resulting per-token relevance into three groups -- tokens
     inside a documented Indian English feature, the applicant name token,
     and the rest of the essay -- and reports the ratios.
  5. Writes a per-head table so individual heads can be named, plus an HTML
     heatmap for figures.

Usage:
    python scripts/attention_rollout.py --model llama --top 5

Requires attn_implementation="eager". FlashAttention and SDPA never
materialise the N x N matrix, so there is nothing to take gradients of.

Memory: attentions and their gradients are both L x H x N x N. For LLaMA
(32 layers, 32 heads) at ~900 tokens that is roughly 3.5 GB on top of the
model. It fits on a free T4 but with little room. If you hit OOM, run on an
L4 or A100, or lower --top and run in several sessions.
"""

import argparse
import csv
import html
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import (ANSWER_PREFIX, MODELS, ROOT, SCORES_PATH, build_prompt,
                    load_names)
from check_variants import FEATURE_PATTERNS

VAR_DIR = ROOT / "data" / "variants"
OUT_DIR = ROOT / "results" / "attention"
SUMMARY_PATH = ROOT / "results" / "attention_summary.csv"
CONDITION = "A"          # the constrained condition: the instruction says
                         # dialect must not matter, so this is where a
                         # dialect-attending head is a genuine failure
BASELINE_CONDITION = "B"  # run with --condition B to compare


def load_model(model_id):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        attn_implementation="eager",   # required: keeps the N x N matrices
    )
    model.eval()
    return tok, model


# ---------------------------------------------------------------------------
# picking which essays to look at
# ---------------------------------------------------------------------------

def pick_items(model_key, condition, top_n):
    """Essays whose tier2 version scored lowest relative to its urban version."""
    rows = [r for r in csv.DictReader(open(SCORES_PATH))
            if r["model"] == model_key and r["condition"] == condition
            and r["parse_ok"] == "yes"]
    if not rows:
        raise SystemExit(f"No scored rows for model={model_key} "
                         f"condition={condition}. Run score_essays.py first.")

    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["essay_id"], r["gender"], r["register"])].append(int(r["score"]))

    gaps = []
    for (eid, g, reg), _ in list(bucket.items()):
        if reg != "urban":
            continue
        urban = bucket.get((eid, g, "urban"), [])
        tier2 = bucket.get((eid, g, "tier2"), [])
        if not urban or not tier2:
            continue
        gap = sum(urban) / len(urban) - sum(tier2) / len(tier2)
        gaps.append((gap, eid, g))

    gaps.sort(reverse=True)
    print(f"Top gaps (urban minus tier2), condition {condition}:")
    for gap, eid, g in gaps[:top_n]:
        print(f"  {gap:+.2f}  {eid}  {g}")
    return gaps[:top_n]


def mean_score(model_key, condition, eid, reg, gender):
    scores = [int(r["score"]) for r in csv.DictReader(open(SCORES_PATH))
              if r["model"] == model_key and r["condition"] == condition
              and r["essay_id"] == eid and r["register"] == reg
              and r["gender"] == gender and r["parse_ok"] == "yes"]
    return round(sum(scores) / len(scores), 2) if scores else None


# ---------------------------------------------------------------------------
# the rollout itself
# ---------------------------------------------------------------------------

def rollout_relevance(tok, model, prompt, score_token_str):
    """Returns (tokens, offsets, relevance, head_table, rendered_text).

    relevance[i] is how much token i contributed to the predicted score,
    following Chefer et al. (2021) combined with Abnar & Zuidema (2020).
    """
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    ) + ANSWER_PREFIX

    enc = tok(rendered, return_offsets_mapping=True,
              add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    offsets = enc["offset_mapping"][0].tolist()
    n_tokens = input_ids.shape[-1]

    out = model(input_ids, output_attentions=True, use_cache=False)
    attns = out.attentions                      # tuple of (1, H, N, N)
    last_logits = out.logits[0, -1]             # distribution over the score

    # The score digit, as the model would emit it right after '{"score": '.
    cand = [score_token_str, " " + score_token_str]
    ids = []
    for c in cand:
        t = tok.encode(c, add_special_tokens=False)
        if len(t) == 1:
            ids.append(t[0])
    if not ids:
        ids = [tok.encode(score_token_str, add_special_tokens=False)[0]]
    target_id = max(ids, key=lambda i: last_logits[i].item())
    target = last_logits[target_id]

    grads = torch.autograd.grad(target, attns)

    eye = torch.eye(n_tokens, device=model.device, dtype=torch.float32)
    R = eye.clone()
    head_table = []

    for layer, (a, g) in enumerate(zip(attns, grads)):
        weighted = (g * a).clamp(min=0).float()[0]       # (H, N, N)
        head_table.append(weighted[:, -1, :].sum(dim=-1).detach().cpu())
        cam = weighted.mean(dim=0)                        # average over heads
        a_hat = cam + eye
        a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        R = a_hat @ R

    relevance = R[-1].detach().cpu().tolist()
    tokens = tok.convert_ids_to_tokens(input_ids[0].tolist())

    del out, attns, grads, R
    torch.cuda.empty_cache()
    return tokens, offsets, relevance, head_table, rendered


# ---------------------------------------------------------------------------
# mapping relevance back onto the essay
# ---------------------------------------------------------------------------

def feature_spans(text, base_offset):
    """Character spans of documented Indian English features, in rendered
    coordinates."""
    spans = []
    low = text.lower()
    for name, pat in FEATURE_PATTERNS.items():
        for m in re.finditer(pat, low):
            spans.append((base_offset + m.start(), base_offset + m.end(), name))
    return spans


def in_any(span_list, start, end):
    for s, e, name in span_list:
        if start < e and end > s:
            return name
    return None


def analyse(tokens, offsets, relevance, rendered, essay_text, name):
    essay_start = rendered.rfind(essay_text.strip()[:80])
    essay_end = essay_start + len(essay_text.strip())
    spans = feature_spans(essay_text, essay_start)

    name_marker = f"Applicant name: {name}"
    name_pos = rendered.rfind(name_marker)
    name_start = name_pos + len("Applicant name: ")
    name_end = name_start + len(name)

    rows, dialect, other, name_rel = [], [], [], []
    for tokn, (s, e), r in zip(tokens, offsets, relevance):
        group = "prompt"
        if name_start <= s < name_end:
            group = "name"
            name_rel.append(r)
        elif essay_start <= s < essay_end:
            feat = in_any(spans, s, e)
            group = f"dialect:{feat}" if feat else "essay"
            (dialect if feat else other).append(r)
        rows.append({"token": tokn, "char_start": s, "char_end": e,
                     "group": group, "relevance": round(r, 8)})

    def mean(x):
        return sum(x) / len(x) if x else 0.0

    stats = {
        "n_dialect_tokens": len(dialect),
        "n_essay_tokens": len(other),
        "rel_dialect_mean": mean(dialect),
        "rel_essay_mean": mean(other),
        "rel_name_mean": mean(name_rel),
        "dialect_ratio": round(mean(dialect) / mean(other), 3) if other and dialect else "",
        "name_ratio": round(mean(name_rel) / mean(other), 3) if other and name_rel else "",
    }
    return rows, stats


def write_heatmap(path, rows, rendered, title):
    vals = [r["relevance"] for r in rows if r["group"] != "prompt"]
    hi = max(vals) if vals else 1.0
    parts = []
    for r in rows:
        if r["group"] == "prompt":
            continue
        shade = min(r["relevance"] / hi, 1.0) if hi else 0
        text = html.escape(rendered[r["char_start"]:r["char_end"]]) or "&nbsp;"
        border = ("border-bottom:2px solid #c00;"
                  if r["group"].startswith("dialect") else "")
        if r["group"] == "name":
            border = "border-bottom:2px solid #06c;"
        parts.append(
            f'<span title="{html.escape(r["group"])} {r["relevance"]:.2e}" '
            f'style="background:rgba(220,40,40,{shade:.3f});{border}">{text}</span>'
        )
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<style>body{font:16px/1.9 Georgia,serif;max-width:46em;margin:3em auto;"
        "padding:0 1em}h1{font-size:1.1em;font-family:system-ui}"
        ".key{font:13px system-ui;color:#555;margin-bottom:2em}</style>"
        f"<h1>{html.escape(title)}</h1>"
        "<p class='key'>Shading: relevance to the predicted score. "
        "Red underline: documented Indian English feature. "
        "Blue underline: applicant name.</p><p>"
        + "".join(parts) + "</p>"
    )


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS.keys(), required=True)
    ap.add_argument("--top", type=int, default=5,
                    help="how many essays to analyse (both registers each)")
    ap.add_argument("--condition", default=CONDITION)
    args = ap.parse_args()

    picks = pick_items(args.model, args.condition, args.top)
    names = load_names()
    model_id = MODELS[args.model]

    print(f"\nLoading {model_id} with eager attention ...")
    tok, model = load_model(model_id)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, head_rows = [], []

    for gap, eid, gender in picks:
        for reg in ["urban", "tier2"]:
            text = (VAR_DIR / f"{eid}__{reg}.txt").read_text()
            name = names[eid][gender]
            score = mean_score(args.model, args.condition, eid, reg, gender)
            if score is None:
                continue

            prompt = build_prompt(args.condition, name, text)
            tokens, offsets, rel, heads, rendered = rollout_relevance(
                tok, model, prompt, str(int(round(score)))
            )
            rows, stats = analyse(tokens, offsets, rel, rendered, text, name)

            stem = f"{args.model}_{args.condition}_{eid}_{reg}_{gender}"
            with open(OUT_DIR / f"{stem}_tokens.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            write_heatmap(OUT_DIR / f"{stem}.html", rows, rendered,
                          f"{eid} / {reg} / {name} / score {score}")

            for layer, vec in enumerate(heads):
                for head, v in enumerate(vec.tolist()):
                    head_rows.append({
                        "model": args.model, "condition": args.condition,
                        "essay_id": eid, "register": reg, "gender": gender,
                        "layer": layer, "head": head,
                        "contribution": round(v, 8),
                    })

            summary.append({
                "model": args.model, "condition": args.condition,
                "essay_id": eid, "register": reg, "gender": gender,
                "name": name, "mean_score": score,
                "urban_minus_tier2": round(gap, 2), **stats,
            })
            print(f"{eid:<24} {reg:<6} {gender} score={score} "
                  f"dialect_ratio={stats['dialect_ratio']} "
                  f"name_ratio={stats['name_ratio']}")

    if summary:
        with open(SUMMARY_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            if f.tell() == 0:
                w.writeheader()
            w.writerows(summary)
        with open(ROOT / "results" / "attention_heads.csv", "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(head_rows[0].keys()))
            if f.tell() == 0:
                w.writeheader()
            w.writerows(head_rows)
        print(f"\nWrote {SUMMARY_PATH} and per-token files in {OUT_DIR}")


if __name__ == "__main__":
    main()
