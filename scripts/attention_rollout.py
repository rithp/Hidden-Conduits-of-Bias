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

LOGIT_PATH = ROOT / "results" / "scores_logit.csv"
from check_variants import FEATURE_PATTERNS, article_density, deletion_spans

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

def _load_scores(model_key, condition):
    """Prefer the logit scores: they cover both models and all four
    conditions, and argmax_score is exactly the token the model would emit,
    which is the token this analysis differentiates."""
    if LOGIT_PATH.exists():
        rows = [r for r in csv.DictReader(open(LOGIT_PATH))
                if r["model"] == model_key and r["condition"] == condition
                and r["expected_score"] not in ("", None)]
        if rows:
            return [{"essay_id": r["essay_id"], "register": r["register"],
                     "gender": r["gender"],
                     "score": float(r["expected_score"]),
                     "target": int(r["argmax_score"])} for r in rows], "logit"
    if SCORES_PATH.exists():
        rows = [r for r in csv.DictReader(open(SCORES_PATH))
                if r["model"] == model_key and r["condition"] == condition
                and r["parse_ok"] == "yes"]
        if rows:
            return [{"essay_id": r["essay_id"], "register": r["register"],
                     "gender": r["gender"], "score": float(r["score"]),
                     "target": int(r["score"])} for r in rows], "sampled"
    raise SystemExit(
        f"No scores for model={model_key} condition={condition}. "
        f"Run score_logits.py --model {model_key} first."
    )


def pick_items(model_key, condition, top_n, select="register"):
    """Which essays to analyse.

    select='register': largest urban-minus-tier2 gap. Use where the register
        penalty is the effect of interest (LLaMA).
    select='gender': largest female-minus-male gap. Use where the gender
        effect is the effect of interest (Qwen), since its register gap is
        near zero and ranking by it would be arbitrary.
    """
    rows, source = _load_scores(model_key, condition)
    print(f"[scores from {source}]")

    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["essay_id"], r["gender"], r["register"])].append(r["score"])

    def avg(key):
        v = bucket.get(key, [])
        return sum(v) / len(v) if v else None

    gaps = []
    seen = set()
    for (eid, g, reg) in bucket:
        if select == "register":
            if reg != "urban" or (eid, g) in seen:
                continue
            seen.add((eid, g))
            u, t = avg((eid, g, "urban")), avg((eid, g, "tier2"))
            if u is None or t is None:
                continue
            gaps.append((u - t, eid, g))
        else:
            if g != "F" or (eid, reg) in seen:
                continue
            seen.add((eid, reg))
            f, m = avg((eid, "F", reg)), avg((eid, "M", reg))
            if f is None or m is None:
                continue
            gaps.append((f - m, eid, "F"))

    gaps.sort(reverse=True)
    lab = "urban - tier2" if select == "register" else "female - male"
    print(f"Top gaps ({lab}), condition {condition}:")
    for gap, eid, g in gaps[:top_n]:
        print(f"  {gap:+.4f}  {eid}  {g}")
    return gaps[:top_n]


def item_score(model_key, condition, eid, reg, gender):
    """Returns (score_for_reporting, target_digit_to_differentiate)."""
    rows, _ = _load_scores(model_key, condition)
    hits = [r for r in rows if r["essay_id"] == eid and r["register"] == reg
            and r["gender"] == gender]
    if not hits:
        return None, None
    score = sum(h["score"] for h in hits) / len(hits)
    targets = [h["target"] for h in hits]
    target = max(set(targets), key=targets.count)
    return round(score, 4), target


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


def analyse(tokens, offsets, relevance, rendered, essay_text, name,
            counterpart_text=None):
    # build_prompt() inserts essay_text.strip(), so the regex must run on the
    # SAME string the offsets refer to, or every span is shifted by the
    # length of the stripped-off whitespace
    essay = essay_text.strip()
    essay_start = rendered.rfind(essay[:80])
    essay_end = essay_start + len(essay)
    assert essay_start >= 0, "essay text not found in rendered prompt"
    assert rendered[essay_start:essay_end] == essay, \
        "essay region misaligned: check for duplicate text in the prompt"

    spans = feature_spans(essay, essay_start)
    n_regex = len(spans)
    # recover omitted articles by aligning against the matched urban twin
    n_deletion = 0
    if counterpart_text:
        dels = deletion_spans(essay, counterpart_text.strip(), essay_start)
        spans += dels
        n_deletion = len(dels)

    label = "Applicant name: "
    name_pos = rendered.rfind(f"{label}{name}")
    label_start = name_pos
    name_start = name_pos + len(label)
    name_end = name_start + len(name)

    rows, dialect, other, name_rel, label_rel = [], [], [], [], []
    for tokn, (s, e), r in zip(tokens, offsets, relevance):
        group = "prompt"
        # overlap, not containment: a token may carry a leading space and so
        # start one character before the span it belongs to
        if s < name_end and e > name_start:
            group = "name"
            name_rel.append(r)
        elif s < name_start and e > label_start:
            # the words "Applicant name:" themselves, as a position-matched
            # control for the name
            group = "name_label"
            label_rel.append(r)
        elif s < essay_end and e > essay_start:
            feat = in_any(spans, s, e)
            group = f"dialect:{feat}" if feat else "essay"
            (dialect if feat else other).append(r)
        rows.append({"token": tokn, "char_start": s, "char_end": e,
                     "group": group, "relevance": round(r, 8)})

    if not name_rel:
        print(f"  WARNING: name '{name}' not matched to any token")

    def mean(x):
        return sum(x) / len(x) if x else 0.0

    def ratio(a, b):
        return round(mean(a) / mean(b), 3) if a and b and mean(b) else ""

    stats = {
        "n_feature_spans_regex": n_regex,
        "n_feature_spans_deletion": n_deletion,
        "n_dialect_tokens": len(dialect),
        "n_essay_tokens": len(other),
        "n_name_tokens": len(name_rel),
        "rel_dialect_mean": mean(dialect),
        "rel_essay_mean": mean(other),
        "rel_name_mean": mean(name_rel),
        "rel_name_label_mean": mean(label_rel),
        "dialect_ratio": ratio(dialect, other),
        "name_ratio": ratio(name_rel, other),
        # the name against its own neighbouring words, which controls for
        # the fact that relevance varies with position in the sequence
        "name_vs_label_ratio": ratio(name_rel, label_rel),
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
    ap.add_argument("--select", choices=["register", "gender"],
                    default="register",
                    help="rank essays by the register gap or the gender gap")
    args = ap.parse_args()

    picks = pick_items(args.model, args.condition, args.top, args.select)
    names = load_names()
    model_id = MODELS[args.model]

    print(f"\nLoading {model_id} with eager attention ...")
    tok, model = load_model(model_id)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, head_rows = [], []

    for gap, eid, gender in picks:
        urban_text = (VAR_DIR / f"{eid}__urban.txt").read_text()
        tier2_text = (VAR_DIR / f"{eid}__tier2.txt").read_text()

        for reg in ["urban", "tier2"]:
            text = urban_text if reg == "urban" else tier2_text
            # the twin, used to locate omitted articles in the tier2 variant
            twin = urban_text if reg == "tier2" else None
            name = names[eid][gender]
            score, target = item_score(args.model, args.condition,
                                       eid, reg, gender)
            if score is None:
                continue

            prompt = build_prompt(args.condition, name, text)
            tokens, offsets, rel, heads, rendered = rollout_relevance(
                tok, model, prompt, str(target)
            )
            rows, stats = analyse(tokens, offsets, rel, rendered, text, name,
                                  counterpart_text=twin)
            stats["article_density"] = article_density(text)

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
                "select": args.select,
                "essay_id": eid, "register": reg, "gender": gender,
                "name": name, "mean_score": score, "target_digit": target,
                "selection_gap": round(gap, 5), **stats,
            })
            print(f"{eid:<24} {reg:<6} {gender} score={score} "
                  f"feats={stats['n_feature_spans_regex']}+"
                  f"{stats['n_feature_spans_deletion']} "
                  f"({stats['n_dialect_tokens']} tok) "
                  f"d_ratio={stats['dialect_ratio']} "
                  f"n_vs_label={stats['name_vs_label_ratio']}")

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