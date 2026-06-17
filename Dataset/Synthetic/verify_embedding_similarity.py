#!/usr/bin/env python3
"""
embedding_audit.py
==================

Normalized embedding similarity audit for CrossMemo dataset curation.

Implements the mean-centered, L2-normalized cosine similarity score defined in
`embedding_score_definition.md`, applied to a HuggingFace dataset (default
`dbaysal/all-content`, column `content`).

Score definition (per comparison pool being audited):

    e_i  = model.encode(text_i, normalize_embeddings=True)   # ||e_i|| ~ 1
    mu   = mean_i(e_i)                                        # pool mean
    c_i  = e_i - mu                                           # mean-center
    z_i  = c_i / (||c_i|| + 1e-12)                            # renormalize
    score(i, j) = z_i . z_j                                   # centered cosine

Raw cosine (e_i . e_j) is retained only as a diagnostic, never as the final
duplicate decision, because it over-counts generic "both inputs are code"
similarity in an anisotropic code-embedding space.

Curation gates (from the note):
    review    : centered cosine >= 0.60
    hard-fail : centered cosine >= 0.75
    cross-pool (e.g. heldout-vs-training): aim below 0.50

Two code embedding models are used by default, matching the duplicate screen:
    Salesforce/SFR-Embedding-Code-400M_R
    Qodo/Qodo-Embed-1-1.5B

Examples
--------
    # Audit the whole pool with both models
    python embedding_audit.py --dataset dbaysal/all-content --column content

    # Single model, lighter run
    python embedding_audit.py --models Salesforce/SFR-Embedding-Code-400M_R

    # Heldout-vs-training style cross-pool check: pool mean over ALL rows,
    # but only report pairs whose split labels differ, against the 0.50 gate.
    python embedding_audit.py --label-column split --cross-threshold 0.50

    # Verify the centered-cosine math against the spec's toy example (no model)
    python embedding_audit.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from huggingface_hub import login
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # only helps if torch built with it

login(token="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")

# Models the duplicate screen was run with (see embedding_score_definition.md).
DEFAULT_MODELS = [
    "Salesforce/SFR-Embedding-Code-400M_R",
    "Qodo/Qodo-Embed-1-1.5B",
]


# --------------------------------------------------------------------------- #
# Core score (pure numpy, no model dependency -- this is what --self-test hits)
# --------------------------------------------------------------------------- #
def center_and_normalize(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-center over the pool, then L2-normalize each row.

    `emb` is (N, d). Rows are assumed already ~unit length (the model is asked
    for normalize_embeddings=True), but this function does not rely on that.
    Returns (centered_unit_vectors, pool_mean).
    """
    emb = emb.astype(np.float32, copy=False)
    mu = emb.mean(axis=0, keepdims=True)
    centered = emb - mu
    centered /= np.linalg.norm(centered, axis=1, keepdims=True) + 1e-12
    return centered, mu


@dataclass
class Pair:
    i: int
    j: int
    centered_score: float
    raw_score: float
    flag: str
    text_i: str = ""
    text_j: str = ""


def find_pairs(
    raw_emb: np.ndarray,
    centered: np.ndarray,
    review_threshold: float,
    hard_fail_threshold: float,
    block_size: int = 1024,
    labels: np.ndarray | None = None,
    cross_threshold: float | None = None,
) -> list[Pair]:
    """Find pairs (i < j) above the review threshold using row-blocked matmuls.

    The N x N similarity matrix is never fully materialized; we process the
    centered matrix in blocks of rows so memory stays O(block_size * N).

    If `labels` is given, only pairs with differing labels are kept and they are
    scored against `cross_threshold` (the heldout-vs-training case). The pool
    mean used to build `centered` is still the mean over the *entire* pool, as
    the note specifies ("mean over heldout plus training source functions").
    """
    n = centered.shape[0]
    thr = cross_threshold if labels is not None else review_threshold
    pairs: list[Pair] = []

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        # (B, N) block of centered cosine similarities.
        sims = centered[start:end] @ centered.T
        for local, i in enumerate(range(start, end)):
            row = sims[local]
            # Only consider j > i to avoid self-pairs and double counting.
            cand = np.nonzero(row[i + 1 :] >= thr)[0] + (i + 1)
            for j in cand:
                if labels is not None and labels[i] == labels[j]:
                    continue
                cscore = float(row[j])
                rscore = float(raw_emb[i] @ raw_emb[j])  # raw cosine, diagnostic
                if cscore >= hard_fail_threshold:
                    flag = "HARD_FAIL"
                elif cscore >= review_threshold:
                    flag = "REVIEW"
                else:
                    flag = "CROSS"  # below review gate but flagged for cross-pool
                pairs.append(Pair(int(i), int(j), cscore, rscore, flag))

    pairs.sort(key=lambda p: p.centered_score, reverse=True)
    return pairs


# --------------------------------------------------------------------------- #
# Data + model loading
# --------------------------------------------------------------------------- #
def load_texts(
    dataset: str,
    column: str,
    split: str | None,
    label_column: str | None,
    max_rows: int | None,
) -> tuple[list[str], np.ndarray | None]:
    from datasets import concatenate_datasets, load_dataset

    dsd = load_dataset(dataset)  # DatasetDict
    if split:
        ds = dsd[split]
    elif len(dsd) == 1:
        ds = dsd[next(iter(dsd))]
    else:
        ds = concatenate_datasets([dsd[s] for s in dsd])

    if column not in ds.column_names:
        raise KeyError(
            f"Column '{column}' not in dataset. Available: {ds.column_names}"
        )

    texts = ds[column]
    labels = np.asarray(ds[label_column]) if label_column else None

    # Drop rows with empty/None content, keep labels aligned.
    keep = [k for k, t in enumerate(texts) if isinstance(t, str) and t.strip()]
    texts = [texts[k] for k in keep]
    if labels is not None:
        labels = labels[keep]

    if max_rows is not None and len(texts) > max_rows:
        texts = texts[:max_rows]
        if labels is not None:
            labels = labels[:max_rows]

    return texts, labels


def embed(
    texts: list[str],
    model_name: str,
    device: str | None,
    batch_size: int,
    max_seq_length: int | None,
    trust_remote_code: bool,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name, device=device, trust_remote_code=True,
    )
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    # Symmetric duplicate screen -> embed every text the same way (no
    # asymmetric query/document prompt). normalize_embeddings=True so each raw
    # embedding has length ~1, exactly as the note specifies.
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return emb.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Per-model audit
# --------------------------------------------------------------------------- #
def audit_model(
    texts: list[str],
    labels: np.ndarray | None,
    model_name: str,
    args: argparse.Namespace,
    out_dir: str,
) -> dict:
    t0 = time.time()
    print(f"\n=== model: {model_name} ===", flush=True)
    raw_emb = embed(
        texts,
        model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"embedded {raw_emb.shape[0]} items, dim={raw_emb.shape[1]} "
          f"in {time.time() - t0:.1f}s", flush=True)

    centered, _mu = center_and_normalize(raw_emb)

    pairs = find_pairs(
        raw_emb,
        centered,
        review_threshold=args.review_threshold,
        hard_fail_threshold=args.hard_fail_threshold,
        block_size=args.block_size,
        labels=labels,
        cross_threshold=args.cross_threshold if labels is not None else None,
    )

    # Attach short content previews for human review.
    preview = lambda s: " ".join(s.split())[:120]
    for p in pairs:
        p.text_i = preview(texts[p.i])
        p.text_j = preview(texts[p.j])

    safe = model_name.replace("/", "__")
    if args.save_embeddings:
        np.save(os.path.join(out_dir, f"emb_raw_{safe}.npy"), raw_emb)
        np.save(os.path.join(out_dir, f"emb_centered_{safe}.npy"), centered)

    # Write pairs CSV.
    import csv
    csv_path = os.path.join(out_dir, f"pairs_{safe}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["i", "j", "centered_score", "raw_score", "flag",
                    "text_i", "text_j"])
        for p in pairs:
            w.writerow([p.i, p.j, f"{p.centered_score:.4f}",
                        f"{p.raw_score:.4f}", p.flag, p.text_i, p.text_j])

    n_hard = sum(1 for p in pairs if p.flag == "HARD_FAIL")
    n_review = sum(1 for p in pairs if p.flag == "REVIEW")
    n_cross = sum(1 for p in pairs if p.flag == "CROSS")

    print(f"flagged pairs -> hard_fail: {n_hard}  review: {n_review}"
          + (f"  cross: {n_cross}" if labels is not None else ""))
    print(f"wrote {csv_path}")
    for p in pairs[:10]:
        print(f"  [{p.flag:9s}] c={p.centered_score:.3f} raw={p.raw_score:.3f} "
              f"({p.i},{p.j})")

    return {
        "model": model_name,
        "n_items": int(raw_emb.shape[0]),
        "embedding_dim": int(raw_emb.shape[1]),
        "n_hard_fail": n_hard,
        "n_review": n_review,
        "n_cross": n_cross,
        "csv": os.path.basename(csv_path),
        "elapsed_sec": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------- #
# Self-test against the note's toy example
# --------------------------------------------------------------------------- #
def self_test() -> int:
    emb = np.array([[0.80, 0.60], [0.75, 0.66], [0.10, 0.99]], dtype=np.float32)
    centered, mu = center_and_normalize(emb)
    sims = centered @ centered.T

    ok = True
    ok &= np.allclose(mu.ravel(), [0.55, 0.75], atol=1e-3)
    ok &= np.allclose(centered[0], [0.857, -0.514], atol=1e-3)
    ok &= np.allclose(centered[1], [0.912, -0.410], atol=1e-3)
    ok &= np.allclose(centered[2], [-0.882, 0.471], atol=1e-3)
    ok &= abs(float(sims[0, 1]) - 0.993) < 1e-3
    ok &= abs(float(sims[0, 2]) - (-0.998)) < 1e-3

    print("mu       :", mu.ravel())
    print("z_A      :", centered[0])
    print("z_B      :", centered[1])
    print("z_C      :", centered[2])
    print("score AB :", float(sims[0, 1]))
    print("score AC :", float(sims[0, 2]))
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dbaysal/all-content")
    p.add_argument("--column", default="content")
    p.add_argument("--split", default=None,
                   help="Specific split; default auto-detects / concatenates.")
    p.add_argument("--label-column", default=None,
                   help="If set, run a cross-pool screen: pool mean over all "
                        "rows, but only report pairs with differing labels.")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--review-threshold", type=float, default=0.60)
    p.add_argument("--hard-fail-threshold", type=float, default=0.75)
    p.add_argument("--cross-threshold", type=float, default=0.50,
                   help="Gate for cross-pool pairs (used with --label-column).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=1024,
                   help="Row block size for the pairwise pass (memory control).")
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--device", default="cuda", help="cuda / cpu; default auto.")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code",
                   action="store_false")
    p.add_argument("--save-embeddings", action="store_true")
    p.add_argument("--out-dir", default="embedding_audit_out")
    p.add_argument("--self-test", action="store_true",
                   help="Verify centered-cosine math against the spec; no model.")
    args, unknown = p.parse_known_args()
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"loading {args.dataset} (column='{args.column}')...", flush=True)
    texts, labels = load_texts(
        args.dataset, args.column, args.split, args.label_column, args.max_rows
    )
    print(f"loaded {len(texts)} non-empty rows", flush=True)
    if labels is not None:
        uniq, counts = np.unique(labels, return_counts=True)
        print("label distribution:", dict(zip(map(str, uniq), map(int, counts))))

    summaries = [audit_model(texts, labels, m, args, args.out_dir)
                 for m in args.models]

    summary = {
        "dataset": args.dataset,
        "column": args.column,
        "n_items": len(texts),
        "review_threshold": args.review_threshold,
        "hard_fail_threshold": args.hard_fail_threshold,
        "cross_threshold": args.cross_threshold if labels is not None else None,
        "label_column": args.label_column,
        "models": summaries,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())