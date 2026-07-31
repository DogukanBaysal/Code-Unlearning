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

Curation gates:
    review    : centered cosine >= 0.60
    hard-fail : centered cosine >= 0.75
    cross-pool: aim below 0.50

Examples
--------
    python embedding_audit.py --dataset dbaysal/all-content --column content

    python embedding_audit.py --models Salesforce/SFR-Embedding-Code-400M_R

    python embedding_audit.py --label-column split --cross-threshold 0.50

    python embedding_audit.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from huggingface_hub import login


os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")


DEFAULT_MODELS = [
    "Salesforce/SFR-Embedding-Code-400M_R",
    "Qodo/Qodo-Embed-1-1.5B",
]


# --------------------------------------------------------------------------- #
# Core score
# --------------------------------------------------------------------------- #
def center_and_normalize(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-center over the pool, then L2-normalize each row."""
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
    hist_bins: int = 100,
) -> tuple[list[Pair], dict]:
    """Find pairs above threshold using row-blocked matmuls.

    If labels are given, only pairs with differing labels are kept and scored
    against cross_threshold.

    Histograms are accumulated for every upper-triangle pair without storing the
    full N x N matrix.
    """
    n = centered.shape[0]
    thr = cross_threshold if labels is not None else review_threshold
    pairs: list[Pair] = []

    edges = np.linspace(-1.0, 1.0, hist_bins + 1)
    centered_hist = np.zeros(hist_bins, dtype=np.int64)
    raw_hist = np.zeros(hist_bins, dtype=np.int64)

    for start in range(0, n, block_size):
        end = min(start + block_size, n)

        sims = centered[start:end] @ centered.T
        raw_sims = raw_emb[start:end] @ raw_emb.T

        for local, i in enumerate(range(start, end)):
            row = sims[local]

            upper = row[i + 1 :]
            raw_upper = raw_sims[local, i + 1 :]

            centered_hist += np.histogram(upper, bins=edges)[0]
            raw_hist += np.histogram(raw_upper, bins=edges)[0]

            cand = np.nonzero(upper >= thr)[0] + (i + 1)

            for j in cand:
                if labels is not None and labels[i] == labels[j]:
                    continue

                cscore = float(row[j])
                rscore = float(raw_emb[i] @ raw_emb[j])

                if cscore >= hard_fail_threshold:
                    flag = "HARD_FAIL"
                elif cscore >= review_threshold:
                    flag = "REVIEW"
                else:
                    flag = "CROSS"

                pairs.append(
                    Pair(
                        i=int(i),
                        j=int(j),
                        centered_score=cscore,
                        raw_score=rscore,
                        flag=flag,
                    )
                )

    pairs.sort(key=lambda p: p.centered_score, reverse=True)

    hist = {
        "edges": edges,
        "centered": centered_hist,
        "raw": raw_hist,
        "total_pairs": int(n * (n - 1) // 2),
    }

    return pairs, hist


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
PLOT_THRESHOLD = 0.6
PLOT_FONT_SIZES = {
    "title": 30,
    "suptitle": 34,
    "label": 28,
    "tick": 24,
    "legend": 24,
}


def plot_embedding_diversity(
    hist: dict,
    model_name: str,
    out_dir: str,
    review_threshold: float,
    hard_fail_threshold: float,
) -> str | None:
    """Plot raw vs centered cosine distribution for one model."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped ({exc}); install matplotlib to enable plots")
        return None

    edges = hist["edges"]
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    fig, ax = plt.subplots(figsize=(13, 9))

    ax.bar(
        centers,
        hist["raw"],
        width=width,
        alpha=0.4,
        color="gray",
        label="raw cosine diagnostic",
    )
    ax.bar(
        centers,
        hist["centered"],
        width=width,
        alpha=0.6,
        color="steelblue",
        label="centered cosine used",
    )

    ax.axvline(
        PLOT_THRESHOLD,
        color="red",
        ls="--",
        linewidth=2.2,
        label=f"threshold = {PLOT_THRESHOLD}",
    )

    ax.set_yscale("log")
    ax.set_xlabel("pairwise cosine similarity", fontsize=PLOT_FONT_SIZES["label"])
    ax.set_ylabel("number of pairs, log scale", fontsize=PLOT_FONT_SIZES["label"])
    ax.set_title(
        f"Pairwise similarity distribution ({hist['total_pairs']} pairs)\n"
        f"{model_name}",
        fontsize=PLOT_FONT_SIZES["title"],
    )
    ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])
    ax.legend(fontsize=PLOT_FONT_SIZES["legend"])

    fig.tight_layout()

    safe = model_name.replace("/", "__")
    path = os.path.join(out_dir, f"diversity_{safe}.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"wrote {path}")
    return path


def plot_models_side_by_side(
    model_hists: list[tuple[str, dict]],
    out_dir: str,
    review_threshold: float,
    hard_fail_threshold: float,
) -> str | None:
    """Plot raw and centered cosine distributions for all models side by side.

    Each panel corresponds to one embedding model.
    Within each panel:
      - gray bars: raw cosine diagnostic
      - blue bars: centered cosine used for duplicate decisions
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped ({exc}); install matplotlib to enable plots")
        return None

    if not model_hists:
        return None

    n_models = len(model_hists)

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(11 * n_models, 9),
        sharex=True,
        sharey=True,
    )

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, hist) in zip(axes, model_hists):
        edges = hist["edges"]
        centers = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]

        ax.bar(
            centers,
            hist["raw"],
            width=width,
            alpha=0.40,
            color="gray",
            label="raw cosine",
        )
        ax.bar(
            centers,
            hist["centered"],
            width=width,
            alpha=0.65,
            color="steelblue",
            label="centered cosine",
        )

        ax.axvline(
            PLOT_THRESHOLD,
            color="red",
            ls="--",
            linewidth=2.2,
            label=f"threshold = {PLOT_THRESHOLD}",
        )

        ax.set_title(model_name, fontsize=PLOT_FONT_SIZES["title"])
        ax.set_xlabel("pairwise cosine similarity", fontsize=PLOT_FONT_SIZES["label"])
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])
        ax.legend(fontsize=PLOT_FONT_SIZES["legend"])

    axes[0].set_ylabel("number of pairs, log scale", fontsize=PLOT_FONT_SIZES["label"])

    fig.suptitle(
        "Raw vs centered pairwise similarity distribution by embedding model",
        fontsize=PLOT_FONT_SIZES["suptitle"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    path = os.path.join(out_dir, "diversity_models_side_by_side.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"wrote {path}")
    return path


def write_embedding_histogram_csv(hist: dict, model_name: str, out_dir: str) -> str:
    """Dump histogram bin counts so figures can be regenerated later."""
    edges = hist["edges"]
    safe = model_name.replace("/", "__")
    path = os.path.join(out_dir, f"diversity_hist_{safe}.csv")

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "bin_left",
                "bin_right",
                "bin_center",
                "centered_count",
                "raw_count",
            ]
        )

        for k in range(len(edges) - 1):
            left = float(edges[k])
            right = float(edges[k + 1])
            center = (left + right) / 2

            w.writerow(
                [
                    f"{left:.6f}",
                    f"{right:.6f}",
                    f"{center:.6f}",
                    int(hist["centered"][k]),
                    int(hist["raw"][k]),
                ]
            )

    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------------- #
# Data and model loading
# --------------------------------------------------------------------------- #
def maybe_login_to_huggingface() -> None:
    """Log in to Hugging Face if HF_TOKEN is set."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return

    try:
        from huggingface_hub import login

        login(token=token)
    except Exception as exc:
        print(f"[hf] login skipped/failed: {exc}", file=sys.stderr)


def load_texts(
    dataset: str,
    column: str,
    split: str | None,
    label_column: str | None,
    max_rows: int | None,
) -> tuple[list[str], np.ndarray | None]:
    from datasets import concatenate_datasets, load_dataset

    dsd = load_dataset(dataset)

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

    if label_column and label_column not in ds.column_names:
        raise KeyError(
            f"Label column '{label_column}' not in dataset. "
            f"Available: {ds.column_names}"
        )

    texts_raw = ds[column]
    labels = np.asarray(ds[label_column]) if label_column else None

    keep = [
        k
        for k, text in enumerate(texts_raw)
        if isinstance(text, str) and text.strip()
    ]

    texts = [texts_raw[k] for k in keep]

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
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
    )

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

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
        texts=texts,
        model_name=model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        trust_remote_code=args.trust_remote_code,
    )

    print(
        f"embedded {raw_emb.shape[0]} items, dim={raw_emb.shape[1]} "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )

    centered, _mu = center_and_normalize(raw_emb)

    pairs, hist = find_pairs(
        raw_emb=raw_emb,
        centered=centered,
        review_threshold=args.review_threshold,
        hard_fail_threshold=args.hard_fail_threshold,
        block_size=args.block_size,
        labels=labels,
        cross_threshold=args.cross_threshold if labels is not None else None,
        hist_bins=args.hist_bins,
    )

    hist_csv_path = write_embedding_histogram_csv(hist, model_name, out_dir)

    plot_path = None
    if not args.no_plots:
        plot_path = plot_embedding_diversity(
            hist=hist,
            model_name=model_name,
            out_dir=out_dir,
            review_threshold=args.review_threshold,
            hard_fail_threshold=args.hard_fail_threshold,
        )

    def preview(text: str) -> str:
        return " ".join(text.split())[:120]

    for p in pairs:
        p.text_i = preview(texts[p.i])
        p.text_j = preview(texts[p.j])

    safe = model_name.replace("/", "__")

    if args.save_embeddings:
        np.save(os.path.join(out_dir, f"emb_raw_{safe}.npy"), raw_emb)
        np.save(os.path.join(out_dir, f"emb_centered_{safe}.npy"), centered)

    csv_path = os.path.join(out_dir, f"pairs_{safe}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "i",
                "j",
                "centered_score",
                "raw_score",
                "flag",
                "text_i",
                "text_j",
            ]
        )

        for p in pairs:
            w.writerow(
                [
                    p.i,
                    p.j,
                    f"{p.centered_score:.4f}",
                    f"{p.raw_score:.4f}",
                    p.flag,
                    p.text_i,
                    p.text_j,
                ]
            )

    n_hard = sum(1 for p in pairs if p.flag == "HARD_FAIL")
    n_review = sum(1 for p in pairs if p.flag == "REVIEW")
    n_cross = sum(1 for p in pairs if p.flag == "CROSS")

    print(
        f"flagged pairs -> hard_fail: {n_hard}  review: {n_review}"
        + (f"  cross: {n_cross}" if labels is not None else "")
    )
    print(f"wrote {csv_path}")

    for p in pairs[:10]:
        print(
            f"  [{p.flag:9s}] c={p.centered_score:.3f} "
            f"raw={p.raw_score:.3f} ({p.i},{p.j})"
        )

    return {
        "model": model_name,
        "n_items": int(raw_emb.shape[0]),
        "embedding_dim": int(raw_emb.shape[1]),
        "n_hard_fail": n_hard,
        "n_review": n_review,
        "n_cross": n_cross,
        "csv": os.path.basename(csv_path),
        "histogram_csv": os.path.basename(hist_csv_path),
        "plot": os.path.basename(plot_path) if plot_path else None,
        "hist": hist,
        "elapsed_sec": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def self_test() -> int:
    emb = np.array(
        [
            [0.80, 0.60],
            [0.75, 0.66],
            [0.10, 0.99],
        ],
        dtype=np.float32,
    )

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
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--dataset", default="dbaysal/all-content")
    p.add_argument("--column", default="content")
    p.add_argument(
        "--split",
        default=None,
        help="Specific split; default auto-detects or concatenates.",
    )
    p.add_argument(
        "--label-column",
        default=None,
        help=(
            "If set, run a cross-pool screen: pool mean over all rows, "
            "but only report pairs with differing labels."
        ),
    )
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--review-threshold", type=float, default=0.60)
    p.add_argument("--hard-fail-threshold", type=float, default=0.75)
    p.add_argument(
        "--cross-threshold",
        type=float,
        default=0.50,
        help="Gate for cross-pool pairs, used with --label-column.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Row block size for the pairwise pass.",
    )
    p.add_argument("--hist-bins", type=int, default=100)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--device",
        default="cuda",
        help="cuda / cpu / auto-supported value for SentenceTransformer.",
    )
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )
    p.add_argument("--save-embeddings", action="store_true")
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable per-model and combined diversity plots.",
    )
    p.add_argument("--out-dir", default="embedding_audit_out")
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Verify centered-cosine math against the spec; no model.",
    )

    args, _unknown = p.parse_known_args(argv)
    return args


def make_json_safe_summary(summaries: list[dict]) -> list[dict]:
    """Remove non-JSON histogram arrays from model summaries."""
    json_summaries = []

    for summary in summaries:
        clean = dict(summary)
        clean.pop("hist", None)
        json_summaries.append(clean)

    return json_summaries


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return self_test()

    maybe_login_to_huggingface()

    os.makedirs(args.out_dir, exist_ok=True)

    print(
        f"loading {args.dataset} (column='{args.column}')...",
        flush=True,
    )

    texts, labels = load_texts(
        dataset=args.dataset,
        column=args.column,
        split=args.split,
        label_column=args.label_column,
        max_rows=args.max_rows,
    )

    print(f"loaded {len(texts)} non-empty rows", flush=True)

    if labels is not None:
        uniq, counts = np.unique(labels, return_counts=True)
        print("label distribution:", dict(zip(map(str, uniq), map(int, counts))))

    summaries = [
        audit_model(
            texts=texts,
            labels=labels,
            model_name=model_name,
            args=args,
            out_dir=args.out_dir,
        )
        for model_name in args.models
    ]

    combined_plot_path = None
    if not args.no_plots:
        combined_plot_path = plot_models_side_by_side(
            model_hists=[(s["model"], s["hist"]) for s in summaries],
            out_dir=args.out_dir,
            review_threshold=args.review_threshold,
            hard_fail_threshold=args.hard_fail_threshold,
        )

    summary = {
        "dataset": args.dataset,
        "column": args.column,
        "n_items": len(texts),
        "review_threshold": args.review_threshold,
        "hard_fail_threshold": args.hard_fail_threshold,
        "cross_threshold": args.cross_threshold if labels is not None else None,
        "label_column": args.label_column,
        "combined_plot": (
            os.path.basename(combined_plot_path) if combined_plot_path else None
        ),
        "models": make_json_safe_summary(summaries),
    }

    summary_path = os.path.join(args.out_dir, "summary.json")

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nwrote {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
