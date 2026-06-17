"""
BERTopic-style grid search over UMAP + HDBSCAN, selecting the configuration that
jointly maximizes topic coherence and silhouette score.

Cross-modal setup (this is the important bit):
  - Clustering runs on the SOLUTION embeddings (the .npy you already produced).
  - Silhouette is therefore measured in SOLUTION space  -> "are the code clusters
    well separated / compact?"
  - Topic coherence is computed on the QUESTION text grouped by those same labels
    -> "do clusters formed in code space also correspond to coherent NL topics?"
  This makes coherence a *cross-modal validity check* on solution-side clusters,
  which dovetails with the thesis framing that diversity/structure should be
  assessed on the solution side, not only the prompt side.

Pipeline mirrors the cited passage: UMAP -> HDBSCAN -> class-based TF-IDF +
CountVectorizer term filtering + (POS x MMR) representation, with a constrained
grid search over UMAP (n_neighbors, n_components) and HDBSCAN (min_cluster_size,
min_samples), scored by coherence and silhouette.

Install:
    pip install bertopic gensim umap-learn hdbscan scikit-learn matplotlib pandas
    # POS representation (final fit only) needs a spaCy model:
    python -m spacy download en_core_web_sm
"""

import os
import warnings
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction

warnings.filterwarnings("ignore")


# =============================================================================
# Config
# =============================================================================

EMB_PATH       = "embeddings_output/solution_embeddings.npy"   # solution embeddings
TEXTS_CSV_PATH = "embeddings_output/solution_texts.csv"        # must contain a question column
OUT_DIR        = "bertopic_grid_search_outputs"

# Auto-detected if None; otherwise force the column name holding the NL question/prompt.
QUESTION_COL = None
QUESTION_COL_CANDIDATES = ["question", "prompt", "problem", "instruction", "query", "text", "nl"]

RANDOM_STATE = 42

# --- fixed UMAP settings (kept constant across the grid, as in the original script) ---
UMAP_MIN_DIST = 0.0          # 0.0 packs clusters tightly -> better for HDBSCAN
UMAP_METRIC   = "cosine"

# --- the grid (centered on the cited best config: n_comp=20, n_neigh=3, mcs=10, ms=1) ---
GRID = {
    "umap_n_neighbors":         [5, 10, 15],
    "umap_n_components":        [10, 20],
    "hdbscan_min_cluster_size": [10, 20, 30],
    "hdbscan_min_samples":      [1, 5],
}


HDBSCAN_CLUSTER_SELECTION = "eom"   # "eom" or "leaf"

# --- vectorizer / c-TF-IDF (CountVectorizer-based term filtering + class TF-IDF) ---
VECTORIZER_STOPWORDS = "english"
VECTORIZER_NGRAM     = (1, 1)   # (1,2) also fine; keep (1,1) for stable coherence
VECTORIZER_MIN_DF    = 2        # drop very rare terms
COHERENCE_METRIC     = "c_v"    # "c_v" (human-correlated) | "c_npmi" | "u_mass"
COHERENCE_TOPN       = 10       # top words per topic used for coherence

# --- silhouette space used FOR SELECTION ---
#   "reduced"  : euclidean silhouette in the UMAP space clustering happened in.
#                Comparable in magnitude to the ~0.57 reported in the passage, but the
#                space changes per config (slightly apples-to-oranges across configs).
#   "original" : cosine silhouette in the fixed high-dim solution space. More
#                conservative (lower numbers) and a fairer cross-config comparison;
#                good to report as a robustness check if a reviewer pushes back.
SELECTION_SILHOUETTE = "reduced"

# --- selection: only consider configs within a sane topic-count band ---
MIN_TOPICS = 2
MAX_TOPICS = None     # e.g. 200 to discard over-fragmented configs; None = no cap
COH_WEIGHT = 0.5      # weight on (normalized) coherence; silhouette gets (1 - COH_WEIGHT)

# --- final fit (the winning config) ---
APPLY_POS_MMR  = True         # POS x MMR representation for nicer final topic labels
SPACY_MODEL    = "en_core_web_sm"
MMR_DIVERSITY  = 0.3

os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_CSV   = f"{OUT_DIR}/grid_search_results.csv"
PARETO_PNG    = f"{OUT_DIR}/coherence_vs_silhouette.png"
ASSIGN_CSV    = f"{OUT_DIR}/best_cluster_assignments.csv"
TOPICS_CSV    = f"{OUT_DIR}/best_topic_info.csv"
UMAP2D_PNG    = f"{OUT_DIR}/best_umap_2d.png"


# =============================================================================
# Helpers
# =============================================================================

def normalize_rows(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def detect_question_col(df):
    if QUESTION_COL is not None:
        if QUESTION_COL not in df.columns:
            raise ValueError(f"QUESTION_COL='{QUESTION_COL}' not in {list(df.columns)}")
        return QUESTION_COL
    for c in QUESTION_COL_CANDIDATES:
        if c in df.columns:
            return c
    # fall back to the first string-like column
    for c in df.columns:
        if df[c].dtype == object:
            print(f"[warn] no known question column; falling back to '{c}'")
            return c
    raise ValueError(f"Could not find a question/text column in {list(df.columns)}")


def build_umap(n_neighbors, n_components):
    return umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
        low_memory=False,
    )


def build_hdbscan(min_cluster_size, min_samples):
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION,
        metric="euclidean",          # euclidean on the UMAP output
        prediction_data=True,
    )


def build_vectorizer():
    return CountVectorizer(
        stop_words=VECTORIZER_STOPWORDS,
        ngram_range=VECTORIZER_NGRAM,
        min_df=VECTORIZER_MIN_DF,
    )


def safe_silhouette(emb, labels, metric):
    """Silhouette over non-noise points only; NaN when undefined."""
    mask = labels != -1
    e, l = emb[mask], labels[mask]
    n_lab = len(set(l.tolist()))
    if n_lab < 2 or n_lab > len(l) - 1:
        return float("nan")
    try:
        return float(silhouette_score(e, l, metric=metric))
    except Exception:
        return float("nan")


def compute_coherence(topic_model, tokenized_docs, dictionary):
    """Coherence on QUESTION text using each topic's top words (c-TF-IDF terms)."""
    topics_words = []
    for tid in topic_model.get_topics():
        if tid == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(tid)[:COHERENCE_TOPN] if w]
        # keep only words present in the dictionary so coherence is well-defined
        words = [w for w in words if w in dictionary.token2id]
        if len(words) >= 2:
            topics_words.append(words)
    if len(topics_words) < 1:
        return float("nan")
    try:
        cm_ = CoherenceModel(
            topics=topics_words,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence=COHERENCE_METRIC,
        )
        return float(cm_.get_coherence())
    except Exception:
        return float("nan")


def fit_one(reduced, questions, hdbscan_model, vectorizer, representation_model=None):
    """
    BERTopic with an EMPTY reducer so we control UMAP ourselves:
      - `reduced` are the precomputed UMAP coords -> HDBSCAN clusters on them
      - `questions` are the documents -> class-based TF-IDF / coherence
    Returns the fitted model and integer labels.
    """
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=BaseDimensionalityReduction(),     # skip reduction; reduced are final
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        representation_model=representation_model,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(documents=questions, embeddings=reduced)
    return topic_model, np.asarray(topics)


# =============================================================================
# Grid search
# =============================================================================

def run_grid(embeddings_orig, reduced_cache_key_unused, questions, tokenized_docs, dictionary):
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"Grid: {len(combos)} configurations\n")

    rows = []
    for i, combo in enumerate(combos, 1):
        cfg = dict(zip(keys, combo))
        tag = (f"nn={cfg['umap_n_neighbors']:>2} nc={cfg['umap_n_components']:>2} "
               f"mcs={cfg['hdbscan_min_cluster_size']:>2} ms={cfg['hdbscan_min_samples']:>2}")
        try:
            # UMAP reduction (done once per config, here, so we keep `reduced` for silhouette)
            reduced = build_umap(cfg["umap_n_neighbors"],
                                 cfg["umap_n_components"]).fit_transform(embeddings_orig)

            hdb = build_hdbscan(cfg["hdbscan_min_cluster_size"], cfg["hdbscan_min_samples"])
            topic_model, labels = fit_one(reduced, questions, hdb, build_vectorizer())

            n_topics = len(set(labels.tolist()) - {-1})
            n_noise  = int((labels == -1).sum())

            sil_red = safe_silhouette(reduced,         labels, "euclidean")
            sil_org = safe_silhouette(embeddings_orig, labels, "cosine")
            coh     = compute_coherence(topic_model, tokenized_docs, dictionary)

            rows.append({**cfg,
                         "n_topics": n_topics, "n_outliers": n_noise,
                         "coherence": coh,
                         "silhouette_reduced": sil_red,
                         "silhouette_original": sil_org})
            print(f"[{i:>3}/{len(combos)}] {tag} | topics={n_topics:>3} "
                  f"outliers={n_noise:>4} | coh={coh:.3f} "
                  f"sil_red={sil_red:.3f} sil_org={sil_org:.3f}")
        except Exception as e:
            rows.append({**cfg, "n_topics": np.nan, "n_outliers": np.nan,
                         "coherence": np.nan, "silhouette_reduced": np.nan,
                         "silhouette_original": np.nan, "error": str(e)[:120]})
            print(f"[{i:>3}/{len(combos)}] {tag} | FAILED: {str(e)[:80]}")

        # incremental save so a crash never loses progress
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    return pd.DataFrame(rows)


def select_best(results):
    sil_col = f"silhouette_{SELECTION_SILHOUETTE}"
    df = results.copy()
    valid = df["coherence"].notna() & df[sil_col].notna() & (df["n_topics"] >= MIN_TOPICS)
    if MAX_TOPICS is not None:
        valid &= df["n_topics"] <= MAX_TOPICS
    cand = df[valid].copy()
    if cand.empty:
        raise RuntimeError("No valid configurations. Loosen MIN_TOPICS / widen the grid.")

    def mm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.5, index=s.index)

    cand["coh_norm"] = mm(cand["coherence"])
    cand["sil_norm"] = mm(cand[sil_col])
    cand["combined"] = COH_WEIGHT * cand["coh_norm"] + (1 - COH_WEIGHT) * cand["sil_norm"]

    # Pareto frontier (maximize both raw metrics)
    pareto = []
    for idx, r in cand.iterrows():
        dominated = ((cand["coherence"] >= r["coherence"]) &
                     (cand[sil_col] >= r[sil_col]) &
                     ((cand["coherence"] > r["coherence"]) | (cand[sil_col] > r[sil_col]))).any()
        if not dominated:
            pareto.append(idx)
    cand["pareto"] = cand.index.isin(pareto)

    cand = cand.sort_values("combined", ascending=False)
    return cand, sil_col


def plot_pareto(cand, sil_col):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cand["coherence"], cand[sil_col], s=25, alpha=0.5,
               c="steelblue", label="configs")
    pf = cand[cand["pareto"]].sort_values("coherence")
    ax.plot(pf["coherence"], pf[sil_col], "-o", color="crimson", ms=6,
            label="Pareto frontier")
    best = cand.iloc[0]
    ax.scatter([best["coherence"]], [best[sil_col]], s=180, marker="*",
               color="gold", edgecolor="black", zorder=5, label="selected")
    ax.set_xlabel("Topic coherence (question text)")
    ax.set_ylabel(f"Silhouette ({SELECTION_SILHOUETTE} space)")
    ax.set_title("Constrained grid search: coherence vs silhouette")
    ax.legend(); ax.grid(True, linewidth=0.3)
    plt.tight_layout(); plt.savefig(PARETO_PNG, dpi=200); plt.close(fig)
    print(f"Saved: {PARETO_PNG}")


def build_final_representation():
    if not APPLY_POS_MMR:
        return None
    try:
        from bertopic.representation import PartOfSpeech, MaximalMarginalRelevance
        rep = [PartOfSpeech(SPACY_MODEL),
               MaximalMarginalRelevance(diversity=MMR_DIVERSITY)]
        return rep
    except Exception as e:
        print(f"[warn] POS/MMR unavailable ({str(e)[:80]}); using c-TF-IDF only. "
              f"Run: python -m spacy download {SPACY_MODEL}")
        return None


def final_fit(embeddings_orig, questions, best):
    print("\n" + "=" * 80)
    print("REFITTING WINNING CONFIG (with POS x MMR representation)")
    print("=" * 80)
    reducer = build_umap(int(best["umap_n_neighbors"]), int(best["umap_n_components"]))
    reduced = reducer.fit_transform(embeddings_orig)
    hdb = build_hdbscan(int(best["hdbscan_min_cluster_size"]), int(best["hdbscan_min_samples"]))
    topic_model, labels = fit_one(reduced, questions, hdb, build_vectorizer(),
                                  representation_model=build_final_representation())

    n_topics = len(set(labels.tolist()) - {-1})
    n_noise  = int((labels == -1).sum())
    print(f"Final: {n_topics} topics, {n_noise} outliers")

    info = topic_model.get_topic_info()
    info.to_csv(TOPICS_CSV, index=False)
    assign = pd.DataFrame({"row_id": np.arange(len(labels)),
                           "question": questions, "topic": labels})
    assign.to_csv(ASSIGN_CSV, index=False)
    print(f"Saved: {TOPICS_CSV}\nSaved: {ASSIGN_CSV}")

    # 2D UMAP just for the figure
    umap2d = build_umap(int(best["umap_n_neighbors"]), 2)
    umap2d.n_components = 2
    coords = umap.UMAP(n_neighbors=int(best["umap_n_neighbors"]), n_components=2,
                       min_dist=0.1, metric=UMAP_METRIC,
                       random_state=RANDOM_STATE).fit_transform(embeddings_orig)
    uniq = sorted(set(labels.tolist()) - {-1})
    colors = cm.tab20(np.linspace(0, 1, max(len(uniq), 1)))
    fig, ax = plt.subplots(figsize=(10, 7))
    nm = labels == -1
    if nm.any():
        ax.scatter(coords[nm, 0], coords[nm, 1], c="lightgrey", s=8, alpha=0.4, label="outliers")
    for j, c in enumerate(uniq):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], color=colors[j % len(colors)], s=12, alpha=0.7)
    ax.set_title(f"Best config: {n_topics} topics, {n_noise} outliers")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.grid(True, linewidth=0.3)
    plt.tight_layout(); plt.savefig(UMAP2D_PNG, dpi=200); plt.close(fig)
    print(f"Saved: {UMAP2D_PNG}")
    return topic_model


# =============================================================================
# Main
# =============================================================================

def main():
    embeddings = np.load(EMB_PATH).astype(np.float32)
    df = pd.read_csv(TEXTS_CSV_PATH)
    if len(embeddings) != len(df):
        raise ValueError(f"{len(embeddings)} embeddings vs {len(df)} rows")
    if np.isnan(embeddings).any():
        raise ValueError("Embeddings contain NaNs.")

    qcol = detect_question_col(df)
    questions = df[qcol].fillna("").astype(str).tolist()
    print(f"Embeddings: {embeddings.shape} | question column: '{qcol}'")

    # cosine -> normalize so UMAP(cosine) and downstream are consistent
    embeddings = normalize_rows(embeddings)

    # tokenized reference corpus + dictionary for coherence (same analyzer as vectorizer)
    analyzer = build_vectorizer().build_analyzer()
    tokenized_docs = [analyzer(q) for q in questions]
    dictionary = Dictionary(tokenized_docs)

    results = run_grid(embeddings, None, questions, tokenized_docs, dictionary)
    cand, sil_col = select_best(results)
    plot_pareto(cand, sil_col)

    best = cand.iloc[0]
    print("\n" + "=" * 80)
    print("SELECTED CONFIGURATION")
    print("=" * 80)
    print(f"  umap_n_neighbors        = {int(best['umap_n_neighbors'])}")
    print(f"  umap_n_components       = {int(best['umap_n_components'])}")
    print(f"  hdbscan_min_cluster_size= {int(best['hdbscan_min_cluster_size'])}")
    print(f"  hdbscan_min_samples     = {int(best['hdbscan_min_samples'])}")
    print(f"  -> coherence  = {best['coherence']:.3f}")
    print(f"  -> silhouette = {best[sil_col]:.3f}  ({SELECTION_SILHOUETTE} space)")
    print(f"  -> topics={int(best['n_topics'])}, outliers={int(best['n_outliers'])}")

    print("\nTop 5 by combined score:")
    cols = list(GRID.keys()) + ["n_topics", "n_outliers", "coherence", sil_col, "combined", "pareto"]
    print(cand[cols].head(5).to_string(index=False))

    final_fit(embeddings, questions, best)
    print(f"\nAll outputs in: {OUT_DIR}/\nDone.")


if __name__ == "__main__":
    main()