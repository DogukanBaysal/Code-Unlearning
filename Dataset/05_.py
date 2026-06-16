# 02_kmeans_k3_select_50.py
# pip install -U numpy pandas scikit-learn matplotlib

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA


# -----------------------------
# Config
# -----------------------------

OUT_DIR = "kmeans_solution_clustering_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = f"{OUT_DIR}/solution_embeddings.npy"
TEXTS_CSV_PATH = f"{OUT_DIR}/solution_texts.csv"

RANDOM_STATE = 42
K = 3
N_SELECT_PER_CLUSTER = 40

# Selection mode:
# "diverse" = farthest-first traversal, maximizes diversity inside each cluster.
# "core" = selects points closest to own centroid and farthest from other centroids.
#
# Important:
# If you expect max intra-selected distance < min inter-selected distance,
# "core" is more likely to satisfy that.
# "diverse" intentionally picks far-apart points, so it may violate that condition.
SELECTION_MODE = "core"

ASSIGNMENTS_CSV_PATH = f"{OUT_DIR}/cluster_assignments_k3.csv"

FULL_CLUSTER_DISTANCE_REPORT_CSV_PATH = (
    f"{OUT_DIR}/full_cluster_distance_report_k3.csv"
)

FULL_PAIRWISE_DISTANCE_REPORT_CSV_PATH = (
    f"{OUT_DIR}/full_pairwise_cluster_distance_report_k3.csv"
)

SELECTED_CSV_PATH = (
    f"{OUT_DIR}/selected_50_each_cluster_k3_{SELECTION_MODE}.csv"
)

SELECTED_SUMMARY_CSV_PATH = (
    f"{OUT_DIR}/selected_50_each_cluster_summary_k3_{SELECTION_MODE}.csv"
)

SELECTED_INTRA_DISTANCE_CSV_PATH = (
    f"{OUT_DIR}/selected_intra_cluster_distance_report_k3_{SELECTION_MODE}.csv"
)

SELECTED_INTER_DISTANCE_CSV_PATH = (
    f"{OUT_DIR}/selected_inter_cluster_distance_report_k3_{SELECTION_MODE}.csv"
)

SELECTED_SEPARATION_SUMMARY_CSV_PATH = (
    f"{OUT_DIR}/selected_separation_summary_k3_{SELECTION_MODE}.csv"
)

PCA_ALL_CSV_PATH = f"{OUT_DIR}/pca_clusters_k3.csv"
PCA_ALL_PNG_PATH = f"{OUT_DIR}/pca_clusters_k3.png"

PCA_SELECTED_CSV_PATH = (
    f"{OUT_DIR}/pca_selected_50_each_cluster_k3_{SELECTION_MODE}.csv"
)

PCA_SELECTED_PNG_PATH = (
    f"{OUT_DIR}/pca_selected_50_each_cluster_k3_{SELECTION_MODE}.png"
)


# -----------------------------
# Distance reports for full clusters
# -----------------------------

def full_cluster_distance_report(embeddings, labels, metric="cosine"):
    """
    Computes:
    1. Max cosine distance within each full KMeans cluster.
    2. Min cosine distance between every pair of full KMeans clusters.
    """

    unique_clusters = sorted(np.unique(labels))

    intra_rows = []
    inter_rows = []

    # Full intra-cluster distances
    for cluster_id in unique_clusters:
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]

        if len(cluster_indices) <= 1:
            intra_rows.append({
                "cluster": int(cluster_id),
                "cluster_size": int(len(cluster_indices)),
                "max_intra_cluster_cosine_distance": np.nan,
                "mean_intra_cluster_cosine_distance": np.nan,
                "min_intra_cluster_cosine_distance": np.nan,
                "max_intra_row_id_a": None,
                "max_intra_row_id_b": None,
                "min_intra_row_id_a": None,
                "min_intra_row_id_b": None,
            })
            continue

        dist = pairwise_distances(
            cluster_embeddings,
            metric=metric,
        )

        np.fill_diagonal(dist, np.nan)

        max_pos = np.unravel_index(np.nanargmax(dist), dist.shape)
        min_pos = np.unravel_index(np.nanargmin(dist), dist.shape)

        intra_rows.append({
            "cluster": int(cluster_id),
            "cluster_size": int(len(cluster_indices)),
            "max_intra_cluster_cosine_distance": float(np.nanmax(dist)),
            "mean_intra_cluster_cosine_distance": float(np.nanmean(dist)),
            "min_intra_cluster_cosine_distance": float(np.nanmin(dist)),
            "max_intra_row_id_a": int(cluster_indices[max_pos[0]]),
            "max_intra_row_id_b": int(cluster_indices[max_pos[1]]),
            "min_intra_row_id_a": int(cluster_indices[min_pos[0]]),
            "min_intra_row_id_b": int(cluster_indices[min_pos[1]]),
        })

    # Full inter-cluster distances
    for i, cluster_a in enumerate(unique_clusters):
        for cluster_b in unique_clusters[i + 1:]:
            indices_a = np.where(labels == cluster_a)[0]
            indices_b = np.where(labels == cluster_b)[0]

            emb_a = embeddings[indices_a]
            emb_b = embeddings[indices_b]

            dist = pairwise_distances(
                emb_a,
                emb_b,
                metric=metric,
            )

            min_pos = np.unravel_index(np.argmin(dist), dist.shape)
            max_pos = np.unravel_index(np.argmax(dist), dist.shape)

            inter_rows.append({
                "cluster_a": int(cluster_a),
                "cluster_b": int(cluster_b),
                "size_a": int(len(indices_a)),
                "size_b": int(len(indices_b)),
                "min_inter_cluster_cosine_distance": float(np.min(dist)),
                "mean_inter_cluster_cosine_distance": float(np.mean(dist)),
                "max_inter_cluster_cosine_distance": float(np.max(dist)),
                "closest_row_id_a": int(indices_a[min_pos[0]]),
                "closest_row_id_b": int(indices_b[min_pos[1]]),
                "farthest_row_id_a": int(indices_a[max_pos[0]]),
                "farthest_row_id_b": int(indices_b[max_pos[1]]),
            })

    intra_df = pd.DataFrame(intra_rows)
    inter_df = pd.DataFrame(inter_rows)

    return intra_df, inter_df


# -----------------------------
# Selection algorithms
# -----------------------------

def select_diverse_points_from_cluster(
    embeddings,
    candidate_indices,
    cluster_center,
    n_select=50,
    metric="cosine",
):
    """
    Farthest-first traversal.

    Starts with point closest to centroid.
    Then repeatedly selects the point farthest from already-selected points.

    This maximizes diversity, but may increase max intra-selected distance.
    """

    candidate_indices = np.asarray(candidate_indices)

    if len(candidate_indices) <= n_select:
        return list(candidate_indices)

    candidate_embeddings = embeddings[candidate_indices]

    center_distances = pairwise_distances(
        candidate_embeddings,
        cluster_center.reshape(1, -1),
        metric=metric,
    ).reshape(-1)

    first_local_idx = int(np.argmin(center_distances))
    selected_local = [first_local_idx]

    pairwise_dist = pairwise_distances(
        candidate_embeddings,
        metric=metric,
    )

    while len(selected_local) < n_select:
        distance_to_selected = pairwise_dist[:, selected_local].min(axis=1)
        distance_to_selected[selected_local] = -np.inf

        next_local_idx = int(np.argmax(distance_to_selected))
        selected_local.append(next_local_idx)

    selected_global_indices = candidate_indices[selected_local]

    return list(selected_global_indices)


def select_core_points_from_cluster(
    embeddings,
    candidate_indices,
    cluster_id,
    cluster_centers,
    n_select=50,
    metric="cosine",
):
    """
    Select core points that are:
    1. Close to their own centroid.
    2. Far from other centroids.

    This is more likely to satisfy:
    max intra-selected distance < min inter-selected distance.
    """

    candidate_indices = np.asarray(candidate_indices)

    if len(candidate_indices) <= n_select:
        return list(candidate_indices)

    candidate_embeddings = embeddings[candidate_indices]

    center_distances = pairwise_distances(
        candidate_embeddings,
        cluster_centers,
        metric=metric,
    )

    own_distance = center_distances[:, cluster_id]

    other_cluster_ids = [
        c for c in range(cluster_centers.shape[0])
        if c != cluster_id
    ]

    nearest_other_distance = center_distances[:, other_cluster_ids].min(axis=1)
    margin = nearest_other_distance - own_distance

    # Sort:
    # 1. larger margin first
    # 2. smaller own centroid distance first
    ranking_df = pd.DataFrame({
        "local_idx": np.arange(len(candidate_indices)),
        "own_distance": own_distance,
        "nearest_other_distance": nearest_other_distance,
        "margin": margin,
    })

    ranking_df = ranking_df.sort_values(
        by=["margin", "own_distance"],
        ascending=[False, True],
    )

    selected_local = ranking_df.head(n_select)["local_idx"].to_numpy()
    selected_global_indices = candidate_indices[selected_local]

    return list(selected_global_indices)


def select_points(
    embeddings,
    labels,
    cluster_centers,
    n_select_per_cluster,
    mode,
):
    """
    Selects n points from each cluster.
    """

    selected_rows = []
    selected_summary_rows = []

    unique_clusters = sorted(np.unique(labels))

    for cluster_id in unique_clusters:
        cluster_indices = np.where(labels == cluster_id)[0]

        if mode == "diverse":
            selected_indices = select_diverse_points_from_cluster(
                embeddings=embeddings,
                candidate_indices=cluster_indices,
                cluster_center=cluster_centers[cluster_id],
                n_select=n_select_per_cluster,
                metric="cosine",
            )
        elif mode == "core":
            selected_indices = select_core_points_from_cluster(
                embeddings=embeddings,
                candidate_indices=cluster_indices,
                cluster_id=cluster_id,
                cluster_centers=cluster_centers,
                n_select=n_select_per_cluster,
                metric="cosine",
            )
        else:
            raise ValueError("mode must be either 'diverse' or 'core'.")

        selected_indices = np.asarray(selected_indices)

        center_distances = pairwise_distances(
            embeddings[selected_indices],
            cluster_centers,
            metric="cosine",
        )

        own_center_distances = center_distances[:, cluster_id]
        other_cluster_ids = [
            c for c in unique_clusters
            if c != cluster_id
        ]
        nearest_other_center_distances = center_distances[:, other_cluster_ids].min(axis=1)
        margins = nearest_other_center_distances - own_center_distances

        selected_summary_rows.append({
            "cluster": int(cluster_id),
            "cluster_size": int(len(cluster_indices)),
            "selected_count": int(len(selected_indices)),
            "selection_mode": mode,
            "mean_own_center_cosine_distance": float(np.mean(own_center_distances)),
            "max_own_center_cosine_distance": float(np.max(own_center_distances)),
            "mean_nearest_other_center_cosine_distance": float(np.mean(nearest_other_center_distances)),
            "min_nearest_other_center_cosine_distance": float(np.min(nearest_other_center_distances)),
            "mean_center_margin": float(np.mean(margins)),
            "min_center_margin": float(np.min(margins)),
        })

        for rank, row_id in enumerate(selected_indices):
            selected_rows.append({
                "cluster": int(cluster_id),
                "selection_rank": int(rank),
                "row_id": int(row_id),
                "selection_mode": mode,
                "own_center_cosine_distance": float(own_center_distances[rank]),
                "nearest_other_center_cosine_distance": float(nearest_other_center_distances[rank]),
                "center_margin": float(margins[rank]),
            })

    selected_df = pd.DataFrame(selected_rows)
    selected_summary_df = pd.DataFrame(selected_summary_rows)

    return selected_df, selected_summary_df


# -----------------------------
# Selected 50-50-50 separation report
# -----------------------------

def selected_intra_inter_distance_report(
    embeddings,
    selected_df,
    metric="cosine",
):
    """
    Computes distance diagnostics only among selected examples.

    Reports:
    1. Max distance within selected points of each cluster.
    2. Min distance between selected points of each pair of clusters.
    3. Global condition:
       max intra-selected distance < min inter-selected distance.
    """

    selected_df = selected_df.copy()
    unique_clusters = sorted(selected_df["cluster"].unique())

    intra_rows = []
    inter_rows = []

    # -----------------------------
    # Intra-cluster selected distances
    # -----------------------------

    for cluster_id in unique_clusters:
        cluster_selected = selected_df[selected_df["cluster"] == cluster_id]
        selected_indices = cluster_selected["row_id"].to_numpy()

        if len(selected_indices) <= 1:
            intra_rows.append({
                "cluster": int(cluster_id),
                "selected_count": int(len(selected_indices)),
                "max_intra_selected_cosine_distance": np.nan,
                "mean_intra_selected_cosine_distance": np.nan,
                "min_intra_selected_cosine_distance": np.nan,
                "max_intra_row_id_a": None,
                "max_intra_row_id_b": None,
                "min_intra_row_id_a": None,
                "min_intra_row_id_b": None,
            })
            continue

        selected_embeddings = embeddings[selected_indices]

        dist = pairwise_distances(
            selected_embeddings,
            metric=metric,
        )

        np.fill_diagonal(dist, np.nan)

        max_pos = np.unravel_index(np.nanargmax(dist), dist.shape)
        min_pos = np.unravel_index(np.nanargmin(dist), dist.shape)

        intra_rows.append({
            "cluster": int(cluster_id),
            "selected_count": int(len(selected_indices)),
            "max_intra_selected_cosine_distance": float(np.nanmax(dist)),
            "mean_intra_selected_cosine_distance": float(np.nanmean(dist)),
            "min_intra_selected_cosine_distance": float(np.nanmin(dist)),
            "max_intra_row_id_a": int(selected_indices[max_pos[0]]),
            "max_intra_row_id_b": int(selected_indices[max_pos[1]]),
            "min_intra_row_id_a": int(selected_indices[min_pos[0]]),
            "min_intra_row_id_b": int(selected_indices[min_pos[1]]),
        })

    # -----------------------------
    # Inter-cluster selected distances
    # -----------------------------

    for i, cluster_a in enumerate(unique_clusters):
        for cluster_b in unique_clusters[i + 1:]:
            selected_a = selected_df[selected_df["cluster"] == cluster_a]
            selected_b = selected_df[selected_df["cluster"] == cluster_b]

            indices_a = selected_a["row_id"].to_numpy()
            indices_b = selected_b["row_id"].to_numpy()

            emb_a = embeddings[indices_a]
            emb_b = embeddings[indices_b]

            dist = pairwise_distances(
                emb_a,
                emb_b,
                metric=metric,
            )

            min_pos = np.unravel_index(np.argmin(dist), dist.shape)
            max_pos = np.unravel_index(np.argmax(dist), dist.shape)

            inter_rows.append({
                "cluster_a": int(cluster_a),
                "cluster_b": int(cluster_b),
                "selected_count_a": int(len(indices_a)),
                "selected_count_b": int(len(indices_b)),
                "min_inter_selected_cosine_distance": float(np.min(dist)),
                "mean_inter_selected_cosine_distance": float(np.mean(dist)),
                "max_inter_selected_cosine_distance": float(np.max(dist)),
                "closest_row_id_a": int(indices_a[min_pos[0]]),
                "closest_row_id_b": int(indices_b[min_pos[1]]),
                "farthest_row_id_a": int(indices_a[max_pos[0]]),
                "farthest_row_id_b": int(indices_b[max_pos[1]]),
            })

    intra_df = pd.DataFrame(intra_rows)
    inter_df = pd.DataFrame(inter_rows)

    global_max_intra = float(
        intra_df["max_intra_selected_cosine_distance"].max()
    )

    global_min_inter = float(
        inter_df["min_inter_selected_cosine_distance"].min()
    )

    separation_margin = global_min_inter - global_max_intra

    summary_df = pd.DataFrame([{
        "selection_mode": SELECTION_MODE,
        "global_max_intra_selected_cosine_distance": global_max_intra,
        "global_min_inter_selected_cosine_distance": global_min_inter,
        "separation_margin_min_inter_minus_max_intra": separation_margin,
        "is_strictly_separated": bool(global_max_intra < global_min_inter),
    }])

    return intra_df, inter_df, summary_df


# -----------------------------
# PCA plots
# -----------------------------

def make_pca_plots(
    embeddings,
    labels,
    selected_df,
):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_2d = pca.fit_transform(embeddings)

    pca_df = pd.DataFrame({
        "row_id": np.arange(len(embeddings)),
        "cluster": labels,
        "pca_x": pca_2d[:, 0],
        "pca_y": pca_2d[:, 1],
    })

    pca_df.to_csv(PCA_ALL_CSV_PATH, index=False)

    print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)
    print("PCA total explained variance:", pca.explained_variance_ratio_.sum())

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pca_df["pca_x"],
        pca_df["pca_y"],
        c=pca_df["cluster"],
        alpha=0.8,
        s=12,
    )
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title(f"PCA of all KMeans clusters, k={K}")
    plt.grid(True)
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(PCA_ALL_PNG_PATH, dpi=200)
    plt.show()

    selected_pca_df = selected_df.merge(
        pca_df[["row_id", "pca_x", "pca_y"]],
        on="row_id",
        how="left",
    )

    selected_pca_df.to_csv(PCA_SELECTED_CSV_PATH, index=False)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        selected_pca_df["pca_x"],
        selected_pca_df["pca_y"],
        c=selected_pca_df["cluster"],
        alpha=0.9,
        s=30,
    )
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title(
        f"Selected {N_SELECT_PER_CLUSTER} per cluster, "
        f"k={K}, mode={SELECTION_MODE}"
    )
    plt.grid(True)
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(PCA_SELECTED_PNG_PATH, dpi=200)
    plt.show()

    return pca_df, selected_pca_df


# -----------------------------
# Main
# -----------------------------

def main():
    # -----------------------------
    # 1. Load embeddings/texts
    # -----------------------------

    embeddings = np.load(EMB_PATH)
    texts_df = pd.read_csv(TEXTS_CSV_PATH)

    print("Loaded embeddings:", embeddings.shape)
    print("Loaded texts:", texts_df.shape)

    if len(embeddings) != len(texts_df):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings but {len(texts_df)} text rows."
        )

    if np.isnan(embeddings).any():
        raise ValueError("Embeddings contain NaNs.")

    if len(embeddings) < K:
        raise ValueError(f"Need at least {K} embeddings for KMeans k={K}.")

    print("Embedding std:", embeddings.std())

    # -----------------------------
    # 2. KMeans k=3
    # -----------------------------

    kmeans = KMeans(
        n_clusters=K,
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300,
    )

    labels = kmeans.fit_predict(embeddings)

    assignments_df = texts_df.copy()
    assignments_df["cluster"] = labels

    assignments_df.to_csv(ASSIGNMENTS_CSV_PATH, index=False)

    print("\nSaved assignments:", ASSIGNMENTS_CSV_PATH)
    print("\nCluster counts:")
    print(
        assignments_df
        .groupby("cluster")
        .size()
        .reset_index(name="count")
        .sort_values("cluster")
    )

    # -----------------------------
    # 3. Full cluster distance reports
    # -----------------------------

    full_intra_df, full_inter_df = full_cluster_distance_report(
        embeddings=embeddings,
        labels=labels,
        metric="cosine",
    )

    full_intra_df.to_csv(
        FULL_CLUSTER_DISTANCE_REPORT_CSV_PATH,
        index=False,
    )

    full_inter_df.to_csv(
        FULL_PAIRWISE_DISTANCE_REPORT_CSV_PATH,
        index=False,
    )

    print("\nFull cluster intra-distance report:")
    print(full_intra_df)

    print("\nFull cluster inter-distance report:")
    print(full_inter_df)

    print("\nSaved:", FULL_CLUSTER_DISTANCE_REPORT_CSV_PATH)
    print("Saved:", FULL_PAIRWISE_DISTANCE_REPORT_CSV_PATH)

    # -----------------------------
    # 4. Select 50 elements per cluster
    # -----------------------------

    selected_df, selected_summary_df = select_points(
        embeddings=embeddings,
        labels=labels,
        cluster_centers=kmeans.cluster_centers_,
        n_select_per_cluster=N_SELECT_PER_CLUSTER,
        mode=SELECTION_MODE,
    )

    selected_df = selected_df.merge(
        assignments_df,
        on=["row_id", "cluster"],
        how="left",
    )

    selected_df.to_csv(SELECTED_CSV_PATH, index=False)
    selected_summary_df.to_csv(SELECTED_SUMMARY_CSV_PATH, index=False)

    print("\nSelected counts:")
    print(
        selected_df
        .groupby("cluster")
        .size()
        .reset_index(name="selected_count")
        .sort_values("cluster")
    )

    print("\nSelected summary:")
    print(selected_summary_df)

    print("\nSaved selected examples:", SELECTED_CSV_PATH)
    print("Saved selected summary:", SELECTED_SUMMARY_CSV_PATH)

    # -----------------------------
    # 5. Selected 50-50-50 distance report
    # -----------------------------

    selected_intra_df, selected_inter_df, selected_separation_summary_df = (
        selected_intra_inter_distance_report(
            embeddings=embeddings,
            selected_df=selected_df,
            metric="cosine",
        )
    )

    selected_intra_df.to_csv(
        SELECTED_INTRA_DISTANCE_CSV_PATH,
        index=False,
    )

    selected_inter_df.to_csv(
        SELECTED_INTER_DISTANCE_CSV_PATH,
        index=False,
    )

    selected_separation_summary_df.to_csv(
        SELECTED_SEPARATION_SUMMARY_CSV_PATH,
        index=False,
    )

    print("\nSelected intra-cluster distance report:")
    print(selected_intra_df)

    print("\nSelected inter-cluster distance report:")
    print(selected_inter_df)

    print("\nSelected separation summary:")
    print(selected_separation_summary_df)

    print("\nSaved selected intra report:", SELECTED_INTRA_DISTANCE_CSV_PATH)
    print("Saved selected inter report:", SELECTED_INTER_DISTANCE_CSV_PATH)
    print("Saved selected separation summary:", SELECTED_SEPARATION_SUMMARY_CSV_PATH)

    # -----------------------------
    # 6. PCA plots
    # -----------------------------

    make_pca_plots(
        embeddings=embeddings,
        labels=labels,
        selected_df=selected_df,
    )

    print("\nSaved PCA all CSV:", PCA_ALL_CSV_PATH)
    print("Saved PCA all plot:", PCA_ALL_PNG_PATH)
    print("Saved PCA selected CSV:", PCA_SELECTED_CSV_PATH)
    print("Saved PCA selected plot:", PCA_SELECTED_PNG_PATH)

    # -----------------------------
    # 7. Final interpretation print
    # -----------------------------

    global_max_intra = selected_separation_summary_df.loc[
        0,
        "global_max_intra_selected_cosine_distance",
    ]

    global_min_inter = selected_separation_summary_df.loc[
        0,
        "global_min_inter_selected_cosine_distance",
    ]

    is_separated = selected_separation_summary_df.loc[
        0,
        "is_strictly_separated",
    ]

    print("\nFinal selected-set separation check:")
    print("Global max intra-selected cosine distance:", global_max_intra)
    print("Global min inter-selected cosine distance:", global_min_inter)
    print("Condition max_intra < min_inter:", is_separated)

    if is_separated:
        print("Good: selected groups are strictly separated under this criterion.")
    else:
        print(
            "Not strictly separated: at least one selected pair from different "
            "clusters is closer than the farthest selected pair inside a cluster."
        )
        print(
            "Try SELECTION_MODE='core' if you used 'diverse', or reduce "
            "N_SELECT_PER_CLUSTER."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()