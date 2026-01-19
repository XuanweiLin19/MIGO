"""Unified metrics with task namespaces."""

from types import SimpleNamespace

import numpy as np
import scipy.spatial
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder


def _calculate_performance_metrics(y_true_samples, y_pred_samples):
    metrics_results = []
    for y_true, y_pred in zip(y_true_samples, y_pred_samples):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        metrics_results.append((rmse, mae))
    rmses, maes = zip(*metrics_results)
    return np.mean(rmses), np.mean(maes)


def _cluster_metric(sample_embedding, label_tmp):
    label_encoder = LabelEncoder()
    print("The shape of label_validate_data", label_tmp.shape)
    encoded_labels = label_encoder.fit_transform(label_tmp.reshape(-1,))
    n_clusters = len(set(encoded_labels))

    kmeans = KMeans(n_clusters=n_clusters, random_state=46)
    pred_labels = kmeans.fit_predict(sample_embedding)

    ari = adjusted_rand_score(encoded_labels, pred_labels)
    nmi = normalized_mutual_info_score(encoded_labels, pred_labels)
    ami = adjusted_mutual_info_score(encoded_labels, pred_labels)
    hom = homogeneity_score(encoded_labels, pred_labels)

    return np.round(ari, 4), np.round(nmi, 4), np.round(ami, 4), np.round(hom, 4)


def _foscttm(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    d = scipy.spatial.distance_matrix(x, y, **kwargs)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    return (foscttm_x.mean() + foscttm_y.mean()) / 2


def _calculate_average_correlations_basic(X, Y):
    if X.shape != Y.shape:
        raise ValueError("Input shapes must match.")

    pearson_corrs = []
    spearman_corrs = []
    for i in range(X.shape[0]):
        p_corr, _ = pearsonr(X[i], Y[i])
        s_corr, _ = spearmanr(X[i], Y[i])
        pearson_corrs.append(p_corr)
        spearman_corrs.append(s_corr)

    return np.mean(pearson_corrs), np.mean(spearman_corrs)


def _calculate_average_correlations_robust(X, Y):
    if X.shape != Y.shape:
        raise ValueError("Input shapes must match.")

    if hasattr(X, "cpu"):
        X = X.detach().cpu().numpy()
    if hasattr(Y, "cpu"):
        Y = Y.detach().cpu().numpy()

    X = np.asarray(X)
    Y = np.asarray(Y)

    pearson_corrs = []
    spearman_corrs = []
    for i in range(X.shape[0]):
        x_row = X[i]
        y_row = Y[i]

        if (
            np.any(np.isnan(x_row))
            or np.any(np.isnan(y_row))
            or np.any(np.isinf(x_row))
            or np.any(np.isinf(y_row))
        ):
            continue

        if np.std(x_row) == 0 or np.std(y_row) == 0:
            continue

        try:
            p_corr, _ = pearsonr(x_row, y_row)
            s_corr, _ = spearmanr(x_row, y_row)
            if not np.isnan(p_corr):
                pearson_corrs.append(p_corr)
            if not np.isnan(s_corr):
                spearman_corrs.append(s_corr)
        except (ValueError, RuntimeWarning):
            continue

    avg_pearson = np.mean(pearson_corrs) if pearson_corrs else float("nan")
    avg_spearman = np.mean(spearman_corrs) if spearman_corrs else float("nan")
    return avg_pearson, avg_spearman


def _calculate_average_correlations_try(X, Y):
    if X.shape != Y.shape:
        raise ValueError("Input shapes must match.")

    pearson_corrs = []
    spearman_corrs = []
    for i in range(X.shape[0]):
        try:
            p_corr, _ = pearsonr(X[i], Y[i])
            s_corr, _ = spearmanr(X[i], Y[i])
            if not np.isnan(p_corr):
                pearson_corrs.append(p_corr)
            if not np.isnan(s_corr):
                spearman_corrs.append(s_corr)
        except Exception:
            continue

    avg_pearson = np.mean(pearson_corrs) if pearson_corrs else 0.0
    avg_spearman = np.mean(spearman_corrs) if spearman_corrs else 0.0
    return avg_pearson, avg_spearman


def _calculate_reconstruction_metrics_scatac(original, reconstructed, threshold=0.5):
    if sparse.issparse(original):
        original = original.toarray()
    if sparse.issparse(reconstructed):
        reconstructed = reconstructed.toarray()

    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}")

    original_flat = (original > 0).astype(np.int8).flatten()
    reconstructed_flat = reconstructed.flatten()
    binary_recon = (reconstructed_flat >= threshold).astype(np.int8)

    tp = ((original_flat == 1) & (binary_recon == 1)).sum()
    tn = ((original_flat == 0) & (binary_recon == 0)).sum()
    fp = ((original_flat == 0) & (binary_recon == 1)).sum()
    fn = ((original_flat == 1) & (binary_recon == 0)).sum()
    total = tp + tn + fp + fn
    if total == 0:
        _ = total  # keep behavior aligned with original flow

    auc = 0.0
    aupr = 0.0
    n_pos = original_flat.sum()
    n_total = len(original_flat)
    pos_ratio = n_pos / n_total
    print(f"Positive ratio {pos_ratio:.6f} ({n_pos}/{n_total})")

    if 0 < n_pos < n_total:
        try:
            auc = roc_auc_score(original_flat, reconstructed_flat)
        except ValueError as exc:
            print(f"AUC calculation error: {exc}")

    if n_pos > 0:
        try:
            aupr = average_precision_score(original_flat, reconstructed_flat)
        except ValueError as exc:
            print(f"AUPR calculation error: {exc}")

    return {"auc": round(auc, 4), "aupr": round(aupr, 4)}


def _calculate_feature_correlations_scatac(X, Y):
    if X.shape != Y.shape:
        raise ValueError("Input shapes must match.")

    pearson_corrs = []
    spearman_corrs = []

    for j in range(X.shape[1]):
        try:
            feature_x = X[:, j]
            feature_y = Y[:, j]
            if np.std(feature_x) > 1e-10 and np.std(feature_y) > 1e-10:
                p_corr, _ = pearsonr(feature_x, feature_y)
                s_corr, _ = spearmanr(feature_x, feature_y)
                if not np.isnan(p_corr):
                    pearson_corrs.append(p_corr)
                if not np.isnan(s_corr):
                    spearman_corrs.append(s_corr)
        except Exception:
            continue

    avg_pearson = np.mean(pearson_corrs) if pearson_corrs else 0.0
    avg_spearman = np.mean(spearman_corrs) if spearman_corrs else 0.0
    valid_features = len(pearson_corrs)

    return avg_pearson, avg_spearman, valid_features


def _calculate_unsupervised_clustering_metrics(
    embeddings, true_labels=None, n_clusters_range=None, random_state=42
):
    if hasattr(embeddings, "numpy"):
        embeddings = embeddings.numpy()
    if hasattr(true_labels, "numpy"):
        true_labels = true_labels.numpy()

    embeddings = np.asarray(embeddings)
    n_samples = embeddings.shape[0]

    true_n_clusters = None
    if true_labels is not None:
        true_labels = np.asarray(true_labels).flatten()
        true_n_clusters = len(np.unique(true_labels))
        print(f"Detected true labels: n_clusters={true_n_clusters}")

        try:
            print(f"Clustering with true n_clusters={true_n_clusters}")
            kmeans = KMeans(n_clusters=true_n_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            silhouette = silhouette_score(embeddings, cluster_labels)
            calinski = calinski_harabasz_score(embeddings, cluster_labels)
            davies = davies_bouldin_score(embeddings, cluster_labels)

            print(
                "True k metrics: "
                f"silhouette={silhouette:.4f}, calinski={calinski:.2f}, davies={davies:.4f}"
            )

            return {
                "silhouette_score": round(silhouette, 4),
                "calinski_harabasz_score": round(calinski, 4),
                "davies_bouldin_score": round(davies, 4),
                "n_clusters": true_n_clusters,
                "true_n_clusters": true_n_clusters,
                "use_true_labels": True,
            }
        except Exception as exc:
            print(f"True-k clustering failed: {exc}")

    print("Searching cluster range...")
    if n_clusters_range is None:
        min_clusters = max(2, int(np.sqrt(n_samples) / 4))
        max_clusters = min(int(np.sqrt(n_samples)), 20)
        n_clusters_range = range(min_clusters, max_clusters + 1)

    print(f"Cluster search range: {list(n_clusters_range)}")

    best_metrics = {}
    all_metrics = {}
    for n_clusters in n_clusters_range:
        if n_clusters >= n_samples:
            continue
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            silhouette = silhouette_score(embeddings, cluster_labels)
            calinski = calinski_harabasz_score(embeddings, cluster_labels)
            davies = davies_bouldin_score(embeddings, cluster_labels)

            metrics = {
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski,
                "davies_bouldin_score": davies,
                "n_clusters": n_clusters,
            }
            all_metrics[n_clusters] = metrics

            if not best_metrics or silhouette > best_metrics["silhouette_score"]:
                best_metrics = metrics.copy()
        except Exception as exc:
            print(f"Failed metrics for n_clusters={n_clusters}: {exc}")
            continue

    if not best_metrics:
        best_metrics = {
            "silhouette_score": 0.0,
            "calinski_harabasz_score": 0.0,
            "davies_bouldin_score": float("inf"),
            "n_clusters": 2,
        }

    print(
        f"Best clustering: k={best_metrics['n_clusters']}, "
        f"silhouette={best_metrics['silhouette_score']:.4f}"
    )

    return {
        "silhouette_score": round(best_metrics["silhouette_score"], 4),
        "calinski_harabasz_score": round(best_metrics["calinski_harabasz_score"], 4),
        "davies_bouldin_score": round(best_metrics["davies_bouldin_score"], 4),
        "n_clusters": best_metrics["n_clusters"],
        "true_n_clusters": None,
        "use_true_labels": False,
        "all_clusters_results": all_metrics,
    }


def _calculate_multiple_unsupervised_metrics(embeddings_dict, true_labels=None, n_clusters_range=None):
    results = {}
    for embed_name, embeddings in embeddings_dict.items():
        print("=" * 50)
        print(f"Unsupervised metrics for {embed_name}")
        print("=" * 50)
        try:
            metrics = _calculate_unsupervised_clustering_metrics(
                embeddings, true_labels, n_clusters_range
            )
            result_summary = {
                "silhouette_score": metrics["silhouette_score"],
                "calinski_harabasz_score": metrics["calinski_harabasz_score"],
                "davies_bouldin_score": metrics["davies_bouldin_score"],
                "n_clusters": metrics["n_clusters"],
                "true_n_clusters": metrics["true_n_clusters"],
                "use_true_labels": metrics["use_true_labels"],
            }
            results[embed_name] = result_summary

            if metrics["use_true_labels"]:
                print(f"Using true k={metrics['true_n_clusters']}")
                print(f"  - silhouette: {result_summary['silhouette_score']}")
                print(f"  - calinski: {result_summary['calinski_harabasz_score']}")
                print(f"  - davies: {result_summary['davies_bouldin_score']}")
            else:
                print("Using best k from search")
                print(f"  - best k: {result_summary['n_clusters']}")
                print(f"  - silhouette: {result_summary['silhouette_score']}")
                print(f"  - calinski: {result_summary['calinski_harabasz_score']}")
                print(f"  - davies: {result_summary['davies_bouldin_score']}")
        except Exception as exc:
            print(f"Unsupervised metrics failed for {embed_name}: {exc}")
            results[embed_name] = {
                "silhouette_score": 0.0,
                "calinski_harabasz_score": 0.0,
                "davies_bouldin_score": float("inf"),
                "n_clusters": 2,
                "true_n_clusters": None,
                "use_true_labels": False,
            }

    return results


scRNA_ADT = SimpleNamespace(
    calculate_performance_metrics=_calculate_performance_metrics,
    calculate_average_correlations=_calculate_average_correlations_robust,
    Cluster_metric=_cluster_metric,
    foscttm=_foscttm,
)

scRNA_scATAC = SimpleNamespace(
    calculate_reconstruction_metrics=_calculate_reconstruction_metrics_scatac,
    calculate_performance_metrics=_calculate_performance_metrics,
    calculate_average_correlations=_calculate_average_correlations_try,
    calculate_feature_correlations=_calculate_feature_correlations_scatac,
    calculate_unsupervised_clustering_metrics=_calculate_unsupervised_clustering_metrics,
    calculate_multiple_unsupervised_metrics=_calculate_multiple_unsupervised_metrics,
    Cluster_metric=_cluster_metric,
    foscttm=_foscttm,
)

scRNA_scHiC = SimpleNamespace(
    calculate_performance_metrics=_calculate_performance_metrics,
    calculate_average_correlations=_calculate_average_correlations_basic,
    Cluster_metric=_cluster_metric,
    foscttm=_foscttm,
)

RNA_Ribo_seq = SimpleNamespace(
    calculate_performance_metrics=_calculate_performance_metrics,
    calculate_average_correlations=_calculate_average_correlations_basic,
    Cluster_metric=_cluster_metric,
    foscttm=_foscttm,
)


def get_metrics_module(task_name):
    key = task_name.lower()
    if key == "scrna_adt":
        return scRNA_ADT
    if key == "scrna_scatac":
        return scRNA_scATAC
    if key == "scrna_schic":
        return scRNA_scHiC
    if key in ("rna_ribo_seq", "rna_ribo"):
        return RNA_Ribo_seq
    raise ValueError(f"Unknown task name: {task_name}")


__all__ = [
    "scRNA_ADT",
    "scRNA_scATAC",
    "scRNA_scHiC",
    "RNA_Ribo_seq",
    "get_metrics_module",
]
