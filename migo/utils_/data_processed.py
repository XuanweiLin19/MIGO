"""Unified preprocessing utilities."""

import logging
import os
from typing import Optional

import episcanpy as epi
import numpy as np
import scanpy as sc
import scipy
import sklearn.utils.extmath
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from migo.utils_.log import create_logger


def TFIDF(count_mat):
    if scipy.sparse.issparse(count_mat):
        count_mat = count_mat.toarray()

    count_mat = count_mat.T
    peak_sums = np.sum(count_mat, axis=0)
    divide_title = np.tile(peak_sums + 1e-12, (count_mat.shape[0], 1))
    nfreqs = 1.0 * count_mat / divide_title

    cell_sums = np.sum(count_mat, axis=1)
    multiply_title = np.tile(
        np.log(1 + 1.0 * count_mat.shape[1] / (cell_sums + 1e-12)).reshape(-1, 1),
        (1, count_mat.shape[1]),
    )

    tfidf_values = np.multiply(nfreqs, multiply_title)
    if np.any(np.isnan(tfidf_values)) or np.any(np.isinf(tfidf_values)):
        raise ValueError("TF-IDF calculation resulted in NaN or inf values.")

    tfidf_mat = scipy.sparse.csr_matrix(tfidf_values).T
    return tfidf_mat, divide_title, multiply_title


def inverse_TFIDF(TDIDFed_mat, divide_title, multiply_title, max_temp):
    count_mat = TDIDFed_mat.T
    count_mat = count_mat * max_temp
    nfreqs = np.divide(count_mat, multiply_title)
    count_mat = np.multiply(nfreqs, divide_title).T
    return count_mat


def custom_sort(chrom):
    chrom_str = str(chrom)
    if chrom_str.startswith("chr"):
        suffix = chrom_str[3:]
        try:
            num = int(suffix)
            return (0, 0, num)
        except ValueError:
            return (0, 1, suffix)
    return (1, 0, chrom_str)


def lsi(
    adata: AnnData,
    n_components: int = 256,
    use_highly_variable: Optional[bool] = None,
    exclude_first: bool = True,
    **kwargs,
) -> None:
    X = adata.X
    n_components_plus = n_components + 1 if exclude_first else n_components
    U, S, Vt = sklearn.utils.extmath.randomized_svd(X, n_components_plus, **kwargs)
    start_idx = 1 if exclude_first else 0
    X_lsi = U[:, start_idx:n_components + start_idx]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi


def robust_outlier_detection(X, method="iqr", percentile_range=(5, 95)):
    if issparse(X):
        X = X.toarray()

    if method == "iqr":
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
    elif method == "percentile":
        lower_bound = np.percentile(X, percentile_range[0], axis=0)
        upper_bound = np.percentile(X, percentile_range[1], axis=0)
    elif method == "mad":
        median = np.median(X, axis=0)
        mad = np.median(np.abs(X - median), axis=0)
        factor = 1.4826
        lower_bound = median - 3 * factor * mad
        upper_bound = median + 3 * factor * mad
    else:
        raise ValueError(f"Unsupported method: {method}")

    return lower_bound, upper_bound


def adaptive_clip_outliers(X, method="adaptive_percentile", factor=2.0):
    if issparse(X):
        X = X.toarray()

    X_clipped = X.copy()
    outlier_stats = {"features_clipped": [], "clip_ratios": []}

    for i in range(X.shape[1]):
        feature_data = X[:, i]
        q50 = np.percentile(feature_data, 50)
        q90 = np.percentile(feature_data, 90)
        q99 = np.percentile(feature_data, 99)

        if q99 > 10 * q90:
            lower, upper = robust_outlier_detection(
                feature_data.reshape(-1, 1),
                method="percentile",
                percentile_range=(2, 95),
            )
        elif q90 > 5 * q50:
            lower, upper = robust_outlier_detection(
                feature_data.reshape(-1, 1),
                method="percentile",
                percentile_range=(5, 97),
            )
        else:
            lower, upper = robust_outlier_detection(
                feature_data.reshape(-1, 1),
                method="percentile",
                percentile_range=(1, 99),
            )

        original_data = feature_data.copy()
        X_clipped[:, i] = np.clip(feature_data, lower[0], upper[0])

        clipped_ratio = np.sum(original_data != X_clipped[:, i]) / len(original_data)
        outlier_stats["features_clipped"].append(i)
        outlier_stats["clip_ratios"].append(clipped_ratio)

    return X_clipped, outlier_stats


def enhanced_clr_normalize(X, pseudocount=1):
    if issparse(X):
        X = X.toarray()

    min_nonzero = X[X > 0].min() if np.any(X > 0) else 1
    adaptive_pseudocount = min(pseudocount, min_nonzero / 2)

    X_pseudo = X + adaptive_pseudocount
    geometric_mean = np.exp(np.mean(np.log(X_pseudo), axis=1, keepdims=True))
    X_clr = np.log(X_pseudo / geometric_mean)
    return X_clr


def clip_outliers(X, lower=0.01, upper=0.99):
    if issparse(X):
        X = X.toarray()
    X_clipped = X.copy()
    for i in range(X.shape[1]):
        low = np.quantile(X[:, i], lower)
        high = np.quantile(X[:, i], upper)
        X_clipped[:, i] = np.clip(X[:, i], low, high)
    return X_clipped


def clr_normalize(X):
    if issparse(X):
        X = X.toarray()

    X_pseudo = X + 1
    geometric_mean = np.exp(np.mean(np.log(X_pseudo), axis=1, keepdims=True))
    X_clr = np.log(X_pseudo / geometric_mean)
    return X_clr


def RNA_data_preprocessing_adt(
    RNA_data,
    normalize_total=True,
    log1p=True,
    use_hvg=True,
    n_top_genes=3000,
    standard_scale=True,
    save_data=False,
    file_path=None,
    logging_path=None,
    PCA_reduction=True,
    PCA_n_comps=256,
    apply_outlier_clipping=True,
    save_log_target=True,
):
    RNA_data_processed = RNA_data.copy()

    if issparse(RNA_data.X):
        RNA_data_processed.layers["counts_original"] = RNA_data.X.toarray().copy()
    else:
        RNA_data_processed.layers["counts_original"] = RNA_data.X.copy()

    RNA_data_processed.var_names_make_unique()
    my_logger = create_logger(name="RNA preprocessing", ch=True, fh=False, levelname=logging.INFO, overwrite=False)

    if logging_path is not None:
        with open(os.path.join(logging_path, "Parameters_RNA_Record.txt"), mode="w") as file_handle:
            file_handle.writelines(
                [
                    "------------------------------\n",
                    "RNA Preprocessing\n",
                    f"normalize_total: {normalize_total}\n",
                    f"log1p: {log1p}\n",
                    f"use_hvg: {use_hvg}\n",
                    f"standard_scale: {standard_scale}\n",
                    f"n_top_genes: {n_top_genes}\n",
                    f"apply_outlier_clipping: {apply_outlier_clipping}\n",
                    f"save_log_target: {save_log_target}\n",
                ]
            )

    if apply_outlier_clipping:
        my_logger.info("Applying RNA outlier clipping.")
        X_dense = RNA_data_processed.X.toarray() if issparse(RNA_data_processed.X) else RNA_data_processed.X
        X_clipped = clip_outliers(X_dense, lower=0.005, upper=0.995)
        RNA_data_processed.X = X_clipped

    if normalize_total:
        my_logger.info("Normalize size factor.")
        sc.pp.normalize_total(RNA_data_processed)

    if log1p:
        my_logger.info("Log transform RNA data.")
        sc.pp.log1p(RNA_data_processed)
        if issparse(RNA_data_processed.X):
            RNA_data_processed.layers["counts"] = RNA_data_processed.X.toarray().copy()
        else:
            RNA_data_processed.layers["counts"] = RNA_data_processed.X.copy()

    if use_hvg:
        my_logger.info(f"Select top {n_top_genes} genes.")
        sc.pp.highly_variable_genes(RNA_data_processed, n_top_genes=n_top_genes)
        RNA_data_processed = RNA_data_processed[:, RNA_data_processed.var["highly_variable"]]

    if standard_scale:
        my_logger.info("Standard scale RNA data.")
        sc.pp.scale(RNA_data_processed)

    if PCA_reduction:
        my_logger.info("Reduce dimension by PCA.")
        sc.tl.pca(RNA_data_processed, n_comps=PCA_n_comps, svd_solver="auto")

    if save_data and file_path is not None:
        my_logger.warning("Writing processed RNA data to target file.")
        if use_hvg:
            RNA_data_processed.write_h5ad(
                os.path.join(
                    file_path,
                    f"normalize_{normalize_total}_log1p_{log1p}_hvg_{use_hvg}_{n_top_genes}_RNA_processed_data.h5ad",
                )
            )
        else:
            RNA_data_processed.write_h5ad(
                os.path.join(
                    file_path,
                    f"normalize_{normalize_total}_log1p_{log1p}_hvg_{use_hvg}_RNA_processed_data.h5ad",
                )
            )

    return RNA_data_processed


def RNA_data_preprocessing_scatac(
    RNA_data,
    normalize_total=True,
    log1p=True,
    use_hvg=True,
    n_top_genes=3000,
    standard_scale=False,
    save_data=False,
    file_path=None,
    logging_path=None,
    PCA_reduction=True,
):
    RNA_data_processed = RNA_data.copy()
    RNA_data_processed.var_names_make_unique()
    my_logger = create_logger(name="RNA preprocessing", ch=True, fh=False, levelname=logging.INFO, overwrite=False)

    if logging_path is not None:
        with open(os.path.join(logging_path, "Parameters_RNA_Record.txt"), mode="w") as file_handle:
            file_handle.writelines(
                [
                    "------------------------------\n",
                    "RNA Preprocessing\n",
                    f"normalize_total: {normalize_total}\n",
                    f"log1p: {log1p}\n",
                    f"use_hvg: {use_hvg}\n",
                    f"standard_scale: {standard_scale}\n",
                    f"n_top_genes: {n_top_genes}\n",
                ]
            )

    if normalize_total:
        my_logger.info("Normalize size factor.")
        sc.pp.normalize_total(RNA_data_processed)
    if log1p:
        my_logger.info("Log transform RNA data.")
        sc.pp.log1p(RNA_data_processed)
    if use_hvg:
        my_logger.info(f"Select top {n_top_genes} genes.")
        sc.pp.highly_variable_genes(RNA_data_processed, n_top_genes=n_top_genes)
        RNA_data_processed = RNA_data_processed[:, RNA_data_processed.var["highly_variable"]]

    RNA_data_processed.layers["counts"] = RNA_data_processed.X.toarray().copy()

    if standard_scale:
        my_logger.info("Standard scale RNA data.")
        sc.pp.scale(RNA_data_processed)
    if PCA_reduction:
        my_logger.info("Reduce dimension by PCA.")
        sc.tl.pca(RNA_data_processed, n_comps=256, svd_solver="auto")

    if save_data and file_path is not None:
        my_logger.warning("Writing processed RNA data to target file.")
        suffix = f"_hvg_{use_hvg}"
        if use_hvg:
            suffix = f"_hvg_{use_hvg}_{n_top_genes}"
        RNA_data_processed.write_h5ad(
            os.path.join(
                file_path,
                f"normalize_{normalize_total}_log1p_{log1p}{suffix}_RNA_processed_data.h5ad",
            )
        )

    return RNA_data_processed


def RNA_data_preprocessing_schic(
    RNA_data,
    normalize_total=True,
    log1p=True,
    use_hvg=True,
    n_top_genes=3000,
    standard_scale=True,
    save_data=False,
    file_path=None,
    logging_path=None,
    PCA_reduction=True,
):
    RNA_data_processed = RNA_data.copy()
    RNA_data_processed.layers["counts"] = RNA_data_processed.X.toarray().copy()
    RNA_data_processed.var_names_make_unique()
    my_logger = create_logger(name="RNA preprocessing", ch=True, fh=False, levelname=logging.INFO, overwrite=False)

    if logging_path is not None:
        with open(os.path.join(logging_path, "Parameters_RNA_Record.txt"), mode="w") as file_handle:
            file_handle.writelines(
                [
                    "------------------------------\n",
                    "RNA Preprocessing\n",
                    f"normalize_total: {normalize_total}\n",
                    f"log1p: {log1p}\n",
                    f"use_hvg: {use_hvg}\n",
                    f"standard_scale: {standard_scale}\n",
                    f"n_top_genes: {n_top_genes}\n",
                ]
            )

    if normalize_total:
        my_logger.info("Normalize size factor.")
        sc.pp.normalize_total(RNA_data_processed)
    if log1p:
        my_logger.info("Log transform RNA data.")
        sc.pp.log1p(RNA_data_processed)
    if use_hvg:
        my_logger.info(f"Select top {n_top_genes} genes.")
        sc.pp.highly_variable_genes(RNA_data_processed, n_top_genes=n_top_genes)
        RNA_data_processed = RNA_data_processed[:, RNA_data_processed.var["highly_variable"]]
    if standard_scale:
        my_logger.info("Standard scale RNA data.")
        sc.pp.scale(RNA_data_processed)
    if PCA_reduction:
        my_logger.info("Reduce dimension by PCA.")
        sc.tl.pca(RNA_data_processed, n_comps=256, svd_solver="auto")

    if save_data and file_path is not None:
        my_logger.warning("Writing processed RNA data to target file.")
        suffix = f"_hvg_{use_hvg}"
        if use_hvg:
            suffix = f"_hvg_{use_hvg}_{n_top_genes}"
        RNA_data_processed.write_h5ad(
            os.path.join(
                file_path,
                f"normalize_{normalize_total}_log1p_{log1p}{suffix}_RNA_processed_data.h5ad",
            )
        )

    return RNA_data_processed


def RNA_data_preprocessing_ribo(*args, **kwargs):
    return RNA_data_preprocessing_schic(*args, **kwargs)


def RNA_data_preprocessing(*args, task=None, **kwargs):
    if task is None:
        if "apply_outlier_clipping" in kwargs or "PCA_n_comps" in kwargs or "save_log_target" in kwargs:
            task = "scRNA_ADT"
        else:
            task = "scRNA_scATAC"

    task = task.lower()
    if task in ("scrna_adt", "adt", "scrna_adt_task"):
        return RNA_data_preprocessing_adt(*args, **kwargs)
    if task in ("scrna_schic", "schic"):
        return RNA_data_preprocessing_schic(*args, **kwargs)
    if task in ("rna_ribo_seq", "ribo", "rna_ribo"):
        return RNA_data_preprocessing_ribo(*args, **kwargs)
    return RNA_data_preprocessing_scatac(*args, **kwargs)


def ATAC_data_preprocessing(
    ATAC_data,
    binary_data=True,
    filter_features=True,
    fpeaks=0.02,
    tfidf=True,
    normalize=False,
    save_data=False,
    file_path=None,
    logging_path=None,
    LSI_Process=True,
):
    ATAC_data_processed = ATAC_data.copy()
    divide_title, multiply_title, max_temp = None, None, None

    my_logger = create_logger(name="ATAC preprocessing", ch=True, fh=False, levelname=logging.INFO, overwrite=False)
    if logging_path is not None:
        with open(os.path.join(logging_path, "Parameters_ATAC_Record.txt"), mode="a") as file_handle:
            file_handle.writelines(
                [
                    "------------------------------\n",
                    "ATAC Preprocessing\n",
                    f"binary_data: {binary_data}\n",
                    f"filter_features: {filter_features}\n",
                    f"fpeaks: {fpeaks}\n",
                    f"tfidf: {tfidf}\n",
                    f"normalize: {normalize}\n",
                ]
            )

    if binary_data:
        my_logger.info("Binarizing data.")
        epi.pp.binarize(ATAC_data_processed)

    ATAC_data_processed.layers["counts"] = ATAC_data_processed.X.toarray().copy()

    if tfidf:
        my_logger.info("TF-IDF transformation.")
        count_mat = ATAC_data_processed.X.copy()
        ATAC_data_processed.X, divide_title, multiply_title = TFIDF(count_mat)

    if filter_features:
        my_logger.info(f"Filter peaks lower than {fpeaks * 100}% cells.")
        epi.pp.filter_features(ATAC_data_processed, min_cells=np.ceil(fpeaks * ATAC_data.shape[0]))

    if normalize:
        my_logger.info("Normalizing data.")
        max_temp = np.max(ATAC_data_processed.X)
        ATAC_data_processed.X = ATAC_data_processed.X / max_temp

    if LSI_Process:
        my_logger.info("LSI transformation.")
        lsi(ATAC_data_processed)

    if save_data and file_path is not None:
        my_logger.warning("Writing processed ATAC data to target file.")
        ATAC_data_processed.write_h5ad(
            os.path.join(
                file_path,
                f"binarize_{binary_data}_filter_{filter_features}_fpeaks_{fpeaks}_tfidf_{tfidf}_normalize_{LSI_Process}_ATAC_processed_data.h5ad",
            )
        )

    my_logger.info("ATAC processing complete.")
    return ATAC_data_processed, divide_title, multiply_title, max_temp


def preprocess_ribo_data(ribo_adata, n_components=128, use_hvg=True, n_hvg=2000):
    sc.pp.filter_cells(ribo_adata, min_genes=200)
    sc.pp.filter_genes(ribo_adata, min_cells=3)

    ribo_adata.raw = ribo_adata
    ribo_adata.layers["counts"] = ribo_adata.X.copy()

    sc.pp.normalize_total(ribo_adata, target_sum=1e4)
    sc.pp.log1p(ribo_adata)

    if use_hvg:
        sc.pp.highly_variable_genes(ribo_adata, n_top_genes=n_hvg, flavor="seurat_v3")
        ribo_adata = ribo_adata[:, ribo_adata.var.highly_variable]

    sc.pp.scale(ribo_adata, max_value=10)
    sc.tl.pca(ribo_adata, n_comps=n_components)
    return ribo_adata


def RNA_Ribo_unified_preprocessing(
    adata,
    data_type="RNA",
    use_hvg=False,
    n_top_genes=3000,
    standard_scale=True,
    pca_reduction=True,
    n_pca_components=256,
    save_data=False,
    file_path=None,
    logging_path=None,
):
    adata_processed = adata.copy()
    my_logger = create_logger(
        name=f"{data_type} preprocessing",
        ch=True,
        fh=False,
        levelname=logging.INFO,
        overwrite=False,
    )

    if logging_path is not None:
        with open(os.path.join(logging_path, f"Parameters_{data_type}_Record.txt"), mode="w") as file_handle:
            file_handle.writelines(
                [
                    "------------------------------\n",
                    f"{data_type} Unified Preprocessing\n",
                    "Data already CLR transformed: True\n",
                    f"use_hvg: {use_hvg}\n",
                    f"n_top_genes: {n_top_genes}\n",
                    f"standard_scale: {standard_scale}\n",
                    f"pca_reduction: {pca_reduction}\n",
                    f"n_pca_components: {n_pca_components}\n",
                ]
            )

    if issparse(adata_processed.X):
        adata_processed.layers["counts"] = adata_processed.X.copy()
    else:
        adata_processed.layers["counts"] = csr_matrix(adata_processed.X.copy())

    adata_processed.var_names_make_unique()

    n_genes_before = adata_processed.shape[1]
    if issparse(adata_processed.X):
        gene_counts = np.array((adata_processed.X != adata_processed.X.min()).sum(axis=0)).flatten()
    else:
        gene_counts = np.sum(adata_processed.X != adata_processed.X.min(), axis=0)

    min_cells_threshold = max(3, int(0.01 * adata_processed.shape[0]))
    genes_to_keep = gene_counts >= min_cells_threshold
    adata_processed = adata_processed[:, genes_to_keep].copy()
    my_logger.info(f"Filtered genes: {n_genes_before} -> {adata_processed.shape[1]}")

    if use_hvg:
        if issparse(adata_processed.X):
            mean = np.array(adata_processed.X.mean(axis=0)).flatten()
            mean_sq = np.array((adata_processed.X.multiply(adata_processed.X)).mean(axis=0)).flatten()
            var_scores = mean_sq - mean**2
        else:
            var_scores = np.var(adata_processed.X, axis=0)

        n_genes_to_select = min(n_top_genes, len(var_scores))
        top_genes_idx = np.argsort(var_scores)[::-1][:n_genes_to_select]
        adata_processed.var["highly_variable"] = False
        adata_processed.var["highly_variable"].iloc[top_genes_idx] = True
        adata_processed.var["variance_scores"] = var_scores
        adata_processed = adata_processed[:, adata_processed.var["highly_variable"]].copy()

    if standard_scale:
        sc.pp.scale(adata_processed, max_value=10)

    if pca_reduction:
        sc.tl.pca(adata_processed, n_comps=n_pca_components, svd_solver="auto")

    if save_data and file_path is not None:
        suffix = f"_hvg_{n_top_genes}" if use_hvg else "_no_hvg"
        suffix += f"_scale_{standard_scale}_pca_{pca_reduction}"
        filename = f"{data_type}_CLR_processed{suffix}.h5ad"
        adata_processed.write_h5ad(os.path.join(file_path, filename))

    return adata_processed


def preprocess_ribo_data_clr(ribo_adata, n_components=256, use_hvg=True, n_hvg=3000):
    return RNA_Ribo_unified_preprocessing(
        ribo_adata,
        data_type="Ribo",
        use_hvg=use_hvg,
        n_top_genes=n_hvg,
        standard_scale=True,
        pca_reduction=True,
        n_pca_components=n_components,
        save_data=False,
    )


def process_paired_RNA_Ribo_data(rna_adata, ribo_adata, analysis_type="integration"):
    if analysis_type == "integration":
        common_genes = rna_adata.var_names.intersection(ribo_adata.var_names)
        rna_common = rna_adata[:, common_genes].copy()
        ribo_common = ribo_adata[:, common_genes].copy()
        rna_processed = RNA_Ribo_unified_preprocessing(rna_common, data_type="RNA", use_hvg=False)
        ribo_processed = RNA_Ribo_unified_preprocessing(ribo_common, data_type="Ribo", use_hvg=False)
        return rna_processed, ribo_processed

    rna_processed = RNA_Ribo_unified_preprocessing(rna_adata, data_type="RNA", use_hvg=True, n_top_genes=3000)
    ribo_processed = RNA_Ribo_unified_preprocessing(ribo_adata, data_type="Ribo", use_hvg=True, n_top_genes=3000)
    return rna_processed, ribo_processed


def process_adt_data(
    adata,
    min_cells=1,
    target_comps=None,
    save_path=None,
    logging_path=None,
    apply_outlier_clipping=True,
    use_enhanced_clr=True,
    save_diagnostics=True,
):
    logger = create_logger(name="ADT preprocessing", ch=True, fh=False, levelname=logging.INFO, overwrite=False)
    if logging_path is not None:
        with open(os.path.join(logging_path, "Parameters_ADT_Record.txt"), mode="w") as file_handle:
            file_handle.writelines(
                [
                    "------------------------------\n",
                    "ADT Preprocessing\n",
                    f"min_cells: {min_cells}\n",
                    f"target_comps: {target_comps}\n",
                    f"apply_outlier_clipping: {apply_outlier_clipping}\n",
                    f"use_enhanced_clr: {use_enhanced_clr}\n",
                    f"save_diagnostics: {save_diagnostics}\n",
                ]
            )

    adata_processed = adata.copy()
    if issparse(adata_processed.X):
        adata_processed.layers["counts_original"] = adata_processed.X.toarray().copy()
    else:
        adata_processed.layers["counts_original"] = adata_processed.X.copy()

    if min_cells > 0:
        sc.pp.filter_genes(adata_processed, min_cells=min_cells)

    if apply_outlier_clipping:
        X_dense = adata_processed.X.toarray() if issparse(adata_processed.X) else adata_processed.X
        X_clipped = clip_outliers(X_dense, lower=0.005, upper=0.995)
        adata_processed.X = X_clipped

    if use_enhanced_clr:
        adata_processed.X = enhanced_clr_normalize(adata_processed.X)
    else:
        adata_processed.X = clr_normalize(adata_processed.X)

    if issparse(adata_processed.X):
        adata_processed.layers["counts_raw"] = adata_processed.X.toarray().copy()
    else:
        adata_processed.layers["counts_raw"] = adata_processed.X.copy()

    if target_comps:
        sc.tl.pca(adata_processed, n_comps=target_comps, svd_solver="auto")

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        adata_processed.write_h5ad(os.path.join(save_path, "ADT_processed_data.h5ad"))
        if save_diagnostics:
            create_consistency_diagnostics(adata_processed, save_path, logger)

    return adata_processed


def create_consistency_diagnostics(adata, save_path, logger):
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        original_data = adata.layers["counts_original"].flatten()
        original_nonzero = original_data[original_data > 0]

        axes[0, 0].hist(np.log1p(original_nonzero), bins=50, alpha=0.7, color="red", label="Original")
        target_data = adata.layers["counts_raw"].flatten()
        axes[0, 0].hist(target_data, bins=50, alpha=0.7, color="blue", label="Target")
        axes[0, 0].set_title("Data Distribution Comparison")
        axes[0, 0].set_xlabel("log1p(counts)")
        axes[0, 0].legend()

        train_data = adata.X.flatten()
        axes[0, 1].hist(train_data, bins=50, alpha=0.7, color="green")
        axes[0, 1].set_title("Training Data (CLR normalized)")
        axes[0, 1].set_xlabel("CLR values")

        ranges = {
            "Original": [adata.layers["counts_original"].min(), adata.layers["counts_original"].max()],
            "Target": [adata.layers["counts_raw"].min(), adata.layers["counts_raw"].max()],
            "Training": [adata.X.min(), adata.X.max()],
        }

        labels = list(ranges.keys())
        mins = [ranges[k][0] for k in labels]
        maxs = [ranges[k][1] for k in labels]

        x_pos = np.arange(len(labels))
        axes[0, 2].bar(x_pos - 0.2, mins, 0.4, label="Min", alpha=0.7)
        axes[0, 2].bar(x_pos + 0.2, maxs, 0.4, label="Max", alpha=0.7)
        axes[0, 2].set_title("Value Range Comparison")
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(labels)
        axes[0, 2].legend()
        axes[0, 2].set_yscale("symlog")

        target_means = np.mean(adata.layers["counts_raw"], axis=0)
        clr_means = np.mean(adata.X, axis=0)

        axes[1, 0].scatter(target_means, clr_means, alpha=0.6)
        correlation = np.corrcoef(target_means, clr_means)[0, 1]
        axes[1, 0].set_title(f"Target vs Training Correlation: {correlation:.3f}")
        axes[1, 0].set_xlabel("Target Mean")
        axes[1, 0].set_ylabel("Training Mean")

        if "pca" in adata.uns:
            pca_var = adata.uns["pca"]["variance_ratio"][:15]
            axes[1, 1].bar(range(len(pca_var)), pca_var)
            axes[1, 1].set_title("PCA Variance Explained")
            axes[1, 1].set_xlabel("PC")
            axes[1, 1].set_ylabel("Variance Ratio")

        if "X_umap" in adata.obsm and "protein_clusters" in adata.obs:
            axes[1, 2].scatter(
                adata.obsm["X_umap"][:, 0],
                adata.obsm["X_umap"][:, 1],
                c=adata.obs["protein_clusters"].astype("category").cat.codes,
                s=1,
                alpha=0.6,
                cmap="tab20",
            )
            axes[1, 2].set_title("UMAP Clustering")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "adt_consistency_diagnostics.png"), dpi=300, bbox_inches="tight")
        plt.close()

        with open(os.path.join(save_path, "consistency_report.txt"), "w") as handle:
            handle.write("Data consistency report\n")
            handle.write("=" * 40 + "\n")
            handle.write("Training data and reconstruction targets are aligned.\n")
            handle.write(f"Target-training correlation: {correlation:.3f}\n")
            handle.write(
                f"Target range: [{adata.layers['counts_raw'].min():.4f}, {adata.layers['counts_raw'].max():.4f}]\n"
            )

        logger.info("Consistency diagnostics saved.")
    except Exception as exc:
        logger.warning(f"Consistency diagnostics failed: {exc}")


__all__ = [
    "TFIDF",
    "inverse_TFIDF",
    "custom_sort",
    "lsi",
    "RNA_data_preprocessing",
    "RNA_data_preprocessing_adt",
    "RNA_data_preprocessing_scatac",
    "RNA_data_preprocessing_schic",
    "RNA_data_preprocessing_ribo",
    "ATAC_data_preprocessing",
    "preprocess_ribo_data",
    "RNA_Ribo_unified_preprocessing",
    "preprocess_ribo_data_clr",
    "process_paired_RNA_Ribo_data",
    "robust_outlier_detection",
    "adaptive_clip_outliers",
    "enhanced_clr_normalize",
    "clip_outliers",
    "clr_normalize",
    "process_adt_data",
    "create_consistency_diagnostics",
]
