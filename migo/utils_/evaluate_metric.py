"""Unified evaluation helpers."""

import os
from datetime import datetime

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

from migo.model.metrics import (
    RNA_Ribo_seq as metrics_rna_ribo,
    scRNA_ADT as metrics_scRNA_adt,
    scRNA_scATAC as metrics_scRNA_scatac,
    scRNA_scHiC as metrics_scRNA_schic,
)


def _to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def _mi_computer(x, y):
    mi_values = []
    for i in range(x.shape[1]):
        mi = mutual_info_regression(x[:, i].reshape(-1, 1), y[:, i])
        mi_values.append(mi[0])
    return np.mean(mi_values)


def _feature_wise_correlations(original, reconstructed):
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    original = to_numpy(original)
    reconstructed = to_numpy(reconstructed)

    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    pearson_corrs = []
    spearman_corrs = []
    for i in range(original.shape[1]):
        orig_feature = original[:, i]
        recon_feature = reconstructed[:, i]
        if np.var(orig_feature) > 1e-8 and np.var(recon_feature) > 1e-8:
            try:
                pearson_corr, _ = pearsonr(orig_feature, recon_feature)
                spearman_corr, _ = spearmanr(orig_feature, recon_feature)
                if not np.isnan(pearson_corr):
                    pearson_corrs.append(pearson_corr)
                if not np.isnan(spearman_corr):
                    spearman_corrs.append(spearman_corr)
            except Exception:
                continue

    pearson_mean = np.nanmean(pearson_corrs) if pearson_corrs else 0.0
    spearman_mean = np.nanmean(spearman_corrs) if spearman_corrs else 0.0
    return pearson_mean, spearman_mean


def _insulation_corr(preds, targets, method="pearson"):
    from migo.tasks.scRNA_scHiC import insulation_score as insu

    scores = []
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    for pred, target in zip(preds, targets):
        if pred.ndim == 1:
            size = int(np.sqrt(pred.shape[0]))
            if size * size == pred.shape[0]:
                pred = pred.reshape(size, size)
        if target.ndim == 1:
            size = int(np.sqrt(target.shape[0]))
            if size * size == target.shape[0]:
                target = target.reshape(size, size)
        pred_insu = np.array(insu.chr_score(pred))
        label_insu = np.array(insu.chr_score(target))
        nas = np.logical_or(np.isnan(pred_insu), np.isnan(label_insu))
        if nas.sum() == len(pred_insu):
            scores.append(np.nan)
            continue
        if method == "pearson":
            metric, _ = pearsonr(pred_insu[~nas], label_insu[~nas])
        else:
            metric, _ = spearmanr(pred_insu[~nas], label_insu[~nas])
        if np.isnan(metric) or np.isinf(metric):
            scores.append(np.nan)
        else:
            scores.append(metric)
    valid_scores = [s for s in scores if not np.isnan(s)]
    return valid_scores


def _observed_vs_expected(preds, targets):
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    preds_mean = preds.mean(axis=0, keepdims=True)
    targets_mean = targets.mean(axis=0, keepdims=True)

    scores = []
    for pred, target in zip(preds - preds_mean, targets - targets_mean):
        if pred.ndim == 1:
            size = int(np.sqrt(pred.shape[0]))
            if size * size == pred.shape[0]:
                pred = pred.reshape(size, size)
        if target.ndim == 1:
            size = int(np.sqrt(target.shape[0]))
            if size * size == target.shape[0]:
                target = target.reshape(size, size)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        if np.std(pred_flat) > 1e-8 and np.std(target_flat) > 1e-8:
            metric, _ = pearsonr(pred_flat, target_flat)
            if not np.isnan(metric) and not np.isinf(metric):
                scores.append(metric)
    return scores


def _save_metrics_to_txt(metrics, save_dir="./results", filename=None):
    os.makedirs(save_dir, exist_ok=True)
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_omics_metrics_{timestamp}.txt"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "w", encoding="utf-8") as handle:
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if isinstance(value, (float, np.floating)):
                value_str = f"{value:.6f}"
            else:
                value_str = str(value)
            handle.write(f"{key}: {value_str}\n")

    print(f"Saved metrics report to: {filepath}")


def evaluate_metrics_scRNA_ADT(
    scRNA_semantic_tmp,
    scRNA_encoder_tmp,
    ADT_semantic_tmp,
    ADT_encoder_tmp,
    emd_scRNA_data_tmp,
    emd_ADT_data_tmp,
    cpc_scRNA_result_tmp,
    cpc_ADT_result_tmp,
    original_scRNA_data_tmp,
    scRNA_recon_result_tmp,
    original_ADT_data_tmp,
    ADT_recon_result_tmp,
    Cross_scRNA_recon_result_tmp,
    Cross_ADT_recon_result_tmp,
    path_to_save,
):
    scRNA_semantic_tmp = _to_numpy(scRNA_semantic_tmp)
    scRNA_encoder_tmp = _to_numpy(scRNA_encoder_tmp)
    ADT_semantic_tmp = _to_numpy(ADT_semantic_tmp)
    ADT_encoder_tmp = _to_numpy(ADT_encoder_tmp)
    cpc_scRNA_result_tmp = _to_numpy(cpc_scRNA_result_tmp)
    cpc_ADT_result_tmp = _to_numpy(cpc_ADT_result_tmp)
    original_ADT_data_tmp = _to_numpy(original_ADT_data_tmp)
    Cross_ADT_recon_result_tmp = _to_numpy(Cross_ADT_recon_result_tmp)

    metrics = {}
    metrics["scRNA_sp_sm_MI"] = _mi_computer(scRNA_semantic_tmp, scRNA_encoder_tmp)
    metrics["ADT_sp_sm_MI"] = _mi_computer(ADT_semantic_tmp, ADT_encoder_tmp)
    metrics["RTA_rna_atac_MI"] = _mi_computer(cpc_scRNA_result_tmp, cpc_ADT_result_tmp)

    metrics["ADT_Cross_pearson_corr"], metrics["ADT_Cross_spearman_corr"] = (
        metrics_scRNA_adt.calculate_average_correlations(original_ADT_data_tmp, Cross_ADT_recon_result_tmp)
    )

    _save_metrics_to_txt(metrics, save_dir=path_to_save, filename="multi_omics_metrics.txt")
    return metrics


def evaluate_metrics_RNA_Ribo_seq(
    scRNA_semantic_tmp,
    scRNA_encoder_tmp,
    Ribo_semantic_tmp,
    Ribo_encoder_tmp,
    emd_scRNA_data_tmp,
    emd_Ribo_data_tmp,
    cpc_scRNA_result_tmp,
    cpc_Ribo_result_tmp,
    original_scRNA_data_tmp,
    scRNA_recon_result_tmp,
    original_Ribo_data_tmp,
    Ribo_recon_result_tmp,
    Cross_scRNA_recon_result_tmp,
    Cross_Ribo_recon_result_tmp,
    path_to_save,
):
    scRNA_semantic_tmp = _to_numpy(scRNA_semantic_tmp)
    scRNA_encoder_tmp = _to_numpy(scRNA_encoder_tmp)
    Ribo_semantic_tmp = _to_numpy(Ribo_semantic_tmp)
    Ribo_encoder_tmp = _to_numpy(Ribo_encoder_tmp)
    cpc_scRNA_result_tmp = _to_numpy(cpc_scRNA_result_tmp)
    cpc_Ribo_result_tmp = _to_numpy(cpc_Ribo_result_tmp)
    original_scRNA_data_tmp = _to_numpy(original_scRNA_data_tmp)
    original_Ribo_data_tmp = _to_numpy(original_Ribo_data_tmp)
    Cross_scRNA_recon_result_tmp = _to_numpy(Cross_scRNA_recon_result_tmp)
    Cross_Ribo_recon_result_tmp = _to_numpy(Cross_Ribo_recon_result_tmp)

    metrics = {}
    metrics["scRNA_sp_sm_MI"] = _mi_computer(scRNA_semantic_tmp, scRNA_encoder_tmp)
    metrics["Ribo_sp_sm_MI"] = _mi_computer(Ribo_semantic_tmp, Ribo_encoder_tmp)
    metrics["RTA_rna_ribo_MI"] = _mi_computer(cpc_scRNA_result_tmp, cpc_Ribo_result_tmp)

    metrics["scRNA_Cross_avg_rmse"], metrics["scRNA_Cross_avg_mae"] = (
        metrics_rna_ribo.calculate_performance_metrics(
            original_scRNA_data_tmp, Cross_scRNA_recon_result_tmp
        )
    )

    metrics["scRNA_Cross_pearson_corr"], metrics["scRNA_Cross_spearman_corr"] = (
        metrics_rna_ribo.calculate_average_correlations(
            original_scRNA_data_tmp, Cross_scRNA_recon_result_tmp
        )
    )
    metrics["Ribo_Cross_pearson_corr"], metrics["Ribo_Cross_spearman_corr"] = (
        metrics_rna_ribo.calculate_average_correlations(
            original_Ribo_data_tmp, Cross_Ribo_recon_result_tmp
        )
    )

    _save_metrics_to_txt(metrics, save_dir=path_to_save, filename="multi_omics_metrics.txt")
    return metrics


def evaluate_metrics_scRNA_scATAC(
    scRNA_semantic_tmp,
    scRNA_encoder_tmp,
    scATAC_semantic_tmp,
    scATAC_encoder_tmp,
    emd_scRNA_data_tmp,
    emd_scATAC_data_tmp,
    cpc_scRNA_result_tmp,
    cpc_scATAC_result_tmp,
    original_scRNA_data_tmp,
    scRNA_recon_result_tmp,
    original_scATAC_data_tmp,
    scATAC_recon_result_tmp,
    Cross_scRNA_recon_result_tmp,
    Cross_scATAC_recon_result_tmp,
    path_to_save,
    true_labels=None,
):
    scRNA_semantic_tmp = _to_numpy(scRNA_semantic_tmp)
    scRNA_encoder_tmp = _to_numpy(scRNA_encoder_tmp)
    scATAC_semantic_tmp = _to_numpy(scATAC_semantic_tmp)
    scATAC_encoder_tmp = _to_numpy(scATAC_encoder_tmp)
    cpc_scRNA_result_tmp = _to_numpy(cpc_scRNA_result_tmp)
    cpc_scATAC_result_tmp = _to_numpy(cpc_scATAC_result_tmp)
    original_scRNA_data_tmp = _to_numpy(original_scRNA_data_tmp)
    original_scATAC_data_tmp = _to_numpy(original_scATAC_data_tmp)
    Cross_scRNA_recon_result_tmp = _to_numpy(Cross_scRNA_recon_result_tmp)
    Cross_scATAC_recon_result_tmp = _to_numpy(Cross_scATAC_recon_result_tmp)

    metrics = {}
    metrics["scRNA_sp_sm_MI"] = _mi_computer(scRNA_semantic_tmp, scRNA_encoder_tmp)
    metrics["scATAC_sp_sm_MI"] = _mi_computer(scATAC_semantic_tmp, scATAC_encoder_tmp)
    metrics["RTA_rna_atac_MI"] = _mi_computer(cpc_scRNA_result_tmp, cpc_scATAC_result_tmp)

    metrics["scRNA_Cross_avg_rmse"], metrics["scRNA_Cross_avg_mae"] = (
        metrics_scRNA_scatac.calculate_performance_metrics(
            original_scRNA_data_tmp, Cross_scRNA_recon_result_tmp
        )
    )

    rec_cro_a_metrics = metrics_scRNA_scatac.calculate_reconstruction_metrics(
        original_scATAC_data_tmp, Cross_scATAC_recon_result_tmp
    )
    metrics["Cross_scATAC_auc_roc"] = rec_cro_a_metrics["auc"]
    metrics["Cross_scATAC_aupr"] = rec_cro_a_metrics["aupr"]

    _save_metrics_to_txt(metrics, save_dir=path_to_save, filename="multi_omics_metrics.txt")
    return metrics


def evaluate_metrics_scRNA_scHiC(
    scRNA_semantic_tmp,
    scRNA_encoder_tmp,
    scHiC_semantic_tmp,
    scHiC_encoder_tmp,
    emd_scRNA_data_tmp,
    emd_scHiC_data_tmp,
    cpc_scRNA_result_tmp,
    cpc_scHiC_result_tmp,
    original_scRNA_data_tmp,
    scRNA_recon_result_tmp,
    original_scHiC_data_tmp,
    scHiC_recon_result_tmp,
    Cross_scRNA_recon_result_tmp,
    Cross_scHiC_recon_result_tmp,
    path_to_save,
):
    scRNA_semantic_tmp = _to_numpy(scRNA_semantic_tmp)
    scRNA_encoder_tmp = _to_numpy(scRNA_encoder_tmp)
    scHiC_semantic_tmp = _to_numpy(scHiC_semantic_tmp)
    scHiC_encoder_tmp = _to_numpy(scHiC_encoder_tmp)
    cpc_scRNA_result_tmp = _to_numpy(cpc_scRNA_result_tmp)
    cpc_scHiC_result_tmp = _to_numpy(cpc_scHiC_result_tmp)
    original_scRNA_data_tmp = _to_numpy(original_scRNA_data_tmp)
    original_scHiC_data_tmp = _to_numpy(original_scHiC_data_tmp)
    Cross_scRNA_recon_result_tmp = _to_numpy(Cross_scRNA_recon_result_tmp)
    Cross_scHiC_recon_result_tmp = _to_numpy(Cross_scHiC_recon_result_tmp)

    metrics = {}
    metrics["scRNA_sp_sm_MI"] = _mi_computer(scRNA_semantic_tmp, scRNA_encoder_tmp)
    metrics["scHiC_sp_sm_MI"] = _mi_computer(scHiC_semantic_tmp, scHiC_encoder_tmp)
    metrics["RTA_rna_hic_MI"] = _mi_computer(cpc_scRNA_result_tmp, cpc_scHiC_result_tmp)

    Cross_scRNA_recon_result_tmp = np.clip(
        np.round(Cross_scRNA_recon_result_tmp), a_min=0, a_max=None
    )
    metrics["scRNA_Cross_avg_rmse"], metrics["scRNA_Cross_avg_mae"] = (
        metrics_scRNA_schic.calculate_performance_metrics(
            original_scRNA_data_tmp, Cross_scRNA_recon_result_tmp
        )
    )

    cross_insu_p_list = _insulation_corr(
        Cross_scHiC_recon_result_tmp, original_scHiC_data_tmp, method="pearson"
    )
    cross_insu_s_list = _insulation_corr(
        Cross_scHiC_recon_result_tmp, original_scHiC_data_tmp, method="spearman"
    )
    cross_oe_list = _observed_vs_expected(
        Cross_scHiC_recon_result_tmp, original_scHiC_data_tmp
    )

    metrics["Cross_Insulation_Pearson"] = float(np.nanmean(cross_insu_p_list)) if cross_insu_p_list else 0.0
    metrics["Cross_Insulation_Spearman"] = float(np.nanmean(cross_insu_s_list)) if cross_insu_s_list else 0.0
    metrics["Cross_Observed_Expected"] = float(np.nanmean(cross_oe_list)) if cross_oe_list else 0.0

    _save_metrics_to_txt(metrics, save_dir=path_to_save, filename="multi_omics_metrics.txt")
    return metrics


__all__ = [
    "evaluate_metrics_scRNA_ADT",
    "evaluate_metrics_scRNA_scATAC",
    "evaluate_metrics_scRNA_scHiC",
    "evaluate_metrics_RNA_Ribo_seq",
]
