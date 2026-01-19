import argparse
import importlib
import os
import sys
import random
import logging
from collections import defaultdict

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


from migo.model.dataset import MultiModalDataset, split_data
from . import insulation_score as insu
from migo.config import load_config, select_section, apply_overrides
from migo.runtime import seed_everything, resolve_device


def insulation_corr(preds, targets, method="pearson"):
    scores = []
    for pred, target in zip(preds, targets):
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
    return scores


def observed_vs_expected(preds, targets):
    scores = []
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    preds_mean = preds.mean(axis=0, keepdims=True)
    targets_mean = targets.mean(axis=0, keepdims=True)
    for pred, target in zip(preds - preds_mean, targets - targets_mean):
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        if np.std(pred_flat) > 1e-8 and np.std(target_flat) > 1e-8:
            metric, _ = pearsonr(pred_flat, target_flat)
            if not np.isnan(metric) and not np.isinf(metric):
                scores.append(metric)
    return scores


def save_merged_hic_by_celltype(original_dict, cross_dict, output_dir, merge_size=30, min_cells=30):
    os.makedirs(output_dir, exist_ok=True)

    npy_data_dir = os.path.join(output_dir, "merged_hic_npy_data")
    os.makedirs(npy_data_dir, exist_ok=True)

    celltype_performance = {}
    all_merged_data = {
        "original": [],
        "cross": [],
        "cell_types": [],
        "merge_indices": [],
    }

    npy_file_info = []

    for cell_type in original_dict.keys():
        if len(original_dict[cell_type]) < min_cells:
            print(f"Skipping {cell_type}: only {len(original_dict[cell_type])} cells (< {min_cells})")
            continue

        print(f"\nProcessing {cell_type} ({len(original_dict[cell_type])} cells)...")

        original_matrices = torch.stack(original_dict[cell_type]).cpu().numpy()
        cross_matrices = torch.stack(cross_dict[cell_type]).cpu().numpy()
        num_cells = len(original_dict[cell_type])

        num_groups = num_cells // merge_size
        if num_groups == 0:
            print(f"  Warning: {cell_type} has only {num_cells} cells, cannot form groups of {merge_size}")
            continue

        print(f"  Creating {num_groups} merged groups of {merge_size} cells each")

        celltype_metrics = {
            "cross_insu_p": [],
            "cross_insu_s": [],
        }

        clean_cell_type = (
            cell_type.replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace("(", "_")
            .replace(")", "_")
        )

        for group_idx in range(num_groups):
            start_idx = group_idx * merge_size
            end_idx = start_idx + merge_size

            original_merged = np.mean(original_matrices[start_idx:end_idx], axis=0)
            cross_merged = np.mean(cross_matrices[start_idx:end_idx], axis=0)

            group_name = f"{clean_cell_type}_group{group_idx+1:02d}_merged{merge_size}cells"

            original_npy_path = os.path.join(npy_data_dir, f"original_{group_name}.npy")
            np.save(original_npy_path, original_merged)

            cross_npy_path = os.path.join(npy_data_dir, f"cross_modality_{group_name}.npy")
            np.save(cross_npy_path, cross_merged)

            npy_file_info.append({
                "cell_type": cell_type,
                "clean_cell_type": clean_cell_type,
                "group_index": group_idx + 1,
                "merge_size": merge_size,
                "start_cell_idx": start_idx,
                "end_cell_idx": end_idx - 1,
                "original_file": f"original_{group_name}.npy",
                "cross_modality_file": f"cross_modality_{group_name}.npy",
                "matrix_shape": original_merged.shape,
            })

            all_merged_data["original"].append(original_merged)
            all_merged_data["cross"].append(cross_merged)
            all_merged_data["cell_types"].append(cell_type)
            all_merged_data["merge_indices"].append(group_idx)

            cross_insu_p_list = insulation_corr([cross_merged], [original_merged], method="pearson")
            cross_insu_s_list = insulation_corr([cross_merged], [original_merged], method="spearman")
            cross_insu_p = np.mean(cross_insu_p_list) if cross_insu_p_list else np.nan
            cross_insu_s = np.mean(cross_insu_s_list) if cross_insu_s_list else np.nan

            celltype_metrics["cross_insu_p"].append(cross_insu_p)
            celltype_metrics["cross_insu_s"].append(cross_insu_s)

        celltype_performance[cell_type] = {
            "num_cells": num_cells,
            "num_groups": num_groups,
            "merge_size": merge_size,
            "avg_cross_insu_p": np.nanmean(celltype_metrics["cross_insu_p"]),
            "avg_cross_insu_s": np.nanmean(celltype_metrics["cross_insu_s"]),
        }

        print(f"  Processed {num_groups} merged groups for {cell_type}")

    npy_info_df = pd.DataFrame(npy_file_info)
    npy_info_csv_path = os.path.join(npy_data_dir, "npy_file_manifest.csv")
    npy_info_df.to_csv(npy_info_csv_path, index=False)
    print(f"\n.npy file manifest saved to: {npy_info_csv_path}")
    print(f"Total .npy files created: {len(npy_file_info) * 2}")

    readme_path = os.path.join(npy_data_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("Hi-C Merged Data (.npy format)\n")
        f.write("================================\n\n")
        f.write("File naming convention:\n")
        f.write("- original_<celltype>_group<XX>_merged<YY>cells.npy: Original Hi-C matrices\n")
        f.write("- cross_modality_<celltype>_group<XX>_merged<YY>cells.npy: Cross-modal generated Hi-C matrices (RNA->Hi-C)\n\n")
        f.write(f"Each file contains a single Hi-C matrix merged from {merge_size} cells of the same type.\n")
        f.write("Matrix dimensions: (height, width) representing contact frequency between genomic bins.\n")
        f.write("Data type: float32\n\n")
        f.write("For detailed file information, see 'npy_file_manifest.csv'\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n")

    performance_df = pd.DataFrame.from_dict(celltype_performance, orient="index")
    performance_df.reset_index(inplace=True)
    performance_df.rename(columns={"index": "Cell_Type"}, inplace=True)
    performance_csv_path = os.path.join(output_dir, "celltype_merged_performance_summary.csv")
    performance_df.to_csv(performance_csv_path, index=False)
    print(f"\nCell type merged performance summary saved to {performance_csv_path}")

    return celltype_performance, all_merged_data


def _resolve_resolution(root_cfg, overrides):
    if isinstance(overrides, dict):
        override_resolution = overrides.get("resolution")
        if override_resolution:
            return str(override_resolution)
    if isinstance(root_cfg, dict):
        cfg_resolution = root_cfg.get("resolution")
        if cfg_resolution:
            return str(cfg_resolution)
    return os.environ.get("MIGO_RESOLUTION", "1M")


def _load_scHic_layers(resolution):
    os.environ["MIGO_RESOLUTION"] = resolution
    sc_module = importlib.import_module("migo.model.layer.scRNA_scHiC")
    if getattr(sc_module, "_RESOLUTION", None) != resolution:
        sc_module = importlib.reload(sc_module)
    return sc_module.RA_VQVAE_Encoder, sc_module.RA_VQVAE_Decoder, sc_module.Cross_CPC_RA


def main(config=None, overrides=None):
    if isinstance(config, str):
        root_cfg = load_config(config)
    else:
        root_cfg = config or {}
    resolution = _resolve_resolution(root_cfg, overrides)
    os.environ["MIGO_RESOLUTION"] = resolution
    cfg = select_section(root_cfg, "infer")
    cfg = apply_overrides(cfg, overrides)
    seed_everything(cfg.get("seed", 0))
    logger = logging.getLogger(__name__)

    device_override = cfg.get("device")
    devices = resolve_device(device_override)

    rna_path = cfg.get("rna_h5ad")
    hic_path = cfg.get("hic_h5ad")
    if not rna_path or not hic_path:
        raise ValueError("infer.rna_h5ad and infer.hic_h5ad are required")

    rna = ad.read_h5ad(rna_path)
    HiC = ad.read_h5ad(hic_path)

    scRNA_datasets = torch.tensor(rna.obsm["X_pca"]).float()
    scHiC_datasets = torch.tensor(HiC.X.toarray()).float()
    ori_rna_dataset = torch.tensor(rna.layers["counts"].toarray()).float()
    ori_HiC_train_datasets = torch.tensor(HiC.X.toarray()).float()

    label = rna.obs["Celltype"]
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(label.values)
    label_tensor = torch.tensor(label_encoded)

    cell_type_names = label_encoder.classes_

    train_size1 = cfg.get("train_size1", 0.8)
    (RNA_train_data, HiC_train_data, label_train_data, RNA_validate_data, HiC_validate_data, label_validate_data,
     ori_RNA_train_data, ori_HiC_train_data, ori_RNA_test_data, ori_HiC_test_data) = split_data(
        ori_rna_dataset,
        scRNA_datasets,
        ori_HiC_train_datasets,
        scHiC_datasets,
        label_tensor,
        train_size1=train_size1,
    )

    validate_datasets = MultiModalDataset(
        RNA_validate_data,
        HiC_validate_data,
        ori_RNA_test_data,
        ori_HiC_test_data,
        label_validate_data,
    )
    batch_size = cfg.get("batch_size", 64)
    drop_last = cfg.get("val_drop_last", False)
    validate_loader = DataLoader(validate_datasets, batch_size=batch_size, shuffle=False, drop_last=drop_last)

    RA_VQVAE_Encoder, RA_VQVAE_Decoder, Cross_CPC_RA = _load_scHic_layers(resolution)

    scRNA_dim = rna.obsm["X_pca"].shape[1]
    scHiC_input_size = int(np.sqrt(HiC.X.shape[1]))
    scRNA_output_dim = rna.layers["counts"].shape[1]
    scHiC_output_dim = scHiC_input_size

    scRNA_encoder_output_dim = cfg.get("scRNA_encoder_output_dim", 128)
    scHiC_encoder_output_dim = cfg.get("scHiC_encoder_output_dim", 128)
    embedding_dim = cfg.get("embedding_dim", 128)
    projection_dim = cfg.get("projection_dim", 128)
    cpc_dim = cfg.get("cpc_dim", 64)
    n_embeddings = cfg.get("n_embeddings", 1024)

    Encoder = RA_VQVAE_Encoder(
        scRNA_dim=scRNA_dim,
        scRNA_output_dim=scRNA_encoder_output_dim,
        HiC_output_dim=scHiC_encoder_output_dim,
        embedding_dim=embedding_dim,
        n_embed=n_embeddings,
        hic_input_size=scHiC_input_size,
    )
    Decoder = RA_VQVAE_Decoder(
        scRNA_intermediate_dim=projection_dim,
        scRNA_output_dim=scRNA_output_dim,
        HiC_output_dim=scHiC_output_dim,
        latent_dim=embedding_dim,
        scRNA_specific_dim=scRNA_encoder_output_dim,
        perceptual_weight=1.0,
        hic_input_size=scHiC_input_size,
    )
    CPC = Cross_CPC_RA(cpc_dim, projection_dim, drop=0.2)

    load_state_dict_file_path = cfg.get("checkpoint_path")
    if not load_state_dict_file_path:
        raise ValueError("infer.checkpoint_path is required")

    strict_load = cfg.get("strict_load", False)
    try:
        state_dict = torch.load(load_state_dict_file_path, map_location=devices)
        Encoder.load_state_dict(state_dict["Encoder_parameters"], strict=strict_load)
        CPC.load_state_dict(state_dict["CPC_parameters"])
        Decoder.load_state_dict(state_dict["Decoder_parameters"], strict=strict_load)
        print(f"Loaded model weights from {load_state_dict_file_path}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {exc}")

    all_model = [Encoder, CPC, Decoder]
    for model in all_model:
        model.float()
        model.eval()
        model.to(devices)

    original_hic_by_celltype = defaultdict(list)
    cross_hic_by_celltype = defaultdict(list)

    path = cfg.get("output_dir")
    if not path:
        raise ValueError("infer.output_dir is required")
    os.makedirs(path, exist_ok=True)

    flag = 0
    with torch.no_grad():
        for _, batch_data in tqdm(enumerate(validate_loader), total=len(validate_loader), desc="Evaluating"):
            scRNA_ori_data, scRNA_raw_embedding, HiC_ori_data, HiC_raw_embedding, labels = batch_data

            scRNA_ori_data = scRNA_ori_data.to(devices)
            scRNA_raw_embedding = scRNA_raw_embedding.to(devices)
            HiC_ori_data = HiC_ori_data.to(devices)
            HiC_raw_embedding = HiC_raw_embedding.to(devices)

            hic_size = int(np.sqrt(HiC_ori_data.shape[1]))
            HiC_2d_data = HiC_ori_data.reshape(-1, hic_size, hic_size).to(devices)

            (scRNA_semantic_result, HiC_semantic_result, scRNA_encoder_result, HiC_encoder_result,
             scRNA_vq, HiC_semantic_vq,
             _, _, _,
             hic_skips,
             hic_quantized_bottleneck) = Encoder(scRNA_raw_embedding, HiC_2d_data, flag, CPC, devices)

            _, _, _, _, _, _, _, cross_scHiC, _ = Decoder(
                scRNA_ori_data,
                HiC_ori_data,
                scRNA_vq,
                scRNA_semantic_result,
                scRNA_encoder_result,
                hic_quantized_bottleneck,
                HiC_semantic_vq,
                hic_skips,
                hic_unet_module=Encoder.hic_unet,
            )

            for i in range(len(labels)):
                cell_type_idx = labels[i].item()
                cell_type_name = cell_type_names[cell_type_idx]

                original_hic_by_celltype[cell_type_name].append(HiC_2d_data[i].cpu())
                cross_hic_by_celltype[cell_type_name].append(cross_scHiC[i].cpu())

    print("\n--- Processing Hi-C data with merge strategy ---")

    print("\nCell type distribution:")
    for cell_type in sorted(original_hic_by_celltype.keys()):
        count = len(original_hic_by_celltype[cell_type])
        print(f"  {cell_type}: {count} cells")

    merged_output_dir = os.path.join(path, "merged_hic_celltype_analysis")

    merge_size = cfg.get("merge_size", 30)
    min_cells = cfg.get("min_cells", 30)
    celltype_performance, all_merged_data = save_merged_hic_by_celltype(
        original_hic_by_celltype,
        cross_hic_by_celltype,
        merged_output_dir,
        merge_size=merge_size,
        min_cells=min_cells,
    )

    if all_merged_data["original"]:
        all_original_merged = np.array(all_merged_data["original"])
        all_cross_merged = np.array(all_merged_data["cross"])

        overall_cross_insu_p_list = insulation_corr(all_cross_merged, all_original_merged, method="pearson")
        overall_cross_insu_s_list = insulation_corr(all_cross_merged, all_original_merged, method="spearman")
        overall_cross_insu_p = np.nanmean(overall_cross_insu_p_list)
        overall_cross_insu_s = np.nanmean(overall_cross_insu_s_list)
        overall_cross_oe_list = observed_vs_expected(all_cross_merged, all_original_merged)
        overall_cross_oe = np.nanmean(overall_cross_oe_list) if overall_cross_oe_list else np.nan

        print("\n--- scHiC Cross-Modality Metrics (Merged Data) ---")
        print(f"Cross-modality insulation correlation (Pearson): {overall_cross_insu_p:.4f}")
        print(f"Cross-modality insulation correlation (Spearman): {overall_cross_insu_s:.4f}")
        print(f"Cross-modality observed/expected correlation (Pearson): {overall_cross_oe:.4f}")
    else:
        print("No merged data available for cross-modality metrics calculation.")
        overall_cross_insu_p = 0.0
        overall_cross_insu_s = 0.0
        overall_cross_oe = 0.0

    merged_metrics_data = {
        "Metric": [
            "Cross_Insulation_Pearson_Merged",
            "Cross_Insulation_Spearman_Merged",
            "Cross_Observed_Expected_Merged",
        ],
        "Value": [
            overall_cross_insu_p,
            overall_cross_insu_s,
            overall_cross_oe,
        ],
    }

    df_merged = pd.DataFrame(merged_metrics_data)
    csv_path_merged = os.path.join(path, "merged_evaluation_metrics.csv")
    df_merged.to_csv(csv_path_merged, index=False)
    print(f"Merged metrics saved to {csv_path_merged}")

    celltype_stats = []
    for cell_type in sorted(original_hic_by_celltype.keys()):
        count = len(original_hic_by_celltype[cell_type])
        groups = count // merge_size if count >= merge_size else 0
        celltype_stats.append({
            "Cell_Type": cell_type,
            "Cell_Count": count,
            "Merged_Groups": groups,
            "Cells_Used_in_Merge": groups * merge_size,
        })

    celltype_df = pd.DataFrame(celltype_stats)
    celltype_csv_path = os.path.join(path, "celltype_distribution_merged.csv")
    celltype_df.to_csv(celltype_csv_path, index=False)
    print(f"Cell type distribution (with merge info) saved to {celltype_csv_path}")

    print("\nAnalysis complete")
    print(f"Results saved in: {path}")
    print(f"Merged Hi-C analysis saved in: {merged_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer scRNA-scHiC model")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    args = parser.parse_args()
    main(load_config(args.config))




