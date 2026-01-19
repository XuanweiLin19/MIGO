import argparse
import os
import sys
import random


from .pretrain import pretrain_step, validate_test_epoch
from migo.model.layer.scRNA_scATAC import RA_VQVAE_Encoder, Cross_VQ_RA, RA_VQVAE_Decoder
from migo.model.utils import save_models, model_to_double
from migo.model.rta import Cross_CPC_RA
from torch.utils.data import DataLoader
from migo.model.dataset import (
    load_datasets,
    MultiModalDataset,
    split_data,
    create_data_loaders,
    split_data_train_val_test,
)
import torch
import anndata as ad
from sklearn.preprocessing import LabelEncoder
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from migo.model.metrics import scRNA_scATAC as metrics
from migo.utils_.evaluate_metric import evaluate_metrics_scRNA_scATAC as evaluate_metrics
from migo.utils_.anndata_utils_scRNA_scATAC import build_anndata_with_var
import csv

from migo.config import load_config, select_section, get_cfg, apply_overrides
from migo.runtime import seed_everything, resolve_device, ensure_dir


def build_anndata_with_obs(matrix, feature_template, obs_template, modality, labels, split_labels=None):
    adata = build_anndata_with_var(matrix, feature_template)
    obs_df = obs_template.copy(deep=True)
    obs_df['modality'] = modality
    obs_df['label'] = labels
    if split_labels is not None:
        obs_df['split'] = split_labels
    adata.obs = obs_df
    return adata


def parse_args():
    parser = argparse.ArgumentParser(description="Infer scRNA scATAC model")
    parser.add_argument("--config", default=None, help="Path to JSON config")
    return parser.parse_args()

def to_eval_model(all_model, device):
    for model in all_model:
        model.eval()
        model.to(device)

def main(config_path=None, overrides=None):

    cfg_root = load_config(config_path)
    cfg = select_section(cfg_root, "infer")
    cfg = apply_overrides(cfg, overrides)
    seed_everything(get_cfg(cfg, "seed", 0))
    # global args, logger, writer
    logger = logging.getLogger(__name__)
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0

    device_id = get_cfg(cfg, "device")

    devices = resolve_device(device_id)
    print(f"Using device: {devices}")

    rna_path = get_cfg(
        cfg,
        "rna_h5ad",
        "/Datasets/normalize_True_log1p_True_hvg_True_3000_RNA_processed_data.h5ad",
    )
    atac_path = get_cfg(
        cfg,
        "atac_h5ad",
        "/Datasets/binarize_True_filter_True_fpeaks_0.02_tfidf_True_normalize_True_ATAC_processed_data.h5ad",
    )
    rna = ad.read_h5ad(rna_path)
    atac = ad.read_h5ad(atac_path)

    scRNA_datasets = torch.tensor(rna.obsm['X_pca']).double()
    scATAC_datasets = torch.tensor(atac.obsm['X_lsi']).double()

    ori_rna_dataset = torch.tensor(rna.layers['counts']).double()
    ori_atac_train_datasets = torch.tensor(atac.layers['counts']).double()

    label = rna.obs['cell_type']
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(label.values)
    label_tensor = torch.tensor(label_encoded)

    #Split Train validation test

    split_outputs = split_data(
        ori_rna_dataset,
        scRNA_datasets,
        ori_atac_train_datasets,
        scATAC_datasets,
        label_tensor,
        train_size1=0.9,
        return_indices=True
    )

    (RNA_train_data, ATAC_train_data, label_train_data, RNA_validate_data, ATAC_validate_data, label_validate_data,
     ori_RNA_train_data, ori_ATAC_train_data, ori_RNA_test_data, ori_ATAC_test_data,
     train_indices, validate_indices) = split_outputs

    # breakpoint()
    validate_datasets = MultiModalDataset(RNA_validate_data, ATAC_validate_data, ori_RNA_test_data,
                                          ori_ATAC_test_data, label_validate_data)

    batch_size = int(get_cfg(cfg, "batch_size", 64))
    train_datasets = MultiModalDataset(RNA_train_data, ATAC_train_data, ori_RNA_train_data,
                                          ori_ATAC_train_data, label_train_data)
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=False, drop_last=False)
    validate_loader = DataLoader(validate_datasets, batch_size=batch_size, shuffle=False, drop_last=False)

    train_indices = np.asarray(train_indices)
    validate_indices = np.asarray(validate_indices)
    rna_obs_train = rna.obs.iloc[train_indices].copy().reset_index(drop=True)
    atac_obs_train = atac.obs.iloc[train_indices].copy().reset_index(drop=True)
    rna_obs_subset = rna.obs.iloc[validate_indices].copy().reset_index(drop=True)
    atac_obs_subset = atac.obs.iloc[validate_indices].copy().reset_index(drop=True)
    rna_obs_cross = pd.concat([rna_obs_train, rna_obs_subset], ignore_index=True)
    atac_obs_cross = pd.concat([atac_obs_train, atac_obs_subset], ignore_index=True)

    print('The number of validate_datasets:', len(validate_datasets))

    scRNA_dim = 256
    scATAC_dim = 256
    scRNA_output_dim = rna.layers['counts'].shape[1]
    scATAC_output_dim = atac.layers['counts'].shape[1]
    # breakpoint()
    scRNA_encoder_output_dim = 128
    scATAC_encoder_output_dim = 128
    cpc_dim = 64
    projection_dim = 64
    n_embeddings = 400 #Codebook size

    Encoder = RA_VQVAE_Encoder(scRNA_dim, scATAC_dim, scRNA_encoder_output_dim, scATAC_encoder_output_dim, projection_dim, n_embeddings)
    #(256, 256, 128, 128, 128, 400)
    CPC = Cross_CPC_RA(cpc_dim, projection_dim)
    # print('CLUB network:', scATAC_mi_net)
    # breakpoint()
    rna_distribution = "zin" 
    Decoder = RA_VQVAE_Decoder(
        projection_dim,
        projection_dim,
        scRNA_output_dim,
        scATAC_output_dim,
        rna_distribution=rna_distribution
    )

    load_state_dict_file_path = get_cfg(
        cfg,
        "checkpoint_path",
        "/pretrained-model.pt",
    )
    state_dict = torch.load(load_state_dict_file_path, map_location=devices)
    # breakpoint()
    Encoder.load_state_dict(state_dict['Encoder_parameters'])
    CPC.load_state_dict(state_dict['CPC_parameters'])
    Decoder.load_state_dict(state_dict['Decoder_parameters'])

    all_model = [Encoder, CPC, Decoder]
    model_to_double(all_model, devices)
    eval_models = [CPC, Encoder, Decoder]
    to_eval_model(eval_models, devices)

    original_scRNA_data_tmp = torch.Tensor()
    original_scATAC_data_tmp = torch.Tensor()
    emd_scRNA_data_tmp = torch.Tensor()
    emd_scATAC_data_tmp = torch.Tensor()

    scRNA_encoder_tmp = torch.Tensor()
    scATAC_encoder_tmp = torch.Tensor()

    scRNA_recon_result_tmp = torch.Tensor()
    scATAC_recon_result_tmp = torch.Tensor()

    Cross_scRNA_recon_result_tmp = torch.Tensor()
    Cross_scATAC_recon_result_tmp = torch.Tensor()

    Cross_scRNA_recon_train_tmp = torch.Tensor()
    Cross_scATAC_recon_train_tmp = torch.Tensor()

    cpc_scRNA_result_tmp = torch.Tensor()
    cpc_scATAC_result_tmp = torch.Tensor()
    label_tmp = torch.Tensor()
    length_data = len(validate_loader)
    flag = 0
    epoch = 0
    validate_type = 'validate'
    validate_test_dataloader = tqdm(validate_loader)
      
    with torch.no_grad():

        for n_iter, batch_data in enumerate(validate_test_dataloader):

            scRNA_ori_data, scRNA_raw_embedding, scATAC_ori_data, scATAC_raw_embedding, labels = batch_data

            scRNA_ori_data = scRNA_ori_data.to(devices)
            scATAC_ori_data = scATAC_ori_data.to(devices)
            scRNA_raw_embedding = scRNA_raw_embedding.to(devices)
            scATAC_raw_embedding = scATAC_raw_embedding.to(devices)

            scRNA_semantic_result, scATAC_semantic_result, scRNA_encoder_result, scATAC_encoder_result, \
                scRNA_vq, scATAC_vq, scRNA_embedding_loss, scATAC_embedding_loss, \
                Alignment_scRNA_Semantic, Alignment_scATAC_Semantic, batch_stats \
                = Encoder(scRNA_raw_embedding, scATAC_raw_embedding, flag, CPC, devices)
            
            emd_scRNA_data_tmp = torch.cat((emd_scRNA_data_tmp, scRNA_raw_embedding.detach().cpu()), dim=0)
            emd_scATAC_data_tmp = torch.cat((emd_scATAC_data_tmp, scATAC_raw_embedding.detach().cpu()), dim=0)
            original_scRNA_data_tmp = torch.cat((original_scRNA_data_tmp, scRNA_ori_data.detach().cpu()), dim=0)
            original_scATAC_data_tmp = torch.cat((original_scATAC_data_tmp, scATAC_ori_data.detach().cpu()), dim=0)

            scRNA_encoder_tmp = torch.cat((scRNA_encoder_tmp, scRNA_encoder_result.detach().cpu()), dim=0)
            scATAC_encoder_tmp = torch.cat((scATAC_encoder_tmp, scATAC_encoder_result.detach().cpu()), dim=0)

            label_tmp = torch.cat((label_tmp, labels.detach().cpu()), dim=0)

            _, rna_embedding_norm, atac_embedding_norm = CPC(
                scRNA_semantic_result, scATAC_semantic_result, device=devices
            )
            cpc_scRNA_result_tmp = torch.cat((cpc_scRNA_result_tmp, rna_embedding_norm.detach().cpu()), dim=0)
            cpc_scATAC_result_tmp = torch.cat((cpc_scATAC_result_tmp, atac_embedding_norm.detach().cpu()), dim=0)

            _, _, scRNA_recon_result, scATAC_recon_result, _, _, \
            cross_scRNA, cross_scATAC = Decoder(scRNA_ori_data, scATAC_ori_data, scRNA_vq, scATAC_vq)

            scRNA_recon_result_tmp = torch.cat((scRNA_recon_result_tmp, scRNA_recon_result.detach().cpu()), dim=0)
            scATAC_recon_result_tmp = torch.cat((scATAC_recon_result_tmp, scATAC_recon_result.detach().cpu()), dim=0)
            Cross_scRNA_recon_result_tmp = torch.cat((Cross_scRNA_recon_result_tmp, cross_scRNA.detach().cpu()), dim=0)
            Cross_scATAC_recon_result_tmp = torch.cat((Cross_scATAC_recon_result_tmp, cross_scATAC.detach().cpu()), dim=0)

    with torch.no_grad():
        for n_iter, batch_data in enumerate(tqdm(train_loader, desc="Evaluating Train Cross")):
            scRNA_ori_data, scRNA_raw_embedding, scATAC_ori_data, scATAC_raw_embedding, labels = batch_data
            scRNA_ori_data = scRNA_ori_data.to(devices)
            scATAC_ori_data = scATAC_ori_data.to(devices)
            scRNA_raw_embedding = scRNA_raw_embedding.to(devices)
            scATAC_raw_embedding = scATAC_raw_embedding.to(devices)

            scRNA_semantic_result, scATAC_semantic_result, scRNA_encoder_result, scATAC_encoder_result, \
                scRNA_vq, scATAC_vq, _, _, _, _, _ = Encoder(scRNA_raw_embedding, scATAC_raw_embedding, flag, CPC, devices)

            _, _, _, _, _, _, cross_scRNA, cross_scATAC = Decoder(scRNA_ori_data, scATAC_ori_data, scRNA_vq, scATAC_vq)

            Cross_scRNA_recon_train_tmp = torch.cat((Cross_scRNA_recon_train_tmp, cross_scRNA.detach().cpu()), dim=0)
            Cross_scATAC_recon_train_tmp = torch.cat((Cross_scATAC_recon_train_tmp, cross_scATAC.detach().cpu()), dim=0)
    path = get_cfg(
        cfg,
        "output_dir",
        "/compute_test_datasets/10X_PBMC",
    )
    ensure_dir(path)

    label_tmp_numpy = label_tmp.numpy()
    ori_label = label_encoder.inverse_transform(label_tmp_numpy.astype(int))

    # Save original data
    ori_rna = original_scRNA_data_tmp.numpy()
    ori_atac = original_scATAC_data_tmp.numpy()
    ori_rna_data = build_anndata_with_obs(ori_rna, rna, rna_obs_subset, 'RNA', ori_label)
    ori_atac_data = build_anndata_with_obs(ori_atac, atac, atac_obs_subset, 'ATAC', ori_label)
    ori_rna_data.write(os.path.join(path, 'ori_rna_data.h5ad'))
    ori_atac_data.write(os.path.join(path, 'ori_atac_data.h5ad'))

    # Save self-reconstruction data
    self_recon_rna = scRNA_recon_result_tmp.numpy()
    self_recon_atac = scATAC_recon_result_tmp.numpy()
    self_recon_rna_data = build_anndata_with_obs(self_recon_rna, rna, rna_obs_subset, 'RNA', ori_label)
    self_recon_atac_data = build_anndata_with_obs(self_recon_atac, atac, atac_obs_subset, 'ATAC', ori_label)
    self_recon_rna_data.write(os.path.join(path, 'self_recon_rna_data.h5ad'))
    self_recon_atac_data.write(os.path.join(path, 'self_recon_atac_data.h5ad'))

    # Save cross-modal generation data (train + test)
    cross_recon_rna_test = Cross_scRNA_recon_result_tmp.numpy()
    cross_recon_atac_test = Cross_scATAC_recon_result_tmp.numpy()
    cross_recon_rna_train = Cross_scRNA_recon_train_tmp.numpy()
    cross_recon_atac_train = Cross_scATAC_recon_train_tmp.numpy()
    cross_recon_rna = np.concatenate([cross_recon_rna_train, cross_recon_rna_test], axis=0)
    cross_recon_atac = np.concatenate([cross_recon_atac_train, cross_recon_atac_test], axis=0)

    train_labels = label_encoder.inverse_transform(label_train_data.numpy().astype(int))
    test_labels = label_encoder.inverse_transform(label_validate_data.numpy().astype(int))
    split_labels = np.array(['train'] * len(train_labels) + ['test'] * len(test_labels))
    cross_labels = np.concatenate([train_labels, test_labels])

    cross_recon_rna_data = build_anndata_with_obs(cross_recon_rna, rna, rna_obs_cross, 'RNA', cross_labels, split_labels=split_labels)
    cross_recon_atac_data = build_anndata_with_obs(cross_recon_atac, atac, atac_obs_cross, 'ATAC', cross_labels, split_labels=split_labels)
    cross_recon_rna_data.write(os.path.join(path, 'cross_recon_rna_data.h5ad'))
    cross_recon_atac_data.write(os.path.join(path, 'cross_recon_atac_data.h5ad'))
    

    modality1_rna = cpc_scRNA_result_tmp.numpy()
    modality2_atac = cpc_scATAC_result_tmp.numpy()
    rna_data = build_anndata_with_obs(modality1_rna, rna, rna_obs_subset, 'RNA', ori_label)
    atac_data = build_anndata_with_obs(modality2_atac, atac, atac_obs_subset, 'ATAC', ori_label)
    label_tmp = label_tmp.numpy()
    ori_label = label_encoder.inverse_transform(label_tmp.astype(int))


    rna_data.write(os.path.join(path, 'rna_data.h5ad'))
    atac_data.write(os.path.join(path, 'atac_data.h5ad'))


    scRNA_sp = scRNA_encoder_tmp.numpy()
    scATAC_sp = scATAC_encoder_tmp.numpy()
    rna_sp_data = build_anndata_with_obs(scRNA_sp, rna, rna_obs_subset, 'RNA', ori_label)
    atac_sp_data = build_anndata_with_obs(scATAC_sp, atac, atac_obs_subset, 'ATAC', ori_label)
    rna_sp_data.write(os.path.join(path, 'sp_rna_data.h5ad'))
    atac_sp_data.write(os.path.join(path, 'sp_atac_data.h5ad'))


    print(f"*******************************************")
    Results_12_6 = evaluate_metrics(scRNA_semantic_tmp, scRNA_encoder_tmp, 
                               scATAC_semantic_tmp, scATAC_encoder_tmp,
                               emd_scRNA_data_tmp, emd_scATAC_data_tmp,
                               cpc_scRNA_result_tmp, cpc_scATAC_result_tmp,
                               original_scRNA_data_tmp, scRNA_recon_result_tmp,
                               original_scATAC_data_tmp, scATAC_recon_result_tmp,
                               Cross_scRNA_recon_result_tmp, Cross_scATAC_recon_result_tmp, path,
                               label_tmp
                               )
    print("--- Evaluation Metrics from evaluate_metrics ---")
    
    torch.cuda.empty_cache()
    csv_path = os.path.join(path, 'evaluation_metrics.csv')

    formatted_Results_12_6 = {}
    for k, v in Results_12_6.items():
        try:
            formatted_Results_12_6[k] = f"{float(v):.4f}"
        except Exception:
            formatted_Results_12_6[k] = str(v)

    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for k, v in formatted_Results_12_6.items():
                writer.writerow([k, v])
        print(f"\nAll evaluation metrics have been successfully saved to: {csv_path}")
    except IOError as e:
        print(f"\nError: Could not save metrics to CSV file: {e}")


if __name__ == '__main__':
    args = parse_args()
    main(args.config)

















