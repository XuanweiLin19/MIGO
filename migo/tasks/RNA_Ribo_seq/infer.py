import argparse
import os
import sys
import random

from .pretrain import pretrain_step, validate_test_epoch
from migo.model.layer.RNA_Ribo_seq import RA_VQVAE_Encoder, RA_VQVAE_Decoder
from migo.model.utils import save_models, model_to_double
from migo.model.rta import Cross_CPC_RA
from torch.utils.data import DataLoader
import torch
import anndata as ad
import logging
from tqdm import tqdm
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from migo.model.metrics import RNA_Ribo_seq as metrics
import numpy as np
from migo.utils_.evaluate_metric import evaluate_metrics_RNA_Ribo_seq as evaluate_metrics
import csv

from migo.config import load_config, select_section, get_cfg, apply_overrides
from migo.runtime import seed_everything, resolve_device, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Infer RNA Ribo-seq model")
    parser.add_argument("--config", default=None, help="Path to JSON config")
    return parser.parse_args()

def to_eval_model(all_model, device):       
    for model in all_model:
        model.eval()
        model.to(device)

class MultiModalDataset_NoLabels(torch.utils.data.Dataset):
    def __init__(self, data_mod1, data_mod2, ori_data_mod1, ori_data_mod2):
        self.data_mod1 = data_mod1
        self.data_mod2 = data_mod2
        self.ori_data_mod1 = ori_data_mod1
        self.ori_data_mod2 = ori_data_mod2

    def __len__(self):
        return len(self.data_mod1)

    def __getitem__(self, idx):
        return (self.ori_data_mod1[idx], self.data_mod1[idx], 
                self.ori_data_mod2[idx], self.data_mod2[idx])


def split_data_no_labels(ori_rna_dataset, scRNA_datasets, ori_ribo_dataset, scRibo_datasets, train_ratio=0.9):
    num_samples = len(scRNA_datasets)
    indices = torch.randperm(num_samples)
    
    train_size = int(train_ratio * num_samples)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    return (scRNA_datasets[train_idx], scRibo_datasets[train_idx],
            scRNA_datasets[val_idx], scRibo_datasets[val_idx],
            ori_rna_dataset[train_idx], ori_ribo_dataset[train_idx],
            ori_rna_dataset[val_idx], ori_ribo_dataset[val_idx], val_idx)

def main(config_path=None, overrides=None):

    cfg_root = load_config(config_path)
    cfg = select_section(cfg_root, "infer")
    cfg = apply_overrides(cfg, overrides)
    seed_everything(get_cfg(cfg, "seed", 0))
    logger = logging.getLogger(__name__)
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0

    device_id = get_cfg(cfg, "device")

    devices = resolve_device(device_id)
    print(f"Using device: {devices}")

    rna_path = get_cfg(
        cfg,
        "rna_h5ad",
        "/hum_rna_processed.h5ad",
    )
    rna = ad.read_h5ad(rna_path)
    
    ribo_path = get_cfg(
        cfg,
        "ribo_h5ad",
        "/hum_ribo_processed.h5ad",
    )
    ribo = ad.read_h5ad(ribo_path)
    
    print(f"Original RNA data shape: {rna.shape}")
    print(f"Original Ribo data shape: {ribo.shape}")
    
    scRNA_datasets = torch.tensor(rna.obsm['X_pca']).double()
    scRibo_datasets = torch.tensor(ribo.obsm['X_pca']).double()  

    if hasattr(rna.layers['counts'], 'toarray'):
        ori_rna_dataset = torch.tensor(rna.layers['counts'].toarray()).double()
    else:
        ori_rna_dataset = torch.tensor(rna.layers['counts']).double()
        
    if hasattr(ribo.layers['counts'], 'toarray'):
        ori_ribo_dataset = torch.tensor(ribo.layers['counts'].toarray()).double()
    else:
        ori_ribo_dataset = torch.tensor(ribo.layers['counts']).double()

    print(f"PCA RNA tensor shape: {scRNA_datasets.shape}")
    print(f"PCA Ribo tensor shape: {scRibo_datasets.shape}")
    print(f"Original RNA tensor shape: {ori_rna_dataset.shape}")
    print(f"Original Ribo tensor shape: {ori_ribo_dataset.shape}")

    RNA_train_data, Ribo_train_data, RNA_validate_data, Ribo_validate_data, \
        ori_RNA_train_data, ori_Ribo_train_data, ori_RNA_test_data, ori_Ribo_test_data, val_index = \
        split_data_no_labels(ori_rna_dataset, scRNA_datasets, ori_ribo_dataset, scRibo_datasets, train_ratio=0.0)
    

    validate_datasets = MultiModalDataset_NoLabels(
        RNA_validate_data, Ribo_validate_data, 
        ori_RNA_test_data, ori_Ribo_test_data
    )

    batch_size = int(get_cfg(cfg, "batch_size", 64))
    validate_loader = DataLoader(validate_datasets, batch_size=batch_size, shuffle=False, drop_last=False)

    print('The number of validate_datasets:', len(validate_datasets))
    
    scRNA_dim = scRNA_datasets.shape[1]
    scRibo_dim = scRibo_datasets.shape[1]  
    scRNA_output_dim = ori_rna_dataset.shape[1]
    scRibo_output_dim = ori_ribo_dataset.shape[1] 
    
    scRNA_encoder_output_dim = 128
    scRibo_encoder_output_dim = 128  
    cpc_dim = 64
    projection_dim = 64
    n_embeddings = 400  # Codebook size

 
    Encoder = RA_VQVAE_Encoder(scRNA_dim, scRibo_dim, scRNA_encoder_output_dim, scRibo_encoder_output_dim, projection_dim, n_embeddings)
    CPC = Cross_CPC_RA(cpc_dim, projection_dim)
    Decoder = RA_VQVAE_Decoder(projection_dim, projection_dim, scRNA_output_dim, scRibo_output_dim)


    load_state_dict_file_path = get_cfg(
        cfg,
        "checkpoint_path",
        "/scRNA_Ribo/Results_saving/pretrained-model.pt",
    )
    state_dict = torch.load(load_state_dict_file_path, map_location=devices)
    
    Encoder.load_state_dict(state_dict['Encoder_parameters'])
    CPC.load_state_dict(state_dict['CPC_parameters'])
    Decoder.load_state_dict(state_dict['Decoder_parameters'])

    all_model = [Encoder, CPC, Decoder]
    model_to_double(all_model, devices)
    eval_models = [CPC, Encoder, Decoder]
    to_eval_model(eval_models, devices)
  


    original_scRNA_data_tmp = torch.Tensor()
    original_scRibo_data_tmp = torch.Tensor()  
    emd_scRNA_data_tmp = torch.Tensor()
    emd_scRibo_data_tmp = torch.Tensor()  
    scRNA_semantic_tmp = torch.Tensor()
    scRibo_semantic_tmp = torch.Tensor()  
    scRNA_encoder_tmp = torch.Tensor()
    scRibo_encoder_tmp = torch.Tensor()  
    scRNA_vq_tmp = torch.Tensor()
    scRibo_vq_tmp = torch.Tensor()  
    scRNA_recon_result_tmp = torch.Tensor()
    scRibo_recon_result_tmp = torch.Tensor()  

    Cross_scRNA_recon_result_tmp = torch.Tensor()
    Cross_scRibo_recon_result_tmp = torch.Tensor()  

    cpc_scRNA_result_tmp = torch.Tensor()
    cpc_scRibo_result_tmp = torch.Tensor()  
    
    length_data = len(validate_loader)
    flag = 0
    epoch = 0
    validate_type = 'validate'
    validate_test_dataloader = tqdm(validate_loader)
    
    with torch.no_grad():
        for n_iter, batch_data in enumerate(validate_test_dataloader):
            scRNA_ori_data, scRNA_raw_embedding, scRibo_ori_data, scRibo_raw_embedding = batch_data
            
            scRNA_ori_data = scRNA_ori_data.to(devices)
            scRibo_ori_data = scRibo_ori_data.to(devices)  
            scRNA_raw_embedding = scRNA_raw_embedding.to(devices)
            scRibo_raw_embedding = scRibo_raw_embedding.to(devices)  

            
            scRNA_semantic_result, scRibo_semantic_result, scRNA_encoder_result, scRibo_encoder_result, \
                scRNA_vq, scRibo_vq, scRNA_embedding_loss, scRibo_embedding_loss, \
                Alignment_scRNA_Semantic, Alignment_scRibo_Semantic, batch_stats \
                = Encoder(scRNA_raw_embedding, scRibo_raw_embedding, flag, CPC, devices)

            cpc_loss, rna_embedding_norm, ribo_embedding_norm = CPC(
                scRNA_semantic_result, scRibo_semantic_result, device=devices
            )
            
            
            emd_scRNA_data_tmp = torch.cat((emd_scRNA_data_tmp, scRNA_raw_embedding.detach().cpu()), dim=0)
            emd_scRibo_data_tmp = torch.cat((emd_scRibo_data_tmp, scRibo_raw_embedding.detach().cpu()), dim=0)
            original_scRNA_data_tmp = torch.cat((original_scRNA_data_tmp, scRNA_ori_data.detach().cpu()), dim=0)
            original_scRibo_data_tmp = torch.cat((original_scRibo_data_tmp, scRibo_ori_data.detach().cpu()), dim=0)
            scRNA_semantic_tmp = torch.cat((scRNA_semantic_tmp, scRNA_semantic_result.detach().cpu()), dim=0)
            scRibo_semantic_tmp = torch.cat((scRibo_semantic_tmp, scRibo_semantic_result.detach().cpu()), dim=0)
            scRNA_encoder_tmp = torch.cat((scRNA_encoder_tmp, scRNA_encoder_result.detach().cpu()), dim=0)
            scRibo_encoder_tmp = torch.cat((scRibo_encoder_tmp, scRibo_encoder_result.detach().cpu()), dim=0)
            scRNA_vq_tmp = torch.cat((scRNA_vq_tmp, scRNA_vq.detach().cpu()), dim=0)
            scRibo_vq_tmp = torch.cat((scRibo_vq_tmp, scRibo_vq.detach().cpu()), dim=0)

            
            cpc_scRNA_result_tmp = torch.cat((cpc_scRNA_result_tmp, rna_embedding_norm.detach().cpu()), dim=0)
            cpc_scRibo_result_tmp = torch.cat((cpc_scRibo_result_tmp, ribo_embedding_norm.detach().cpu()), dim=0)

            _, _, scRNA_recon_result, scRibo_recon_result, _, _, \
            cross_scRNA, cross_scRibo = Decoder(scRNA_ori_data, scRibo_ori_data, scRNA_vq, scRibo_vq)

            scRNA_recon_result_tmp = torch.cat((scRNA_recon_result_tmp, scRNA_recon_result.detach().cpu()), dim=0)
            scRibo_recon_result_tmp = torch.cat((scRibo_recon_result_tmp, scRibo_recon_result.detach().cpu()), dim=0)
            Cross_scRNA_recon_result_tmp = torch.cat((Cross_scRNA_recon_result_tmp, cross_scRNA.detach().cpu()), dim=0)
            Cross_scRibo_recon_result_tmp = torch.cat((Cross_scRibo_recon_result_tmp, cross_scRibo.detach().cpu()), dim=0)
    
    path = get_cfg(
        cfg,
        "output_dir",
        "/scRNA_Ribo/Results_saving/All_datasets",
    )
    ensure_dir(path)

    val_indices = val_index.numpy()
    
    ori_rna = original_scRNA_data_tmp.numpy()
    ori_ribo = original_scRibo_data_tmp.numpy()
    ori_rna_data = ad.AnnData(X=ori_rna)
    ori_ribo_data = ad.AnnData(X=ori_ribo)
    
    ori_rna_data.var = rna.var.copy()
    ori_rna_data.uns = rna.uns.copy()
    if 'X_pca' in rna.obsm:
        ori_rna_data.obsm['X_pca'] = rna.obsm['X_pca'][val_indices]
    if 'PCs' in rna.varm:
        ori_rna_data.varm['PCs'] = rna.varm['PCs'].copy()
    ori_rna_data.obs = rna.obs.iloc[val_indices].copy()
    ori_rna_data.obs['modality'] = 'RNA'
    
    ori_ribo_data.var = ribo.var.copy()
    ori_ribo_data.uns = ribo.uns.copy()
    if 'X_pca' in ribo.obsm:
        ori_ribo_data.obsm['X_pca'] = ribo.obsm['X_pca'][val_indices]
    if 'PCs' in ribo.varm:
        ori_ribo_data.varm['PCs'] = ribo.varm['PCs'].copy()
    ori_ribo_data.obs = ribo.obs.iloc[val_indices].copy()
    ori_ribo_data.obs['modality'] = 'Ribo'
    
    ori_rna_data.write(os.path.join(path, 'ori_rna_data.h5ad'))
    ori_ribo_data.write(os.path.join(path, 'ori_ribo_data.h5ad'))

    self_recon_rna = scRNA_recon_result_tmp.numpy()
    self_recon_ribo = scRibo_recon_result_tmp.numpy()
    self_recon_rna_data = ad.AnnData(X=self_recon_rna)
    self_recon_ribo_data = ad.AnnData(X=self_recon_ribo)
    
    self_recon_rna_data.var = rna.var.copy()
    self_recon_rna_data.uns = rna.uns.copy()
    self_recon_rna_data.obs = rna.obs.iloc[val_indices].copy()
    self_recon_rna_data.obs['modality'] = 'RNA'
    self_recon_rna_data.obs['data_type'] = 'self_reconstruction'
    if 'PCs' in rna.varm:
        self_recon_rna_data.varm['PCs'] = rna.varm['PCs'].copy()
    
    self_recon_ribo_data.var = ribo.var.copy()
    self_recon_ribo_data.uns = ribo.uns.copy()
    self_recon_ribo_data.obs = ribo.obs.iloc[val_indices].copy()
    self_recon_ribo_data.obs['modality'] = 'Ribo'
    self_recon_ribo_data.obs['data_type'] = 'self_reconstruction'
    if 'PCs' in ribo.varm:
        self_recon_ribo_data.varm['PCs'] = ribo.varm['PCs'].copy()
    
    self_recon_rna_data.write(os.path.join(path, 'self_recon_rna_data.h5ad'))
    self_recon_ribo_data.write(os.path.join(path, 'self_recon_ribo_data.h5ad'))

    cross_recon_rna = Cross_scRNA_recon_result_tmp.numpy()
    cross_recon_ribo = Cross_scRibo_recon_result_tmp.numpy()
    cross_recon_rna_data = ad.AnnData(X=cross_recon_rna)
    cross_recon_ribo_data = ad.AnnData(X=cross_recon_ribo)
    
    cross_recon_rna_data.var = rna.var.copy()
    cross_recon_rna_data.uns = rna.uns.copy()
    cross_recon_rna_data.obs = rna.obs.iloc[val_indices].copy()
    cross_recon_rna_data.obs['modality'] = 'RNA'
    cross_recon_rna_data.obs['data_type'] = 'cross_reconstruction_from_Ribo'
    if 'PCs' in rna.varm:
        cross_recon_rna_data.varm['PCs'] = rna.varm['PCs'].copy()
    
    cross_recon_ribo_data.var = ribo.var.copy()
    cross_recon_ribo_data.uns = ribo.uns.copy()
    cross_recon_ribo_data.obs = ribo.obs.iloc[val_indices].copy()
    cross_recon_ribo_data.obs['modality'] = 'Ribo'
    cross_recon_ribo_data.obs['data_type'] = 'cross_reconstruction_from_RNA'
    if 'PCs' in ribo.varm:
        cross_recon_ribo_data.varm['PCs'] = ribo.varm['PCs'].copy()
    
    cross_recon_rna_data.write(os.path.join(path, 'cross_recon_rna_data.h5ad'))
    cross_recon_ribo_data.write(os.path.join(path, 'cross_recon_ribo_data.h5ad'))


    modality1_rna = cpc_scRNA_result_tmp.numpy()
    modality2_ribo = cpc_scRibo_result_tmp.numpy()
    rna_data = ad.AnnData(X=modality1_rna)
    ribo_data = ad.AnnData(X=modality2_ribo)
    
    rna_data.var_names = [f'cpc_dim_{i}' for i in range(modality1_rna.shape[1])]
    ribo_data.var_names = [f'cpc_dim_{i}' for i in range(modality2_ribo.shape[1])]
    rna_data.obs = rna.obs.iloc[val_indices].copy()
    ribo_data.obs = ribo.obs.iloc[val_indices].copy()
    rna_data.obs['modality'] = 'RNA'
    ribo_data.obs['modality'] = 'Ribo'
    rna_data.obs['data_type'] = 'cpc_aligned_representation'
    ribo_data.obs['data_type'] = 'cpc_aligned_representation'
    
    rna_data.write(os.path.join(path, 'rna_data.h5ad'))
    ribo_data.write(os.path.join(path, 'ribo_data.h5ad'))

    
    rna_sp_data.write(os.path.join(path, 'sp_rna_data.h5ad'))
    ribo_sp_data.write(os.path.join(path, 'sp_ribo_data.h5ad'))



    print(f"*******************************************")
    
    results = evaluate_metrics(scRNA_semantic_tmp, scRNA_encoder_tmp, 
                               scRibo_semantic_tmp, scRibo_encoder_tmp,
                               emd_scRNA_data_tmp, emd_scRibo_data_tmp,
                               cpc_scRNA_result_tmp, cpc_scRibo_result_tmp,
                               original_scRNA_data_tmp, scRNA_recon_result_tmp,
                               original_scRibo_data_tmp, scRibo_recon_result_tmp,
                               Cross_scRNA_recon_result_tmp, Cross_scRibo_recon_result_tmp, path
                               )
    print("--- Evaluation Metrics from evaluate_metrics ---")
    print(results)

    
    torch.cuda.empty_cache()
    print(f"*******************************************")



    model_dir = os.path.dirname(load_state_dict_file_path)
    csv_path = os.path.join(model_dir, 'evaluation_metrics.csv')

    formatted_results = {}
    for k, v in results.items():
        try:
            formatted_results[k] = f"{float(v):.4f}"
        except Exception:
            formatted_results[k] = str(v)

    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for k, v in formatted_results.items():
                writer.writerow([k, v])
        print(f"\nAll evaluation metrics have been successfully saved to: {csv_path}")
    except IOError as e:
        print(f"\nError: Could not save metrics to CSV file: {e}")


if __name__ == '__main__':
    args = parse_args()
    main(args.config)







