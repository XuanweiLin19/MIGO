import argparse
import os
import sys
import random

from .pretrain import pretrain_step, validate_test_epoch
from migo.model.layer.scRNA_ADT import RA_VQVAE_Encoder, RA_VQVAE_Decoder
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
import scanpy as sc
import matplotlib.pyplot as plt
from migo.model.metrics import scRNA_ADT as metrics
import numpy as np
from migo.utils_.evaluate_metric import evaluate_metrics_scRNA_ADT as evaluate_metrics
import csv

from migo.config import load_config, select_section, get_cfg, apply_overrides
from migo.runtime import seed_everything, resolve_device, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Infer scRNA ADT model")
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
        "/Datasets/RNA_ADT/normalize_True_log1p_True_hvg_True_3000_RNA_processed_data.h5ad",
    )
    adt_path = get_cfg(
        cfg,
        "adt_h5ad",
        "/Datasets/RNA_ADT/processed_ADT.h5ad",
    )
    rna = ad.read_h5ad(rna_path)
    ADT = ad.read_h5ad(adt_path)

    original_rna_var = rna.var.copy()
    original_adt_var = ADT.var.copy()
    original_rna_obs = rna.obs.copy()
    original_adt_obs = ADT.obs.copy()

    scRNA_datasets = torch.tensor(rna.obsm['X_pca']).double()
    scADT_datasets = torch.tensor(ADT.obsm['X_pca']).double()

    ori_rna_dataset = torch.tensor(rna.layers['counts']).double()
    ori_ADT_train_datasets = torch.tensor(ADT.layers['counts_raw']).double()

    label = rna.obs['cell_type']
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(label.values)
    label_tensor = torch.tensor(label_encoded)



    RNA_train_data, ADT_train_data, label_train_data, RNA_validate_data, ADT_validate_data, label_validate_data,\
        ori_RNA_train_data, ori_ADT_train_data, ori_RNA_test_data, ori_ADT_test_data\
        = split_data(ori_rna_dataset, scRNA_datasets, ori_ADT_train_datasets, scADT_datasets, label_tensor)

    validate_datasets = MultiModalDataset(RNA_validate_data, ADT_validate_data, ori_RNA_test_data,
                                          ori_ADT_test_data, label_validate_data)

    batch_size = int(get_cfg(cfg, "batch_size", 64))
    validate_loader = DataLoader(validate_datasets, batch_size=batch_size, shuffle=False, drop_last=False)


    print('The number of validate_datasets:', len(validate_datasets))

    scRNA_dim = scRNA_datasets.shape[1]
    scADT_dim = scADT_datasets.shape[1]
    scRNA_output_dim = rna.layers['counts'].shape[1]
    scADT_output_dim = ADT.layers['counts_raw'].shape[1]

    scRNA_encoder_output_dim = 128
    scADT_encoder_output_dim = 128
    cpc_dim = 64
    projection_dim = 64
    n_embeddings = 1024 #Codebook size

    Encoder = RA_VQVAE_Encoder(scRNA_dim, scADT_dim, scRNA_encoder_output_dim, scADT_encoder_output_dim, projection_dim, n_embeddings)

    CPC = Cross_CPC_RA(cpc_dim, projection_dim)

    Decoder = RA_VQVAE_Decoder(projection_dim, projection_dim, scRNA_output_dim, scADT_output_dim)

    load_state_dict_file_path = get_cfg(
        cfg,
        "checkpoint_path",
        "/Results/pretrained-model.pt",
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
    original_scADT_data_tmp = torch.Tensor()
    emd_scRNA_data_tmp = torch.Tensor()
    emd_scADT_data_tmp = torch.Tensor()

    scRNA_encoder_tmp = torch.Tensor()
    scADT_encoder_tmp = torch.Tensor()

    scRNA_recon_result_tmp = torch.Tensor()
    scADT_recon_result_tmp = torch.Tensor()

    Cross_scRNA_recon_result_tmp = torch.Tensor()
    Cross_scADT_recon_result_tmp = torch.Tensor()

    cpc_scRNA_result_tmp = torch.Tensor()
    cpc_scADT_result_tmp = torch.Tensor()
    label_tmp = torch.Tensor()
    length_data = len(validate_loader)
    flag = 0
    epoch = 0
    validate_type = 'validate'
    validate_test_dataloader = tqdm(validate_loader)
    
    with torch.no_grad():

        for n_iter, batch_data in enumerate(validate_test_dataloader):

            scRNA_ori_data, scRNA_raw_embedding, scADT_ori_data, scADT_raw_embedding, labels = batch_data

            scRNA_ori_data = scRNA_ori_data.to(devices)
            scADT_ori_data = scADT_ori_data.to(devices)
            scRNA_raw_embedding = scRNA_raw_embedding.to(devices)
            scADT_raw_embedding = scADT_raw_embedding.to(devices)

            scRNA_semantic_result, scADT_semantic_result, scRNA_encoder_result, scADT_encoder_result, \
                scRNA_vq, scADT_vq, scRNA_embedding_loss, scADT_embedding_loss, \
                Alignment_scRNA_Semantic, Alignment_scADT_Semantic, batch_stats \
                = Encoder(scRNA_raw_embedding, scADT_raw_embedding, flag, CPC, devices)
            
            emd_scRNA_data_tmp = torch.cat((emd_scRNA_data_tmp, scRNA_raw_embedding.detach().cpu()), dim=0)
            emd_scADT_data_tmp = torch.cat((emd_scADT_data_tmp, scADT_raw_embedding.detach().cpu()), dim=0)
            original_scRNA_data_tmp = torch.cat((original_scRNA_data_tmp, scRNA_ori_data.detach().cpu()), dim=0)
            original_scADT_data_tmp = torch.cat((original_scADT_data_tmp, scADT_ori_data.detach().cpu()), dim=0)

            scRNA_encoder_tmp = torch.cat((scRNA_encoder_tmp, scRNA_encoder_result.detach().cpu()), dim=0)
            scADT_encoder_tmp = torch.cat((scADT_encoder_tmp, scADT_encoder_result.detach().cpu()), dim=0)

            label_tmp = torch.cat((label_tmp, labels.detach().cpu()), dim=0)

            _, rna_embedding_norm, ADT_embedding_norm = CPC(
                scRNA_semantic_result, scADT_semantic_result, device=devices
            )
            cpc_scRNA_result_tmp = torch.cat((cpc_scRNA_result_tmp, rna_embedding_norm.detach().cpu()), dim=0)
            cpc_scADT_result_tmp = torch.cat((cpc_scADT_result_tmp, ADT_embedding_norm.detach().cpu()), dim=0)

            _, _, scRNA_recon_result, scADT_recon_result, _, _, \
            cross_scRNA, cross_scADT = Decoder(scRNA_ori_data, scADT_ori_data, scRNA_vq, scADT_vq)

            scRNA_recon_result_tmp = torch.cat((scRNA_recon_result_tmp, scRNA_recon_result.detach().cpu()), dim=0)
            scADT_recon_result_tmp = torch.cat((scADT_recon_result_tmp, scADT_recon_result.detach().cpu()), dim=0)
            Cross_scRNA_recon_result_tmp = torch.cat((Cross_scRNA_recon_result_tmp, cross_scRNA.detach().cpu()), dim=0)
            Cross_scADT_recon_result_tmp = torch.cat((Cross_scADT_recon_result_tmp, cross_scADT.detach().cpu()), dim=0)

    path = get_cfg(
        cfg,
        "output_dir",
        "/MIGO_scRNA_ADT_Test/Results",
    )
    ensure_dir(path)
    os.makedirs(path, exist_ok=True)

    label_tmp_numpy = label_tmp.numpy()
    ori_label = label_encoder.inverse_transform(label_tmp_numpy.astype(int))

    # Save original data with proper var information
    ori_rna = original_scRNA_data_tmp.numpy()
    ori_ADT = original_scADT_data_tmp.numpy()
    
    ori_rna_data = ad.AnnData(X=ori_rna)
    ori_rna_data.var = original_rna_var.copy()
    ori_rna_data.obs['modality'] = 'RNA'
    ori_rna_data.obs['label'] = ori_label
    for col in original_rna_obs.columns:
        if col not in ori_rna_data.obs.columns:
            if len(original_rna_obs[col].iloc[:len(ori_label)]) == len(ori_label):
                ori_rna_data.obs[col] = original_rna_obs[col].iloc[:len(ori_label)].values
    
    ori_ADT_data = ad.AnnData(X=ori_ADT)
    ori_ADT_data.var = original_adt_var.copy()
    ori_ADT_data.obs['modality'] = 'ADT'
    ori_ADT_data.obs['label'] = ori_label
    for col in original_adt_obs.columns:
        if col not in ori_ADT_data.obs.columns:
            if len(original_adt_obs[col].iloc[:len(ori_label)]) == len(ori_label):
                ori_ADT_data.obs[col] = original_adt_obs[col].iloc[:len(ori_label)].values
    
    ori_rna_data.write(os.path.join(path, 'ori_rna_data.h5ad'))
    ori_ADT_data.write(os.path.join(path, 'ori_ADT_data.h5ad'))

    # Save self-reconstruction data with proper var information
    self_recon_rna = scRNA_recon_result_tmp.numpy()
    self_recon_ADT = scADT_recon_result_tmp.numpy()
    
    self_recon_rna_data = ad.AnnData(X=self_recon_rna)
    self_recon_rna_data.var = original_rna_var.copy()
    self_recon_rna_data.obs['modality'] = 'RNA'
    self_recon_rna_data.obs['label'] = ori_label
    
    self_recon_ADT_data = ad.AnnData(X=self_recon_ADT)
    self_recon_ADT_data.var = original_adt_var.copy()
    self_recon_ADT_data.obs['modality'] = 'ADT'
    self_recon_ADT_data.obs['label'] = ori_label
    
    self_recon_rna_data.write(os.path.join(path, 'self_recon_rna_data.h5ad'))
    self_recon_ADT_data.write(os.path.join(path, 'self_recon_ADT_data.h5ad'))

    # Save cross-modal generation data with proper var information
    cross_recon_rna = Cross_scRNA_recon_result_tmp.numpy()
    cross_recon_ADT = Cross_scADT_recon_result_tmp.numpy()
    
    cross_recon_rna_data = ad.AnnData(X=cross_recon_rna)
    cross_recon_rna_data.var = original_rna_var.copy()
    cross_recon_rna_data.obs['modality'] = 'RNA'
    cross_recon_rna_data.obs['label'] = ori_label
    
    cross_recon_ADT_data = ad.AnnData(X=cross_recon_ADT)
    cross_recon_ADT_data.var = original_adt_var.copy()
    cross_recon_ADT_data.obs['modality'] = 'ADT'
    cross_recon_ADT_data.obs['label'] = ori_label
    
    cross_recon_rna_data.write(os.path.join(path, 'cross_recon_rna_data.h5ad'))
    cross_recon_ADT_data.write(os.path.join(path, 'cross_recon_ADT_data.h5ad'))
    

    modality1_rna = cpc_scRNA_result_tmp.numpy()
    modality2_ADT = cpc_scADT_result_tmp.numpy()
    
    rna_data = ad.AnnData(X=modality1_rna)
    rna_data.obs['modality'] = 'RNA'
    rna_data.obs['label'] = ori_label
    rna_data.var_names = [f'cpc_rna_dim_{i}' for i in range(modality1_rna.shape[1])]
    
    ADT_data = ad.AnnData(X=modality2_ADT)
    ADT_data.obs['modality'] = 'ADT'
    ADT_data.obs['label'] = ori_label
    ADT_data.var_names = [f'cpc_adt_dim_{i}' for i in range(modality2_ADT.shape[1])]
    
    rna_data.write(os.path.join(path, 'rna_data.h5ad'))
    ADT_data.write(os.path.join(path, 'ADT_data.h5ad'))


    print(f"\nSaved ADT data with {len(original_adt_var)} proteins:")
    print(f"Protein names: {list(original_adt_var.index[:10])}..." if len(original_adt_var) > 10 else f"Protein names: {list(original_adt_var.index)}")
    print(f"ADT var columns: {list(original_adt_var.columns)}")

    print(f"*******************************************")
    results = evaluate_metrics(scRNA_semantic_tmp, scRNA_encoder_tmp, 
                               scADT_semantic_tmp, scADT_encoder_tmp,
                               emd_scRNA_data_tmp, emd_scADT_data_tmp,
                               cpc_scRNA_result_tmp, cpc_scADT_result_tmp,
                               original_scRNA_data_tmp, scRNA_recon_result_tmp,
                               original_scADT_data_tmp, scADT_recon_result_tmp,
                               Cross_scRNA_recon_result_tmp, Cross_scADT_recon_result_tmp, path
                               )
    print("--- Generation Quality Evaluation Metrics ---")
    print(results)
    
    


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
        print(f"\nGeneration quality metrics have been successfully saved to: {csv_path}")
    except IOError as e:
        print(f"\nError: Could not save metrics to CSV file: {e}")

if __name__ == '__main__':
    args = parse_args()
    main(args.config)







