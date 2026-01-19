import argparse
import os
import sys
import random


from .pretrain import pretrain_step, validate_test_epoch
from migo.model.layer.RNA_Ribo_seq import RA_VQVAE_Encoder, RA_VQVAE_Decoder
from migo.model.CLUB import CLUBSample_group
import anndata as ad
from migo.model.utils import save_models, model_to_double
from migo.model.rta import Cross_CPC_RA
from itertools import chain
from torch.utils.data import DataLoader
from migo.model.utils import AverageMeter, Prepare_logger, save_models
import torch
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from migo.utils_.data_processed import RNA_Ribo_unified_preprocessing

from migo.config import load_config, select_section, get_cfg, apply_overrides
from migo.runtime import seed_everything, resolve_device, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train RNA Ribo-seq model")
    parser.add_argument("--config", default=None, help="Path to JSON config")
    return parser.parse_args()

def split_data_no_labels(ori_rna_dataset, scRNA_datasets, ori_ribo_dataset, scRibo_datasets, train_ratio=0.9):
    num_samples = len(scRNA_datasets)
    indices = torch.randperm(num_samples)
    
    train_size = int(train_ratio * num_samples)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    return (scRNA_datasets[train_idx], scRibo_datasets[train_idx],
            scRNA_datasets[val_idx], scRibo_datasets[val_idx],
            ori_rna_dataset[train_idx], ori_ribo_dataset[train_idx],
            ori_rna_dataset[val_idx], ori_ribo_dataset[val_idx])

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

def main(config_path=None, overrides=None):
    torch.cuda.empty_cache()
    cfg_root = load_config(config_path)
    cfg = select_section(cfg_root, "train")
    cfg = apply_overrides(cfg, overrides)
    seed_everything(get_cfg(cfg, "seed", 0))

    device_id = get_cfg(cfg, "device")

    logger = logging.getLogger(__name__)
    global best_accuracy, best_accuracy_epoch, validate_loss
    best_accuracy, best_accuracy_epoch = 0, 0

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
        ori_RNA_train_data, ori_Ribo_train_data, ori_RNA_test_data, ori_Ribo_test_data = \
        split_data_no_labels(ori_rna_dataset, scRNA_datasets, ori_ribo_dataset, scRibo_datasets)

    train_datasets = MultiModalDataset_NoLabels(RNA_train_data, Ribo_train_data, 
                                               ori_RNA_train_data, ori_Ribo_train_data)
    validate_datasets = MultiModalDataset_NoLabels(RNA_validate_data, Ribo_validate_data, 
                                                  ori_RNA_test_data, ori_Ribo_test_data)

    batch_size = int(get_cfg(cfg, "batch_size", 64))
    num_workers = int(get_cfg(cfg, "num_workers", 20))
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    validate_loader = DataLoader(validate_datasets, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    print('The number of train_datasets:', len(train_datasets))
    print('The number of validate_datasets:', len(validate_datasets))

    scRNA_dim = scRNA_datasets.shape[1]
    scRibo_dim = scRibo_datasets.shape[1]
    scRNA_output_dim = ori_rna_dataset.shape[1]
    scRibo_output_dim = ori_ribo_dataset.shape[1]
    scRNA_encoder_output_dim = int(get_cfg(cfg, "scRNA_encoder_output_dim", 128))
    scRibo_encoder_output_dim = int(get_cfg(cfg, "scRibo_encoder_output_dim", 128))
    cpc_dim = int(get_cfg(cfg, "cpc_dim", 64))
    projection_dim = int(get_cfg(cfg, "projection_dim", 64))

    n_embeddings = int(get_cfg(cfg, "n_embeddings", 400)) #Codebook size
    embedding_dim = 128
    start_epoch = -1
    total_step = 0
    n_epoch = int(get_cfg(cfg, "n_epoch", 500))
    eval_freq = int(get_cfg(cfg, "eval_freq", 1))
    model_save_path = get_cfg(
        cfg,
        "model_save_path",
        "/scRNA_Ribo/Results_saving",
    )
    scRNA_mi_lr = float(get_cfg(cfg, "scRNA_mi_lr", 0.1))
    scRibo_mi_lr = float(get_cfg(cfg, "scRibo_mi_lr", 0.1))
    cpc_lr = 0.01
    lr = float(get_cfg(cfg, "lr", 0.0004))
    best_focs = 1
    best_loss = 99999999999
    
    ensure_dir(model_save_path)
    
    log_dir = get_cfg(cfg, "log_dir", os.path.join(model_save_path, "logs"))
    ensure_dir(log_dir)
    writer = SummaryWriter(log_dir)

    loss_history = {
        'train_total_loss': [],
        'val_total_loss': [],
        'epochs': [],
    }

    Encoder = RA_VQVAE_Encoder(scRNA_dim, scRibo_dim, scRNA_encoder_output_dim, scRibo_encoder_output_dim, projection_dim, n_embeddings)

    CPC = Cross_CPC_RA(cpc_dim, projection_dim)

    scRNA_mi_net = CLUBSample_group(x_dim=projection_dim, y_dim=scRNA_encoder_output_dim, hidden_size=256)
    scRibo_mi_net = CLUBSample_group(x_dim=projection_dim, y_dim=scRibo_encoder_output_dim, hidden_size=256)
    
    Decoder = RA_VQVAE_Decoder(projection_dim, projection_dim, scRNA_output_dim, scRibo_output_dim)

    all_model = [Encoder, CPC, scRNA_mi_net, scRibo_mi_net, Decoder]
    model_to_double(all_model, devices)

    optimizer = torch.optim.Adam(chain(Encoder.parameters(), CPC.parameters(), Decoder.parameters()), lr=lr)

    optimizer_scRNA_mi_net = torch.optim.Adam(scRNA_mi_net.parameters(), lr=scRNA_mi_lr)
    optimizer_scRibo_mi_net = torch.optim.Adam(scRibo_mi_net.parameters(), lr=scRibo_mi_lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True, min_lr=1e-7)
    
    counter = 0
    patience = 30
    
    for epoch in tqdm(range(start_epoch + 1, n_epoch)):
        torch.cuda.empty_cache()
        flag = 1
        train_loss, total_step, train_scRNA_recon_loss, train_Ribo_recon_loss, train_mi_scRNA_loss, \
        train_mi_Ribo_loss, train_cpc_loss, codebook_stats, train_scRNA_embedding_loss, \
        train_Ribo_embedding_loss, train_cross_loss_rna, train_cross_loss_Ribo  = \
            pretrain_step(CPC, Encoder, scRNA_mi_net, scRibo_mi_net, Decoder, train_loader, optimizer,
                         optimizer_scRNA_mi_net, optimizer_scRibo_mi_net, flag, total_step, devices)
        
        writer.add_scalar('Loss/train_total_loss', train_loss, epoch)
        
        if ((epoch + 1) % eval_freq == 0) or (epoch == n_epoch - 1):
            flag = 0
            validate_loss, validate_foscttm_sum, val_cpc_loss, val_scRNA_recon_loss, val_Ribo_recon_loss, \
                val_cross_loss_rna, val_cross_loss_Ribo, val_scRNA_embedding_loss, val_Ribo_embedding_loss = \
                validate_test_epoch(CPC, Encoder, validate_loader, flag, devices, valiate_type='val', epoch=epoch, Decoder=Decoder)

            writer.add_scalar('Loss/val_total_loss', validate_loss, epoch)
            
            loss_history['train_total_loss'].append(train_loss)
            loss_history['val_total_loss'].append(validate_loss)
            loss_history['epochs'].append(epoch)

            if best_loss > validate_loss:
                best_loss = validate_loss
                save_path = os.path.join(model_save_path, 'pretrained-model.pt')
                save_models(CPC, Encoder, scRNA_mi_net, scRibo_mi_net, Decoder, optimizer, optimizer_scRNA_mi_net,
                            optimizer_scRibo_mi_net, epoch, total_step, save_path, train_loss=train_loss, val_loss=validate_loss, modal_key="scRibo")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break
                
            print(f"epoch: {epoch} | train_total_loss: {train_loss:.4f} | val_total_loss: {validate_loss:.4f}")

        scheduler.step(validate_loss)

    def _to_scalar(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return value

    safe_history = {
        key: [_to_scalar(item) for item in values]
        for key, values in loss_history.items()
    }
    df = pd.DataFrame(safe_history)
    df.to_csv(os.path.join(model_save_path, 'loss_history.csv'), index=False)
    
    create_detailed_loss_plots(loss_history, model_save_path)
    
    writer.close()

def create_detailed_loss_plots(loss_history, save_path):
    """Create standardized loss plots for RNA-Ribo training."""
    return
