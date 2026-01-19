import argparse
import os
import sys
import random


from .pretrain import pretrain_step, validate_test_epoch
from migo.model.layer.scRNA_ADT import RA_VQVAE_Encoder, RA_VQVAE_Decoder
from migo.model.CLUB import CLUBSample_group
import anndata as ad
from migo.model.utils import save_models, model_to_double
from migo.model.rta import Cross_CPC_RA
from itertools import chain
from torch.utils.data import DataLoader
from migo.model.dataset import (
    load_datasets,
    MultiModalDataset,
    split_data,
    create_data_loaders,
    split_data_train_val_test,
)
from sklearn.preprocessing import LabelEncoder
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

from migo.config import load_config, select_section, get_cfg, apply_overrides
from migo.runtime import seed_everything, resolve_device, ensure_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Train scRNA ADT model")
    parser.add_argument("--config", default=None, help="Path to JSON config")
    return parser.parse_args()

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
        "/normalize_True_log1p_True_hvg_True_3000_RNA_processed_data.h5ad",
    )
    adt_path = get_cfg(
        cfg,
        "adt_h5ad",
        "/processed_ADT.h5ad",
    )
    rna = ad.read_h5ad(rna_path)
    ADT = ad.read_h5ad(adt_path)

    scRNA_datasets = torch.tensor(rna.obsm['X_pca']).double()
    scADT_datasets = torch.tensor(ADT.obsm['X_pca']).double()

    ori_rna_dataset = torch.tensor(rna.layers['counts']).double()
    ori_ADT_train_datasets = torch.tensor(ADT.layers['counts_raw']).double()

    label = rna.obs['batch']
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(label.values)
    label_tensor = torch.tensor(label_encoded)

    RNA_train_data, ADT_train_data, label_train_data, RNA_validate_data, ADT_validate_data, label_validate_data, \
        ori_RNA_train_data, ori_ADT_train_data, ori_RNA_test_data, ori_ADT_test_data \
        = split_data(ori_rna_dataset, scRNA_datasets, ori_ADT_train_datasets, scADT_datasets, label_tensor, train_size1=0.8)

    train_datasets = MultiModalDataset(RNA_train_data, ADT_train_data, ori_RNA_train_data,
                                       ori_ADT_train_data, label_train_data)
    validate_datasets = MultiModalDataset(RNA_validate_data, ADT_validate_data, ori_RNA_test_data,
                                          ori_ADT_test_data, label_validate_data)

    batch_size = int(get_cfg(cfg, "batch_size", 1024))
    num_workers = int(get_cfg(cfg, "num_workers", 20))
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=False, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    validate_loader = DataLoader(validate_datasets, batch_size=batch_size, shuffle=False, drop_last=False, 
                                 num_workers=num_workers, pin_memory=True, persistent_workers=True)

    print('The number of train_datasets:', len(train_datasets))
    print('The number of validate_datasets:', len(validate_datasets))

    scRNA_dim = scRNA_datasets.shape[1]
    scADT_dim = scADT_datasets.shape[1]
    scRNA_output_dim = rna.layers['counts'].shape[1]
    scADT_output_dim = ADT.layers['counts_raw'].shape[1]
    scRNA_encoder_output_dim = int(get_cfg(cfg, "scRNA_encoder_output_dim", 128))
    scADT_encoder_output_dim = int(get_cfg(cfg, "scADT_encoder_output_dim", 128))
    cpc_dim = int(get_cfg(cfg, "cpc_dim", 64))
    projection_dim = int(get_cfg(cfg, "projection_dim", 64))

    n_embeddings = int(get_cfg(cfg, "n_embeddings", 1024)) #Codebook size
    embedding_dim = 128
    start_epoch = -1
    total_step = 0
    n_epoch = int(get_cfg(cfg, "n_epoch", 500))
    eval_freq = int(get_cfg(cfg, "eval_freq", 1))
    model_save_path = get_cfg(cfg, "model_save_path", "/Results")
    scRNA_mi_lr = float(get_cfg(cfg, "scRNA_mi_lr", 0.1))
    scADT_mi_lr = float(get_cfg(cfg, "scADT_mi_lr", 0.1))
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

    Encoder = RA_VQVAE_Encoder(scRNA_dim, scADT_dim, scRNA_encoder_output_dim, scADT_encoder_output_dim, projection_dim, n_embeddings)

    CPC = Cross_CPC_RA(cpc_dim, projection_dim)

    scRNA_mi_net = CLUBSample_group(x_dim=projection_dim, y_dim=scRNA_encoder_output_dim, hidden_size=256)
    scADT_mi_net = CLUBSample_group(x_dim=projection_dim, y_dim=scADT_encoder_output_dim, hidden_size=256)
    
    Decoder = RA_VQVAE_Decoder(projection_dim, projection_dim, scRNA_output_dim, scADT_output_dim)

    all_model = [Encoder, CPC, scRNA_mi_net, scADT_mi_net, Decoder]
    model_to_double(all_model, devices)

    optimizer = torch.optim.Adam(chain(Encoder.parameters(), CPC.parameters(), Decoder.parameters()), lr=lr)

    optimizer_scRNA_mi_net = torch.optim.Adam(scRNA_mi_net.parameters(), lr=scRNA_mi_lr)
    optimizer_scADT_mi_net = torch.optim.Adam(scADT_mi_net.parameters(), lr=scADT_mi_lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True, min_lr=1e-7)
    
    counter = 0
    patience = 10
    
    for epoch in tqdm(range(start_epoch + 1, n_epoch)):
        torch.cuda.empty_cache()
        flag = 1
        train_loss, total_step, train_scRNA_recon_loss, train_ADT_recon_loss, train_mi_scRNA_loss, \
        train_mi_ADT_loss, train_cpc_loss, codebook_stats, train_scRNA_embedding_loss, \
        train_ADT_embedding_loss, train_cross_loss_rna, train_cross_loss_ADT = \
            pretrain_step(CPC, Encoder, scRNA_mi_net, scADT_mi_net, Decoder, train_loader, optimizer,
                         optimizer_scRNA_mi_net, optimizer_scADT_mi_net, flag, total_step, devices)
        
        writer.add_scalar('Loss/train_total_loss', train_loss, epoch)
        
        if ((epoch + 1) % eval_freq == 0) or (epoch == n_epoch - 1):
            flag = 0
            validate_loss, validate_foscttm_sum, val_cpc_loss, val_scRNA_recon_loss, val_ADT_recon_loss, \
                val_cross_loss_rna, val_cross_loss_ADT, val_scRNA_embedding_loss, val_ADT_embedding_loss = \
                validate_test_epoch(CPC, Encoder, validate_loader, flag, devices, valiate_type='val', epoch=epoch, Decoder=Decoder)


            writer.add_scalar('Loss/val_total_loss', validate_loss, epoch)
            
            loss_history['train_total_loss'].append(train_loss)
            loss_history['val_total_loss'].append(validate_loss)
            loss_history['epochs'].append(epoch)

            # Save model
            if best_loss > validate_loss:
                best_loss = validate_loss
                save_path = os.path.join(model_save_path, 'pretrained-model.pt')
                save_models(
                    CPC,
                    Encoder,
                    scRNA_mi_net,
                    scADT_mi_net,
                    Decoder,
                    optimizer,
                    optimizer_scRNA_mi_net,
                    optimizer_scADT_mi_net,
                    epoch,
                    total_step,
                    save_path,
                    train_loss=train_loss,
                    val_loss=validate_loss,
                    modal_key="scADT",
                )
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break
                
            print(f"epoch: {epoch} | train_total_loss: {train_loss:.4f} | val_total_loss: {validate_loss:.4f}")

        scheduler.step(validate_loss)

    # Save loss history to CSV file
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
    def _to_scalar(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return value

    loss_history = {
        key: [_to_scalar(item) for item in values]
        for key, values in loss_history.items()
    }

    epochs = loss_history.get('epochs', [])
    train_total = loss_history.get('train_total_loss', [])
    val_total = loss_history.get('val_total_loss', [])

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_total, color='#2E86AB', linewidth=2.5, label='Training Total Loss', marker='o', markersize=4, alpha=0.8)
    plt.plot(epochs, val_total, color='#A23B72', linewidth=2.5, label='Validation Total Loss', marker='s', markersize=4, alpha=0.8)
    plt.title('Total Loss Curve', fontsize=18, fontweight='bold', pad=12)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.legend(fontsize=11, loc='upper right', frameon=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'total_loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_training_summary(loss_history, save_path):
    return

if __name__ == '__main__':
    args = parse_args()
    main(args.config)






