import argparse
import os
import sys
import random


from .pretrain import pretrain_step, validate_test_epoch
from migo.model.layer.scRNA_scATAC import RA_VQVAE_Encoder, Cross_VQ_RA, RA_VQVAE_Decoder
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
import matplotlib

from migo.config import load_config, select_section, get_cfg, apply_overrides
from migo.runtime import seed_everything, resolve_device, ensure_dir

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2,
    "legend.frameon": False
})


def parse_args():
    parser = argparse.ArgumentParser(description="Train scRNA scATAC model")
    parser.add_argument("--config", default=None, help="Path to JSON config")
    return parser.parse_args()


def plot_loss_curves(epochs, val_epochs, train_losses, val_losses, save_dir):
    def _prepare_series(values):
        if torch.is_tensor(values):
            return values.detach().cpu().tolist()
        prepared = []
        for v in values:
            if torch.is_tensor(v):
                prepared.append(v.detach().cpu().item())
            else:
                prepared.append(float(v))
        return prepared

    train_losses = _prepare_series(train_losses)
    val_losses = _prepare_series(val_losses)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train', color='#1f77b4', alpha=0.85)
    plt.plot(val_epochs, val_losses, label='Validation', color='#ff7f0e', linestyle='--', alpha=0.85)
    plt.title('Training and Validation Total Loss', fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'total_loss_curves.pdf'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'total_loss_curves.png'), dpi=300)
    plt.show()



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
        "/home/linxw/2025_Projects/2025_journal_Project/scCMIA2_project/scRNA_scATAC_multi_version/2025_11_26_target_norm_version/Datasets/10X_PBMC/normalize_True_log1p_True_hvg_True_3000_RNA_processed_data.h5ad",
    )
    atac_path = get_cfg(
        cfg,
        "atac_h5ad",
        "/home/linxw/2025_Projects/2025_journal_Project/scCMIA2_project/scRNA_scATAC_multi_version/2025_11_26_target_norm_version/Datasets/10X_PBMC/binarize_True_filter_True_fpeaks_0.02_tfidf_True_normalize_True_ATAC_processed_data.h5ad",
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

    RNA_train_data, ATAC_train_data, label_train_data, RNA_validate_data, ATAC_validate_data, label_validate_data, \
        ori_RNA_train_data, ori_ATAC_train_data, ori_RNA_test_data, ori_ATAC_test_data \
        = split_data(ori_rna_dataset, scRNA_datasets, ori_atac_train_datasets, scATAC_datasets, label_tensor)
    # RNA_train_data, ATAC_train_data, label_train_data, RNA_validate_data, ATAC_validate_data, label_validate_data, _, _, _,\
    #     ori_RNA_train_data, ori_ATAC_train_data, ori_RNA_test_data, ori_ATAC_test_data, _, _ \
    #     = split_data_train_val_test(ori_rna_dataset, scRNA_datasets, ori_atac_train_datasets, scATAC_datasets, label_tensor)
    
    train_datasets = MultiModalDataset(RNA_train_data, ATAC_train_data, ori_RNA_train_data,
                                       ori_ATAC_train_data, label_train_data)
    validate_datasets = MultiModalDataset(RNA_validate_data, ATAC_validate_data, ori_RNA_test_data,
                                          ori_ATAC_test_data, label_validate_data)

    batch_size = int(get_cfg(cfg, "batch_size", 64))
    num_workers = int(get_cfg(cfg, "num_workers", 20))
    train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    validate_loader = DataLoader(validate_datasets, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)

    print('The number of train_datasets:', len(train_datasets))
    print('The number of validate_datasets:', len(validate_datasets))

    scRNA_dim = int(get_cfg(cfg, "scRNA_dim", 256))
    scATAC_dim = int(get_cfg(cfg, "scATAC_dim", 256))
    scRNA_output_dim = rna.layers['counts'].shape[1]
    scATAC_output_dim = atac.layers['counts'].shape[1]
    scRNA_encoder_output_dim = int(get_cfg(cfg, "scRNA_encoder_output_dim", 128))
    scATAC_encoder_output_dim = int(get_cfg(cfg, "scATAC_encoder_output_dim", 128))
    cpc_dim = int(get_cfg(cfg, "cpc_dim", 64))
    projection_dim = int(get_cfg(cfg, "projection_dim", 64))

    n_embeddings = int(get_cfg(cfg, "n_embeddings", 400)) #Codebook size
    embedding_dim = 128
    start_epoch = -1
    # model_resume = False
    total_step = 0
    n_epoch = int(get_cfg(cfg, "n_epoch", 500))
    eval_freq = int(get_cfg(cfg, "eval_freq", 1))
    model_save_path = get_cfg(
        cfg,
        "model_save_path",
        "/home/linxw/2025_Projects/2025_journal_Project/scCMIA2_project/scRNA_scATAC_multi_version/2025_12_2_improving_scRNA_scATAC_generated/Results_12_7_Test2/10X_PBMC",
    )
    scRNA_mi_lr = float(get_cfg(cfg, "scRNA_mi_lr", 0.1))
    scATAC_mi_lr = float(get_cfg(cfg, "scATAC_mi_lr", 0.1))
    lr = float(get_cfg(cfg, "lr", 0.0004))
    rna_distribution = get_cfg(cfg, "rna_distribution", "zin")
    best_focs = 1
    best_loss = 99999999999
    ensure_dir(model_save_path)
    log_dir = get_cfg(cfg, "log_dir", os.path.join(model_save_path, "logs"))
    ensure_dir(log_dir)
    writer = SummaryWriter(log_dir)

    train_losses = []
    val_losses = []

    Encoder = RA_VQVAE_Encoder(scRNA_dim, scATAC_dim, scRNA_encoder_output_dim, scATAC_encoder_output_dim, projection_dim, n_embeddings)
    #(256, 256, 128, 128, 128, 400)

    CPC = Cross_CPC_RA(cpc_dim, projection_dim)

    scRNA_mi_net = CLUBSample_group(x_dim=projection_dim, y_dim=scRNA_encoder_output_dim, hidden_size=256)
    scATAC_mi_net = CLUBSample_group(x_dim=projection_dim, y_dim=scATAC_encoder_output_dim, hidden_size=256)
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


    all_model = [Encoder, CPC, scRNA_mi_net, scATAC_mi_net, Decoder]
    model_to_double(all_model, devices)

    # print(all_model)
    optimizer = torch.optim.Adam(chain(Encoder.parameters(), CPC.parameters(), Decoder.parameters()), lr=lr)

    optimizer_scRNA_mi_net = torch.optim.Adam(scRNA_mi_net.parameters(), lr=scRNA_mi_lr)
    optimizer_scATAC_mi_net = torch.optim.Adam(scATAC_mi_net.parameters(), lr=scATAC_mi_lr)
    # optimizer_CPC_net = torch.optim.Adam(CPC.parameters(), lr=cpc_lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-7)

    counter = 0
    patience = 10
    
    for epoch in tqdm(range(start_epoch + 1, n_epoch)):
        torch.cuda.empty_cache()
        flag = 1
        loss, total_step, scRNA_recon_loss, scATAC_recon_loss, mi_scRNA_loss, mi_scATAC_loss, \
        cpc_loss, codebook_stats, scRNA_embedding_loss, scATAC_embedding_loss, cross_loss_rna, cross_loss_atac \
            = pretrain_step(CPC, Encoder, scRNA_mi_net, scATAC_mi_net, Decoder, train_loader, optimizer,
                           optimizer_scRNA_mi_net, optimizer_scATAC_mi_net, flag, total_step, devices)
        
        train_losses.append(loss)

        if ((epoch + 1) % eval_freq == 0) or (epoch == n_epoch - 1):
            flag = 0
            validate_loss, validate_foscttm_sum, val_cpc_loss, val_scRNA_recon_loss, val_scATAC_recon_loss, \
            val_scRNA_embedding_loss, val_scATAC_embedding_loss, val_cross_loss_rna, val_cross_loss_atac \
                = validate_test_epoch(CPC, Encoder, validate_loader, flag, devices, valiate_type='val', epoch=epoch, Decoder=Decoder)
            
            val_losses.append(validate_loss)

            writer.add_scalars('Total_Loss', {'train': loss, 'validation': validate_loss}, epoch)

            writer.add_scalar('Metrics/validate_foscttm_sum', validate_foscttm_sum, epoch)

            if best_loss > validate_loss:
                best_loss = validate_loss

                save_path = os.path.join(model_save_path, 'pretrained-model.pt')
                save_models(CPC, Encoder, scRNA_mi_net, scATAC_mi_net, Decoder, optimizer, optimizer_scRNA_mi_net,
                            optimizer_scATAC_mi_net, epoch, total_step, save_path, train_loss=loss, val_loss=validate_loss, modal_key="scATAC")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break
            print(f"epoch: {epoch} | train_total_loss: {loss:.4f} | val_total_loss: {validate_loss:.4f}")

        scheduler.step(validate_loss)

    plot_loss_curves(
        range(1, len(train_losses) + 1),
        range(1, len(val_losses) + 1),
        train_losses,
        val_losses,
        model_save_path,
    )

    writer.close()

if __name__ == '__main__':
    args = parse_args()
    main(args.config)



