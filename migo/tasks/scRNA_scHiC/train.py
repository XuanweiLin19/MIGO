import argparse
import importlib
from migo.config import load_config, select_section, apply_overrides
from migo.runtime import seed_everything, resolve_device
import os
import sys



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
    cfg = select_section(root_cfg, "train")
    cfg = apply_overrides(cfg, overrides)
    resolution = _resolve_resolution(root_cfg, overrides)
    os.environ["MIGO_RESOLUTION"] = resolution

    import random
    import logging
    import numpy as np
    import torch
    import anndata as ad
    import pandas as pd
    from itertools import chain
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from tqdm import tqdm
    from sklearn.preprocessing import LabelEncoder
    RA_VQVAE_Encoder, RA_VQVAE_Decoder, Cross_CPC_RA = _load_scHic_layers(resolution)
    from migo.model.CLUB import CLUBSample_group
    from migo.model.utils import save_models, save_loss_plot
    from migo.model.dataset import MultiModalDataset, split_data
    from .pretrain import pretrain_step, validate_test_epoch

    def save_losses_to_csv(train_loss_history, val_loss_history, eval_freq, output_dir, filename="training_losses.csv"):
        os.makedirs(output_dir, exist_ok=True)

        train_epochs = list(range(len(train_loss_history["total_loss"])))
        val_epochs = [i * eval_freq for i in range(len(val_loss_history["total_loss"]))]

        train_data = {
            "epoch": train_epochs,
            "phase": ["train"] * len(train_epochs),
        }
        for loss_name, loss_values in train_loss_history.items():
            train_data[loss_name] = loss_values

        val_data = {
            "epoch": val_epochs,
            "phase": ["validation"] * len(val_epochs),
        }
        for loss_name, loss_values in val_loss_history.items():
            val_data[loss_name] = loss_values

        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        combined_df = combined_df.sort_values("epoch").reset_index(drop=True)

        csv_path = os.path.join(output_dir, filename)
        combined_df.to_csv(csv_path, index=False, float_format="%.6f")

        train_csv_path = os.path.join(output_dir, "train_losses.csv")
        val_csv_path = os.path.join(output_dir, "validation_losses.csv")
        train_df.to_csv(train_csv_path, index=False, float_format="%.6f")
        val_df.to_csv(val_csv_path, index=False, float_format="%.6f")

        print(f"Losses saved to: {csv_path}")
        print(f"Training losses saved to: {train_csv_path}")
        print(f"Validation losses saved to: {val_csv_path}")

        return csv_path, train_csv_path, val_csv_path

    def tensor_to_float(value):
        if torch.is_tensor(value):
            return value.detach().cpu().item()
        return float(value)

    seed_everything(cfg.get("seed", 0))
    logger = logging.getLogger(__name__)
    global best_accuracy, best_accuracy_epoch, validate_loss
    best_accuracy, best_accuracy_epoch = 0, 0

    device_override = cfg.get("device")
    devices = resolve_device(device_override)

    if devices.type == "cuda":
        gpu_index = devices.index if devices.index is not None else 0
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_index)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(gpu_index).total_memory / 1024**3:.1f} GB")
    else:
        print("Using CPU")

    rna_path = cfg.get("rna_h5ad")
    hic_path = cfg.get("hic_h5ad")
    if not rna_path or not hic_path:
        raise ValueError("train.rna_h5ad and train.hic_h5ad are required")

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

    train_datasets = MultiModalDataset(
        RNA_train_data,
        HiC_train_data,
        ori_RNA_train_data,
        ori_HiC_train_data,
        label_train_data,
    )
    validate_datasets = MultiModalDataset(
        RNA_validate_data,
        HiC_validate_data,
        ori_RNA_test_data,
        ori_HiC_test_data,
        label_validate_data,
    )

    train_batch_size = cfg.get("train_batch_size", 64)
    val_batch_size = cfg.get("val_batch_size", 64)
    train_loader = DataLoader(train_datasets, batch_size=train_batch_size, shuffle=True, drop_last=True)
    validate_loader = DataLoader(validate_datasets, batch_size=val_batch_size, shuffle=False, drop_last=True)

    print("The number of train_datasets:", len(train_datasets))
    print("The number of validate_datasets:", len(validate_datasets))

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

    start_epoch = -1
    total_step = 0
    n_epoch = cfg.get("n_epoch", 500)
    eval_freq = cfg.get("eval_freq", 1)

    warmup_epochs = cfg.get("warmup_epochs", 20)
    cross_modal_target_weight = cfg.get("cross_modal_target_weight", 2.0)

    model_save_path = cfg.get("model_save_path")
    if not model_save_path:
        raise ValueError("train.model_save_path is required")

    best_loss = 99999999999
    log_dir = cfg.get("log_dir", os.path.join(model_save_path, "logs"))
    writer = SummaryWriter(log_dir)

    train_loss_history = {
        "total_loss": [],
    }
    val_loss_history = {
        "total_loss": [],
    }

    codebook_history = {
        "epoch": [],
        "global_utilization": [],
        "rna_utilization": [],
        "HiC_utilization": [],
        "cross_utilization": [],
        "batch_utilization": [],
        "active_codes": [],
    }

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

    scRNA_mi_net = CLUBSample_group(x_dim=embedding_dim, y_dim=scRNA_encoder_output_dim, hidden_size=256)
    scHiC_mi_net = CLUBSample_group(x_dim=embedding_dim, y_dim=scHiC_encoder_output_dim, hidden_size=256)

    lr = cfg.get("lr", 5e-6)
    weight_decay = cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.Adam(
        chain(Encoder.parameters(), CPC.parameters(), Decoder.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    scRNA_mi_lr = cfg.get("scRNA_mi_lr", 0.0005)
    scHiC_mi_lr = cfg.get("scHiC_mi_lr", 0.0005)
    optimizer_scRNA_mi_net = torch.optim.Adam(scRNA_mi_net.parameters(), lr=scRNA_mi_lr, weight_decay=weight_decay)
    optimizer_scHiC_mi_net = torch.optim.Adam(scHiC_mi_net.parameters(), lr=scHiC_mi_lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True, min_lr=1e-7)

    counter = 0
    patience = 10

    for epoch in tqdm(range(start_epoch + 1, n_epoch)):
        torch.cuda.empty_cache()
        flag = 1
        (train_total_loss, total_step,
         train_scRNA_recon_loss, train_HiC_recon_loss, train_mi_scRNA_loss, train_mi_HiC_loss,
         train_cpc_loss, codebook_stats, train_cross_loss_rna, train_cross_loss_HiC,
         train_scRNA_embedding_loss, train_HiC_embedding_loss,
         train_latent_imitation_loss) = pretrain_step(
            CPC, Encoder, scRNA_mi_net, scHiC_mi_net, Decoder, train_loader, optimizer,
            optimizer_scRNA_mi_net, optimizer_scHiC_mi_net, flag, total_step, devices,
            epoch, warmup_epochs, cross_modal_target_weight
        )

        train_loss_history["total_loss"].append(tensor_to_float(train_total_loss))

        codebook_history["epoch"].append(epoch)
        codebook_history["global_utilization"].append(tensor_to_float(codebook_stats["global_util"]))
        codebook_history["rna_utilization"].append(tensor_to_float(codebook_stats["rna_util"]))
        codebook_history["HiC_utilization"].append(tensor_to_float(codebook_stats["HiC_util"]))
        codebook_history["cross_utilization"].append(tensor_to_float(codebook_stats["cross_util"]))
        codebook_history["batch_utilization"].append(tensor_to_float(codebook_stats["batch_util"]))
        codebook_history["active_codes"].append(int(codebook_stats["active_codes"]))

        if ((epoch + 1) % eval_freq == 0) or (epoch == n_epoch - 1):
            flag = 0
            (val_total_loss, val_cpc_loss, val_scRNA_recon_loss, val_HiC_recon_loss,
             val_cross_loss_rna, val_cross_loss_HiC, val_scRNA_embedding_loss, val_HiC_embedding_loss,
             val_mi_scRNA_loss, val_mi_HiC_loss, val_latent_imitation_loss) = validate_test_epoch(
                CPC,
                Encoder,
                validate_loader,
                flag,
                devices,
                valiate_type="val",
                epoch=epoch,
                Decoder=Decoder,
                warmup_epochs=warmup_epochs,
                cross_modal_target_weight=cross_modal_target_weight,
                scRNA_mi_net=scRNA_mi_net,
                scHiC_mi_net=scHiC_mi_net,
            )

            val_loss_history["total_loss"].append(tensor_to_float(val_total_loss))

            writer.add_scalars("Total_Loss", {"train": tensor_to_float(train_total_loss), "validation": tensor_to_float(val_total_loss)}, epoch)

            if best_loss > tensor_to_float(val_total_loss):
                best_loss = tensor_to_float(val_total_loss)
                save_path = os.path.join(model_save_path, "pretrained-model.pt")
                save_models(
                    CPC,
                    Encoder,
                    scRNA_mi_net,
                    scHiC_mi_net,
                    Decoder,
                    optimizer,
                    optimizer_scRNA_mi_net,
                    optimizer_scHiC_mi_net,
                    epoch,
                    total_step,
                    save_path,
                    modal_key="scHiC",
                )
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break

            print(f"epoch: {epoch} | train_total_loss: {tensor_to_float(train_total_loss):.4f} | val_total_loss: {tensor_to_float(val_total_loss):.4f}")

        scheduler.step(tensor_to_float(val_total_loss))

        if (epoch + 1) % 50 == 0:
            save_losses_to_csv(
                train_loss_history,
                val_loss_history,
                eval_freq,
                model_save_path,
                f"losses_epoch_{epoch+1}.csv",
            )

    writer.close()

    print("Training finished. Saving final loss data to CSV...")
    combined_csv, train_csv, val_csv = save_losses_to_csv(
        train_loss_history,
        val_loss_history,
        eval_freq,
        model_save_path,
        "final_training_losses.csv",
    )

    codebook_df = pd.DataFrame(codebook_history)
    codebook_csv_path = os.path.join(model_save_path, "codebook_statistics.csv")
    codebook_df.to_csv(codebook_csv_path, index=False, float_format="%.6f")
    print(f"Codebook statistics saved to: {codebook_csv_path}")

    save_loss_plot(
        train_losses_dict=train_loss_history,
        val_losses_dict=val_loss_history,
        eval_freq=eval_freq,
        output_dir=model_save_path,
        filename="training_validation_loss.png",
    )

    print(f"All results saved to: {model_save_path}")
    print(f"Combined losses: {combined_csv}")
    print(f"Training losses: {train_csv}")
    print(f"Validation losses: {val_csv}")
    print(f"Codebook statistics: {codebook_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train scRNA-scHiC model")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    args = parser.parse_args()
    main(load_config(args.config))







