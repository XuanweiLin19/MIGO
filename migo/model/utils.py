#!/usr/bin/env python
import copy
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import ModuleList
from torch.nn import functional as F

try:
    import scib
except ImportError:
    scib = None


class EarlyStopping:
    """Early stop if validation loss stops improving."""

    def __init__(self, patience=10, verbose=False, checkpoint_file=""):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.checkpoint_file = checkpoint_file

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.checkpoint_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        if self.verbose:
            print(f"Loss decreased ({self.loss_min:.6f} --> {loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_file)
        self.loss_min = loss


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Prepare_logger(args, eval=False):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(0)
    logger.addHandler(handler)

    date = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
    logfile = args.snapshot_pref + date + ".log" if not eval else args.snapshot_pref + f"/{date}-Eval.log"
    file_handler = logging.FileHandler(logfile, mode="w")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _get_clones(module, n):
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def model_to_double(all_model, device):
    for model in all_model:
        model.double()
        model.to(device)


def save_models(
    CPC,
    Encoder,
    scRNA_mi_net,
    scModal_mi_net,
    Decoder,
    optimizer,
    optimizer_scRNA_mi_net,
    optimizer_scModal_mi_net,
    epoch_num,
    total_step,
    path,
    train_loss=None,
    val_loss=None,
    modal_key="scADT",
):
    state_dict = {
        "Encoder_parameters": Encoder.state_dict(),
        "CPC_parameters": CPC.state_dict(),
        "scRNA_mi_net_parameters": scRNA_mi_net.state_dict(),
        f"{modal_key}_mi_net_parameters": scModal_mi_net.state_dict(),
        "Decoder_parameters": Decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_scRNA_mi_net": optimizer_scRNA_mi_net.state_dict(),
        f"optimizer_{modal_key}_mi_net": optimizer_scModal_mi_net.state_dict(),
        "epoch": epoch_num,
        "total_step": total_step,
    }
    if train_loss is not None:
        state_dict["train_loss"] = train_loss
    if val_loss is not None:
        state_dict["val_loss"] = val_loss

    torch.save(state_dict, path)

    log_message = f"Save model to {path} at epoch {epoch_num}."
    if val_loss is not None:
        log_message += f" Best val_loss: {val_loss:.4f}."
    logging.info(log_message)


def calculate_Batch_Correlation_Score(ori_data, test_data):
    if scib is None:
        raise ImportError("scib is required for batch correlation metrics.")
    all_results = scib.metrics.metrics(
        test_data,
        ori_data,
        batch_key="batch",
        label_key="cell_type",
        type_="embed",
        verbose=True,
    )
    ilisi_score = all_results.get("ilisi")
    gc_score = all_results.get("graph_connectivity")
    kbet_score = all_results.get("kBET")
    return ilisi_score, gc_score, kbet_score, all_results


def to_cpu_float_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, list):
        return [float(v.cpu().item()) if isinstance(v, torch.Tensor) else float(v) for v in x]
    if hasattr(x, "cpu"):
        return x.cpu().numpy().tolist()
    return list(x)


def save_loss_plot(train_losses, val_losses, eval_freq, output_dir, filename="loss_plot.png"):
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(train_losses, dict):
        loss_names = list(train_losses.keys())
        n_losses = len(loss_names)
        ncols = 2
        nrows = (n_losses + 1) // ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))
        axes = axes.flatten()

        for idx, loss_name in enumerate(loss_names):
            ax = axes[idx]
            train_vals = to_cpu_float_list(train_losses[loss_name])
            train_epochs = list(range(1, len(train_vals) + 1))
            ax.plot(train_epochs, train_vals, "b-o", label="Train", markersize=3, alpha=0.7)
            if loss_name in val_losses and len(val_losses[loss_name]) > 0:
                val_vals = to_cpu_float_list(val_losses[loss_name])
                val_epochs = list(range(eval_freq, eval_freq * len(val_vals) + 1, eval_freq))
                ax.plot(val_epochs, val_vals, "r-o", label="Val", markersize=3, alpha=0.7)
            ax.set_title(loss_name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(loss_name)
            ax.legend()
            ax.grid(True)

        for j in range(idx + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        try:
            logging.info(f"Loss plot saved to {save_path}")
        except Exception:
            print(f"Loss plot saved to {save_path}")
        return

    train_epochs = range(1, len(train_losses) + 1)
    val_epochs = range(eval_freq, len(train_losses) + 1, eval_freq)[: len(val_losses)]

    plt.figure(figsize=(12, 8))
    plt.plot(train_epochs, train_losses, "b-o", label="Training Loss", markersize=4, alpha=0.8)
    if val_losses:
        plt.plot(val_epochs, val_losses, "r-s", label="Validation Loss", markersize=5)

    plt.title("Training and Validation Loss Over Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.minorticks_on()

    if val_losses:
        min_val_loss = min(val_losses)
        min_val_epoch = val_epochs[val_losses.index(min_val_loss)]
        plt.axvline(
            x=min_val_epoch,
            color="grey",
            linestyle="--",
            linewidth=1,
            label=f"Best Val Loss Epoch: {min_val_epoch}",
        )
        plt.scatter(
            [min_val_epoch],
            [min_val_loss],
            s=100,
            color="red",
            zorder=5,
            marker="*",
            label=f"Best Val Loss: {min_val_loss:.4f}",
        )
        plt.legend(fontsize=11)

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    try:
        logging.info(f"Loss plot saved to {save_path}")
    except Exception:
        print(f"Loss plot saved to {save_path}")
