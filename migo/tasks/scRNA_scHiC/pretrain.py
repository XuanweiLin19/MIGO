import os

_RESOLUTION = os.environ.get("MIGO_RESOLUTION", "1M")

if _RESOLUTION not in ("1M", "50K"):
    raise ValueError(f"Unsupported MIGO_RESOLUTION: {_RESOLUTION}")

from migo.model.utils import AverageMeter, Prepare_logger, save_models
import torch
import logging
from migo.model.container import metricsContainer
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from migo.model.metrics import scRNA_scHiC as metrics
import torch.nn.functional as F
from itertools import chain


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, (list, tuple)):
        return [to_device(item, device) for item in data]
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    return data


def to_train_model(all_model, device):
    for model in all_model:
        model.train()
        model.to(device)


def to_eval_model(all_model, device):
    for model in all_model:
        model.eval()
        model.to(device)


def _unpack_cpc_output(cpc_output):
    if not isinstance(cpc_output, (tuple, list)):
        return cpc_output, None, None
    if len(cpc_output) == 3:
        return cpc_output
    if len(cpc_output) >= 7:
        return cpc_output[0], cpc_output[-2], cpc_output[-1]
    if len(cpc_output) == 2:
        return cpc_output[0], cpc_output[1], None
    if len(cpc_output) == 1:
        return cpc_output[0], None, None
    return cpc_output[0], None, None


def pretrain_step(
    CPC,
    Encoder,
    scRNA_mi_net,
    scHiC_mi_net,
    Decoder,
    train_dataloader,
    optimizer,
    optimizer_scRNA_mi_net,
    optimizer_scHiC_mi_net,
    flag,
    total_step,
    device,
    epoch,
    warmup_epochs,
    cross_modal_target_weight,
):
    losses = AverageMeter()
    models = [CPC, Encoder, scRNA_mi_net, scHiC_mi_net, Decoder]
    to_train_model(models, device)

    mi_iters = 5
    train_dataloader = tqdm(train_dataloader)
    batch_stats_list = []

    recon_weight = 1.0
    cross_weight = 2.0
    cpc_weight = 0.5
    mi_weight = 0.3
    vq_weight = 0.5

    for n_iter, batch_data in enumerate(train_dataloader):
        scRNA_ori_data, scRNA_raw_embedding, HiC_ori_data, HiC_raw_embedding, labels = batch_data

        scRNA_ori_data = scRNA_ori_data.to(device)
        scRNA_raw_embedding = scRNA_raw_embedding.to(device)
        HiC_ori_data = HiC_ori_data.to(device)
        HiC_raw_embedding = HiC_raw_embedding.to(device)

        hic_size = int(np.sqrt(HiC_ori_data.shape[1]))
        HiC_2d_data = HiC_ori_data.reshape(-1, hic_size, hic_size).to(device)

        for _ in range(mi_iters):
            optimizer_scRNA_mi_net, optimizer_scHiC_mi_net, lld_scRNA_loss, lld_HiC_loss = \
                Mutimodal_Decoupling_Step(
                    scRNA_raw_embedding,
                    HiC_2d_data,
                    Encoder,
                    scRNA_mi_net,
                    scHiC_mi_net,
                    optimizer_scRNA_mi_net,
                    optimizer_scHiC_mi_net,
                    flag,
                    CPC,
                    device,
                )

        (
            scRNA_embedding_loss,
            HiC_embedding_loss,
            mi_scRNA_loss,
            mi_HiC_loss,
            cpc_loss,
            scRNA_recon_loss,
            HiC_recon_loss,
            cross_loss_rna,
            cross_loss_HiC,
            latent_imitation_loss,
            batch_stats,
        ) = Mutimodal_Decoupling_second_forward(
            CPC,
            scRNA_raw_embedding,
            HiC_2d_data,
            Encoder,
            scRNA_mi_net,
            scHiC_mi_net,
            Decoder,
            flag,
            scRNA_ori_data,
            HiC_ori_data,
            device,
        )

        batch_stats_list.append(batch_stats)

        # 在主优化前清零梯度
        optimizer.zero_grad()
        
        loss = (
            recon_weight * (scRNA_recon_loss + HiC_recon_loss)
            + cross_weight * (cross_loss_rna + cross_loss_HiC)
            + vq_weight * (scRNA_embedding_loss + HiC_embedding_loss)
            + cpc_weight * cpc_loss
            + mi_weight * (mi_scRNA_loss + mi_HiC_loss)
        )
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), scRNA_raw_embedding.size(0))

    avg_batch_stats = {
        "rna_util": sum(stats["rna_util"] for stats in batch_stats_list) / len(batch_stats_list),
        "HiC_util": sum(stats["HiC_util"] for stats in batch_stats_list) / len(batch_stats_list),
        "batch_util": sum(stats["batch_util"] for stats in batch_stats_list) / len(batch_stats_list),
    }

    global_stats = Encoder.Cross_quantizer.get_codebook_utilization()
    codebook_stats = {**avg_batch_stats, **global_stats}

    return (
        losses.avg,
        n_iter + total_step,
        scRNA_recon_loss,
        HiC_recon_loss,
        mi_scRNA_loss,
        mi_HiC_loss,
        cpc_loss,
        codebook_stats,
        cross_loss_rna,
        cross_loss_HiC,
        scRNA_embedding_loss,
        HiC_embedding_loss,
        latent_imitation_loss,
    )


def Mutimodal_Decoupling_Step(
    scRNA_raw_embedding,
    HiC_2d_data,
    Encoder,
    scRNA_mi_net,
    scHiC_mi_net,
    optimizer_scRNA_mi_net,
    optimizer_scHiC_mi_net,
    flag,
    CPC,
    device,
):
    optimizer_scRNA_mi_net.zero_grad()
    optimizer_scHiC_mi_net.zero_grad()

    scRNA_raw_embedding = scRNA_raw_embedding.to(device)
    HiC_2d_data = HiC_2d_data.to(device)

    (scRNA_semantic_result, HiC_semantic_result, scRNA_encoder_result, HiC_encoder_result,
     _, _, _, _, _, _, _) = Encoder(scRNA_raw_embedding, HiC_2d_data, flag, CPC, device)

    RtA_scRNA_semantic_result = scRNA_semantic_result.detach()
    RtA_HiC_semantic_result = HiC_semantic_result.detach()
    scRNA_encoder_result = scRNA_encoder_result.detach()
    HiC_encoder_result = HiC_encoder_result.detach()

    lld_scRNA_loss = -scRNA_mi_net.loglikeli(RtA_scRNA_semantic_result, scRNA_encoder_result)
    lld_scRNA_loss.backward()
    optimizer_scRNA_mi_net.step()

    lld_HiC__loss = -scHiC_mi_net.loglikeli(RtA_HiC_semantic_result, HiC_encoder_result)
    lld_HiC__loss.backward()
    optimizer_scHiC_mi_net.step()

    return optimizer_scRNA_mi_net, optimizer_scHiC_mi_net, lld_scRNA_loss, lld_HiC__loss


def Mutimodal_Decoupling_second_forward(
    CPC,
    scRNA_raw_embedding,
    HiC_2d_data,
    Encoder,
    scRNA_mi_net,
    scHiC_mi_net,
    Decoder,
    flag,
    scRNA_ori_data,
    HiC_ori_data,
    device,
):
    scRNA_raw_embedding = scRNA_raw_embedding.to(device)
    HiC_2d_data = HiC_2d_data.to(device)
    scRNA_ori_data = scRNA_ori_data.to(device)
    HiC_ori_data = HiC_ori_data.to(device)

    (scRNA_semantic_result, HiC_semantic_result, scRNA_encoder_result, HiC_encoder_result,
     scRNA_vq, HiC_semantic_vq,
     scRNA_embedding_loss, HiC_embedding_loss,
     batch_stats,
     hic_skips,
     hic_bottleneck) = Encoder(scRNA_raw_embedding, HiC_2d_data, flag, CPC, device)

    mi_scRNA_loss = scRNA_mi_net.mi_est(scRNA_semantic_result, scRNA_encoder_result)
    mi_HiC_loss = scHiC_mi_net.mi_est(HiC_semantic_result, HiC_encoder_result)

    cpc_output = CPC(scRNA_semantic_result, HiC_semantic_result, device)
    cpc_loss, _, _ = _unpack_cpc_output(cpc_output)

    scRNA_recon_loss, HiC_recon_loss, scRNA_recon_result, HiC_recon_result, cross_loss_rna, cross_loss_HiC, \
    cross_scRNA, cross_scHiC, latent_imitation_loss = Decoder(
        scRNA_ori_data,
        HiC_ori_data,
        scRNA_vq,
        scRNA_semantic_result,
        scRNA_encoder_result,
        hic_bottleneck,
        HiC_semantic_vq,
        hic_skips,
        Encoder.hic_unet,
    )

    return (
        scRNA_embedding_loss,
        HiC_embedding_loss,
        mi_scRNA_loss,
        mi_HiC_loss,
        cpc_loss,
        scRNA_recon_loss,
        HiC_recon_loss,
        cross_loss_rna,
        cross_loss_HiC,
        latent_imitation_loss,
        batch_stats,
    )


def validate_test_epoch(
    CPC,
    Encoder,
    validate_test_dataloader,
    flag,
    device,
    valiate_type,
    epoch,
    Decoder,
    warmup_epochs,
    cross_modal_target_weight,
    scRNA_mi_net,
    scHiC_mi_net,
):
    eval_models = [CPC, Encoder, Decoder]
    to_eval_model(eval_models, device)

    validate_foscttm_sum = 0
    length_data = len(validate_test_dataloader)
    validate_test_dataloader = tqdm(validate_test_dataloader)

    total_cpc_loss = 0
    total_scRNA_recon_loss = 0
    total_HiC_recon_loss = 0
    total_cross_loss_rna = 0
    total_cross_loss_HiC = 0
    total_latent_imitation_loss = 0
    total_scRNA_embedding_loss = 0
    total_HiC_embedding_loss = 0
    total_mi_scRNA_loss = 0
    total_mi_HiC_loss = 0

    with torch.no_grad():
        for n_iter, batch_data in enumerate(validate_test_dataloader):
            scRNA_ori_data, scRNA_raw_embedding, HiC_ori_data, HiC_raw_embedding, labels = batch_data

            scRNA_ori_data = scRNA_ori_data.to(device)
            scRNA_raw_embedding = scRNA_raw_embedding.to(device)
            HiC_ori_data = HiC_ori_data.to(device)
            HiC_raw_embedding = HiC_raw_embedding.to(device)

            hic_size = int(np.sqrt(HiC_ori_data.shape[1]))
            HiC_2d_data = HiC_ori_data.reshape(-1, hic_size, hic_size).to(device)

            (scRNA_semantic_result, HiC_semantic_result, scRNA_encoder_result, HiC_encoder_result,
             scRNA_vq, HiC_semantic_vq,
             scRNA_embedding_loss, HiC_embedding_loss,
             batch_stats,
             hic_skips,
             hic_quantized_bottleneck) = Encoder(scRNA_raw_embedding, HiC_2d_data, flag, CPC, device)

            cpc_output = CPC(scRNA_semantic_result, HiC_semantic_result, device)
            cpc_loss, rna_embedding_norm, hic_embedding_norm = _unpack_cpc_output(cpc_output)
            if rna_embedding_norm is not None and hic_embedding_norm is not None:
                validate_foscttm_sum += metrics.foscttm(
                    rna_embedding_norm.detach().cpu().numpy(),
                    hic_embedding_norm.detach().cpu().numpy(),
                )

            scRNA_recon_loss, HiC_recon_loss, scRNA_recon_result, HiC_recon_result, cross_loss_rna, cross_loss_HiC, \
            _, cross_scHiC, latent_imitation_loss = Decoder(
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

            mi_scRNA_loss = scRNA_mi_net.mi_est(scRNA_semantic_result, scRNA_encoder_result)
            mi_HiC_loss = scHiC_mi_net.mi_est(HiC_semantic_result, HiC_encoder_result)

            total_mi_scRNA_loss += mi_scRNA_loss.item()
            total_mi_HiC_loss += mi_HiC_loss.item()

            total_cpc_loss += cpc_loss.item()
            total_scRNA_recon_loss += scRNA_recon_loss.item()
            total_HiC_recon_loss += HiC_recon_loss.item()
            total_cross_loss_rna += cross_loss_rna.item()
            total_cross_loss_HiC += cross_loss_HiC.item()
            if torch.is_tensor(latent_imitation_loss):
                total_latent_imitation_loss += latent_imitation_loss.item()
            else:
                total_latent_imitation_loss += float(latent_imitation_loss)
            total_scRNA_embedding_loss += scRNA_embedding_loss.item()
            total_HiC_embedding_loss += HiC_embedding_loss.item()

    avg_cpc_loss = total_cpc_loss / length_data
    avg_scRNA_recon_loss = total_scRNA_recon_loss / length_data
    avg_HiC_recon_loss = total_HiC_recon_loss / length_data
    avg_cross_loss_rna = total_cross_loss_rna / length_data
    avg_cross_loss_HiC = total_cross_loss_HiC / length_data
    avg_latent_imitation_loss = total_latent_imitation_loss / length_data
    avg_scRNA_embedding_loss = total_scRNA_embedding_loss / length_data
    avg_HiC_embedding_loss = total_HiC_embedding_loss / length_data
    avg_mi_scRNA_loss = total_mi_scRNA_loss / length_data
    avg_mi_HiC_loss = total_mi_HiC_loss / length_data

    recon_weight = 1.0
    cross_weight = 2.0
    cpc_weight = 0.5
    mi_weight = 0.3
    vq_weight = 0.5

    total_loss = (
        recon_weight * (avg_scRNA_recon_loss + avg_HiC_recon_loss)
        + cross_weight * (avg_cross_loss_rna + avg_cross_loss_HiC)
        + vq_weight * (avg_scRNA_embedding_loss + avg_HiC_embedding_loss)
        + cpc_weight * avg_cpc_loss
        + mi_weight * (avg_mi_scRNA_loss + avg_mi_HiC_loss)
    )

    return (
        total_loss,
        avg_cpc_loss,
        avg_scRNA_recon_loss,
        avg_HiC_recon_loss,
        avg_cross_loss_rna,
        avg_cross_loss_HiC,
        avg_scRNA_embedding_loss,
        avg_HiC_embedding_loss,
        avg_mi_scRNA_loss,
        avg_mi_HiC_loss,
        avg_latent_imitation_loss,
    )
