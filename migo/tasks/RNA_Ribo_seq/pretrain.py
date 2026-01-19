from migo.model.utils import AverageMeter, Prepare_logger, save_models
import torch
import os
import logging
from migo.model.container import metricsContainer
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from migo.model.metrics import RNA_Ribo_seq as metrics


def to_train_model(all_model, device):
    for model in all_model:
        model.train()
        model.to(device)


def to_eval_model(all_model, device):
    for model in all_model:
        model.eval()
        model.to(device)

class DynamicWeightScheduler:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
    
    def get_weights(self, epoch):
        return {}


import torch.nn.functional as F
def batch_cosine_similarity(x, y):
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    sim_mat_x = torch.mm(x_norm, x_norm.t())
    sim_mat_y = torch.mm(y_norm, y_norm.t())
    sim_diff = F.mse_loss(sim_mat_x, sim_mat_y)
    return sim_diff


def pretrain_step(CPC, Encoder, scRNA_mi_net, scRibo_mi_net, Decoder, train_dataloader, optimizer,
                  optimizer_scRNA_mi_net, optimizer_scRibo_mi_net, flag, total_step, device):
    losses = AverageMeter()
    models = [CPC, Encoder, scRNA_mi_net, scRibo_mi_net, Decoder]
    to_train_model(models, device)

    optimizer.zero_grad()
    mi_iters = 10
    train_dataloader = tqdm(train_dataloader)
    batch_stats_list = []
    for n_iter, batch_data in enumerate(train_dataloader):
        scRNA_ori_data, scRNA_raw_embedding, scRibo_ori_data, scRibo_raw_embedding = batch_data
        scRNA_ori_data = scRNA_ori_data.to(device)
        scRibo_ori_data = scRibo_ori_data.to(device)
        scRNA_raw_embedding = scRNA_raw_embedding.to(device)
        scRibo_raw_embedding = scRibo_raw_embedding.to(device)

        for _ in range(mi_iters):
            optimizer_scRNA_mi_net, optimizer_scRibo_mi_net, lld_scRNA_loss, lld_scRibo_loss = \
                Mutimodal_Decoupling_Step(
                    scRNA_raw_embedding,
                    scRibo_raw_embedding,
                    Encoder,
                    scRNA_mi_net,
                    scRibo_mi_net,
                    optimizer_scRNA_mi_net,
                    optimizer_scRibo_mi_net,
                    flag,
                    CPC,
                    device,
                )

        scRNA_embedding_loss, scRibo_embedding_loss, mi_scRNA_loss, mi_scRibo_loss, \
            cpc_loss, scRNA_recon_loss, scRibo_recon_loss, cross_loss_rna, cross_loss_ribo, \
            reg_loss, batch_stats = Mutimodal_Decoupling_second_forward(
                CPC,
                scRNA_raw_embedding,
                scRibo_raw_embedding,
                Encoder,
                scRNA_mi_net,
                scRibo_mi_net,
                Decoder,
                flag,
                scRNA_ori_data,
                scRibo_ori_data,
                device,
            )

        batch_stats_list.append(batch_stats)

        loss = (
            0.8 * cpc_loss
            + (mi_scRNA_loss + mi_scRibo_loss)
            + 0.8 * (scRNA_embedding_loss + scRibo_embedding_loss)
            + scRNA_recon_loss
            + scRibo_recon_loss
            + cross_loss_rna
            + cross_loss_ribo
            + reg_loss
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), scRNA_raw_embedding.size(0) * 10)

    avg_batch_stats = {
        "rna_util": sum(stats.get("rna_util", 0.0) for stats in batch_stats_list) / len(batch_stats_list),
        "ribo_util": sum(stats.get("ribo_util", 0.0) for stats in batch_stats_list) / len(batch_stats_list),
        "batch_util": sum(stats.get("batch_util", 0.0) for stats in batch_stats_list) / len(batch_stats_list),
    }

    global_stats = Encoder.Cross_quantizer.get_codebook_utilization()
    codebook_stats = {**avg_batch_stats, **global_stats}

    return losses.avg, n_iter + total_step, scRNA_recon_loss, scRibo_recon_loss, mi_scRNA_loss, mi_scRibo_loss, \
        cpc_loss, codebook_stats, scRNA_embedding_loss, scRibo_embedding_loss, cross_loss_rna, cross_loss_ribo


def Mutimodal_Decoupling_Step(scRNA_raw_embedding, scRibo_raw_embedding, Encoder,
                              scRNA_mi_net, scRibo_mi_net, optimizer_scRNA_mi_net, optimizer_scRibo_mi_net, flag, CPC, device):
    optimizer_scRNA_mi_net.zero_grad()
    optimizer_scRibo_mi_net.zero_grad()

    scRNA_semantic_result, scRibo_semantic_result, scRNA_encoder_result, scRibo_encoder_result, \
        scRNA_vq, scRibo_vq, scRNA_embedding_loss, scRibo_embedding_loss, Alignment_scRNA_Semantic, Alignment_scRibo_Semantic, \
        batch_stats = Encoder(scRNA_raw_embedding, scRibo_raw_embedding, flag, CPC, device)

    RtA_scRNA_semantic_result = scRNA_semantic_result.detach()
    RtA_scRibo_semantic_result = scRibo_semantic_result.detach()
    scRNA_encoder_result = scRNA_encoder_result.detach()
    scRibo_encoder_result = scRibo_encoder_result.detach()

    lld_scRNA_loss = -scRNA_mi_net.loglikeli(RtA_scRNA_semantic_result, scRNA_encoder_result)
    lld_scRNA_loss.backward()
    optimizer_scRNA_mi_net.step()

    lld_scRibo_loss = -scRibo_mi_net.loglikeli(RtA_scRibo_semantic_result, scRibo_encoder_result)
    lld_scRibo_loss.backward()
    optimizer_scRibo_mi_net.step()

    return optimizer_scRNA_mi_net, optimizer_scRibo_mi_net, lld_scRNA_loss, lld_scRibo_loss

def Mutimodal_Decoupling_second_forward(CPC, scRNA_raw_embedding, Ribo_raw_embedding, Encoder,
                                        scRNA_mi_net, Ribo_mi_net, Decoder, flag, scRNA_ori_data, Ribo_ori_data, device):

    scRNA_semantic_result, Ribo_semantic_result, scRNA_encoder_result, Ribo_encoder_result, \
        scRNA_vq, Ribo_vq, scRNA_embedding_loss, Ribo_embedding_loss, \
        Alignment_scRNA_Semantic, Alignment_Ribo_Semantic, batch_stats \
        = Encoder(scRNA_raw_embedding, Ribo_raw_embedding, flag, CPC, device)

    mi_scRNA_loss = scRNA_mi_net.mi_est(scRNA_semantic_result, scRNA_encoder_result)
    mi_Ribo_loss = Ribo_mi_net.mi_est(Ribo_semantic_result, Ribo_encoder_result)

    """Cross_CPC"""

    cpc_loss, _, _ = CPC(scRNA_semantic_result, Ribo_semantic_result, device=device)

    """Reconstruction Loss"""
    scRNA_recon_loss, Ribo_recon_loss, scRNA_recon_result, Ribo_recon_result, cross_loss_rna, cross_loss_Ribo, \
    _, _ = Decoder(scRNA_ori_data, Ribo_ori_data, scRNA_vq, Ribo_vq)

    """2025_4_12"""

    local_structure_loss = 0
    return scRNA_embedding_loss, Ribo_embedding_loss, mi_scRNA_loss, mi_Ribo_loss, \
        cpc_loss, scRNA_recon_loss, Ribo_recon_loss, cross_loss_rna, cross_loss_Ribo, local_structure_loss, batch_stats



def validate_test_epoch(CPC, Encoder, validate_test_dataloader, flag, device, valiate_type, epoch, Decoder, total_epochs=500):
    eval_models = [CPC, Encoder, Decoder]
    to_eval_model(eval_models, device)
    scRNA_semantic_tmp = torch.Tensor()
    Ribo_semantic_tmp = torch.Tensor()
    scRNA_encoder_tmp = torch.Tensor()
    Ribo_encoder_tmp = torch.Tensor()
    scRNA_vq_tmp = torch.Tensor()
    Ribo_vq_tmp = torch.Tensor()
    validate_foscttm_sum = 0
    
    total_cpc_loss = 0.0
    total_scRNA_embedding_loss = 0.0
    total_Ribo_embedding_loss = 0.0
    total_scRNA_recon_loss = 0.0
    total_Ribo_recon_loss = 0.0
    total_cross_loss_rna = 0.0
    total_cross_loss_Ribo = 0.0

    length_data = len(validate_test_dataloader)
    validate_test_dataloader = tqdm(validate_test_dataloader)

    scheduler = DynamicWeightScheduler(total_epochs)
    weights = scheduler.get_weights(epoch)

    recon_weight = 1.0
    cross_weight = 1.0
    cpc_weight = 0.5
    mi_weight = 0.5
    vq_weight = 0.5

    with torch.no_grad():
        for n_iter, batch_data in enumerate(validate_test_dataloader):

            scRNA_ori_data, scRNA_raw_embedding, Ribo_ori_data, Ribo_raw_embedding = batch_data
            
            scRNA_ori_data = scRNA_ori_data.to(device)
            Ribo_ori_data = Ribo_ori_data.to(device)
            scRNA_raw_embedding = scRNA_raw_embedding.to(device)
            Ribo_raw_embedding = Ribo_raw_embedding.to(device)

            scRNA_semantic_result, Ribo_semantic_result, scRNA_encoder_result, Ribo_encoder_result, \
                scRNA_vq, Ribo_vq, scRNA_embedding_loss, Ribo_embedding_loss, \
                Alignment_scRNA_Semantic, Alignment_Ribo_Semantic, batch_stats \
                = Encoder(scRNA_raw_embedding, Ribo_raw_embedding, flag, CPC, device)

            cpc_loss, rna_embedding_norm, Ribo_embedding_norm = CPC(
                scRNA_semantic_result, Ribo_semantic_result, device=device
            )
            validate_foscttm_sum += metrics.foscttm(
                rna_embedding_norm.detach().cpu().numpy(),
                Ribo_embedding_norm.detach().cpu().numpy(),
            )


            scRNA_recon_loss, Ribo_recon_loss, scRNA_recon_result, Ribo_recon_result, cross_loss_rna, cross_loss_Ribo, \
            _, _    = Decoder(scRNA_ori_data, Ribo_ori_data, scRNA_vq, Ribo_vq)

            total_cpc_loss += cpc_loss.item()
            total_scRNA_embedding_loss += scRNA_embedding_loss.item()
            total_Ribo_embedding_loss += Ribo_embedding_loss.item()
            total_scRNA_recon_loss += scRNA_recon_loss.item()
            total_Ribo_recon_loss += Ribo_recon_loss.item()
            total_cross_loss_rna += cross_loss_rna.item()
            total_cross_loss_Ribo += cross_loss_Ribo.item()

        avg_cpc_loss = total_cpc_loss / length_data
        avg_scRNA_embedding_loss = total_scRNA_embedding_loss / length_data
        avg_Ribo_embedding_loss = total_Ribo_embedding_loss / length_data
        avg_scRNA_recon_loss = total_scRNA_recon_loss / length_data
        avg_Ribo_recon_loss = total_Ribo_recon_loss / length_data
        avg_cross_loss_rna = total_cross_loss_rna / length_data
        avg_cross_loss_Ribo = total_cross_loss_Ribo / length_data

        loss = (cpc_weight * avg_cpc_loss + 
                vq_weight * (avg_scRNA_embedding_loss + avg_Ribo_embedding_loss) +
                recon_weight * (avg_scRNA_recon_loss + avg_Ribo_recon_loss) + 
                cross_weight * (avg_cross_loss_rna + avg_cross_loss_Ribo))
        
        avg_scRNA_recon_loss_tensor = torch.tensor(avg_scRNA_recon_loss)
        avg_Ribo_recon_loss_tensor = torch.tensor(avg_Ribo_recon_loss)
        avg_cpc_loss_tensor = torch.tensor(avg_cpc_loss)
        avg_cross_loss_rna_tensor = torch.tensor(avg_cross_loss_rna)
        avg_cross_loss_Ribo_tensor = torch.tensor(avg_cross_loss_Ribo)
        avg_scRNA_embedding_loss_tensor = torch.tensor(avg_scRNA_embedding_loss)
        avg_Ribo_embedding_loss_tensor = torch.tensor(avg_Ribo_embedding_loss)

    return loss, validate_foscttm_sum / length_data, avg_cpc_loss_tensor, avg_scRNA_recon_loss_tensor, \
        avg_Ribo_recon_loss_tensor, avg_cross_loss_rna_tensor, avg_cross_loss_Ribo_tensor, \
        avg_scRNA_embedding_loss_tensor, avg_Ribo_embedding_loss_tensor







