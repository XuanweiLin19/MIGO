from migo.model.utils import AverageMeter, Prepare_logger, save_models
import torch
import torch.nn.functional as F
import os
import logging
from migo.model.container import metricsContainer
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from migo.model.metrics import scRNA_scATAC as metrics

LOCAL_STRUCTURE_WEIGHT = 0.15
PROTO_ALIGNMENT_WEIGHT = 0.2
USAGE_REG_WEIGHT = 0.05
INTRA_CONTRASTIVE_WEIGHT = 0.12
INTRA_CONTRASTIVE_TAU = 0.2
INTRA_NOISE_SCALE = 0.08
INTRA_DROP_PROB = 0.15

def codebook_reg(scRNA_semantic_result, scATAC_semantic_result,
                 scRNA_vq, scATAC_vq, batch_stats, device):
    local_structure_loss = batch_cosine_similarity(scRNA_semantic_result, scATAC_semantic_result)
    proto_alignment_loss = compute_proto_consistency(
        scRNA_semantic_result, scATAC_semantic_result, scRNA_vq, scATAC_vq
    )
    usage_reg_loss = vq_usage_regularizer(batch_stats, device=device)
    rna_intra_contrastive_loss = intra_modal_contrastive(scRNA_semantic_result)
    atac_intra_contrastive_loss = intra_modal_contrastive(scATAC_semantic_result)
    return (
        LOCAL_STRUCTURE_WEIGHT * local_structure_loss
        + PROTO_ALIGNMENT_WEIGHT * proto_alignment_loss
        + USAGE_REG_WEIGHT * usage_reg_loss
        + INTRA_CONTRASTIVE_WEIGHT * (rna_intra_contrastive_loss + atac_intra_contrastive_loss)
    )


def to_train_model(all_model, device):
    for model in all_model:
        model.train()
        model.to(device)


def to_eval_model(all_model, device):
    for model in all_model:
        model.eval()
        model.to(device)

def pretrain_step(CPC, Encoder, scRNA_mi_net, scATAC_mi_net, Decoder, train_dataloader, optimizer,
                                         optimizer_scRNA_mi_net, optimizer_scATAC_mi_net, flag, total_step, device):

    # print("************Pretraining model***********")
    losses = AverageMeter()
    models = [CPC, Encoder, scRNA_mi_net, scATAC_mi_net, Decoder]
    to_train_model(models, device)

    optimizer.zero_grad()
    mi_iters = 10
    # logger = Prepare_logger(args, eval=args.evaluate)
    train_dataloader = tqdm(train_dataloader)
    # print('train_dataloader:', train_dataloader)
    batch_stats_list = []
    for n_iter, batch_data in enumerate(train_dataloader):

        scRNA_ori_data, scRNA_raw_embedding, scATAC_ori_data, scATAC_raw_embedding, labels = batch_data
        # print('label:', type(labels))
        scRNA_ori_data = scRNA_ori_data.to(device)
        scATAC_ori_data = scATAC_ori_data.to(device)
        scRNA_raw_embedding = scRNA_raw_embedding.to(device)
        scATAC_raw_embedding = scATAC_raw_embedding.to(device)
        labels = labels.to(device)

        for i in range(mi_iters):
            # print(type(optimizer_scATAC_mi_net))
            optimizer_scRNA_mi_net, optimizer_scATAC_mi_net, lld_scRNA_loss, lld_scATAC_loss = \
                Mutimodal_Decoupling_Step(scRNA_raw_embedding, scATAC_raw_embedding, Encoder, scRNA_mi_net,
                                          scATAC_mi_net, optimizer_scRNA_mi_net, optimizer_scATAC_mi_net, flag, CPC, device)

        #
        scRNA_embedding_loss, scATAC_embedding_loss, mi_scRNA_loss, mi_scATAC_loss, \
            cpc_loss, scRNA_recon_loss, scATAC_recon_loss, cross_loss_rna, cross_loss_atac, \
            reg_loss, batch_stats = Mutimodal_Decoupling_second_forward(
                CPC, scRNA_raw_embedding, scATAC_raw_embedding, Encoder,
                scRNA_mi_net, scATAC_mi_net, Decoder, flag, scRNA_ori_data,
                scATAC_ori_data, device)

        batch_stats_list.append(batch_stats)

        loss =  (0.8 * cpc_loss + (mi_scRNA_loss + mi_scATAC_loss) + 0.8 * (scRNA_embedding_loss + scATAC_embedding_loss)
            + scRNA_recon_loss + scATAC_recon_loss + cross_loss_rna + cross_loss_atac
            + reg_loss)


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), scRNA_raw_embedding.size(0) * 10)

    avg_batch_stats = {
        'rna_util': sum(stats.get('rna_util', 0.0) for stats in batch_stats_list) / len(batch_stats_list),
        'atac_util': sum(stats.get('atac_util', 0.0) for stats in batch_stats_list) / len(batch_stats_list),
        'batch_util': sum(stats.get('batch_util', 0.0) for stats in batch_stats_list) / len(batch_stats_list),
    }
    
    global_stats = Encoder.Cross_quantizer.get_codebook_utilization()
    
    codebook_stats = {**avg_batch_stats, **global_stats}
    
    return losses.avg, n_iter + total_step, scRNA_recon_loss, scATAC_recon_loss, mi_scRNA_loss, mi_scATAC_loss, \
        cpc_loss, codebook_stats, scRNA_embedding_loss, scATAC_embedding_loss, cross_loss_rna, cross_loss_atac

def Mutimodal_Decoupling_Step(scRNA_raw_embedding, scATAC_raw_embedding, Encoder,
                              scRNA_mi_net, scATAC_mi_net, optimizer_scRNA_mi_net, optimizer_scATAC_mi_net, flag, CPC, device):
    optimizer_scRNA_mi_net.zero_grad()
    optimizer_scATAC_mi_net.zero_grad()
    # optimizer_CPC_net.zero_grad()

    scRNA_semantic_result, scATAC_semantic_result, scRNA_encoder_result, scATAC_encoder_result, \
        scRNA_vq, scATAC_vq, scRNA_embedding_loss, scATAC_embedding_loss, Alignment_scRNA_Semantic, Alignment_scATAC_Semantic, \
    batch_stats = Encoder(scRNA_raw_embedding, scATAC_raw_embedding, flag, CPC, device)

    RtA_scRNA_semantic_result = scRNA_semantic_result.detach()
    RtA_scATAC_semantic_result = scATAC_semantic_result.detach()
    scRNA_encoder_result = scRNA_encoder_result.detach()
    scATAC_encoder_result = scATAC_encoder_result.detach()
    # scRNA_vq = scRNA_vq.detach()
    # scATAC_vq = scATAC_vq.detach()

    lld_scRNA_loss = -scRNA_mi_net.loglikeli(RtA_scRNA_semantic_result, scRNA_encoder_result)
    lld_scRNA_loss.backward()
    optimizer_scRNA_mi_net.step()

    lld_scATAC__loss = -scATAC_mi_net.loglikeli(RtA_scATAC_semantic_result, scATAC_encoder_result)
    lld_scATAC__loss.backward()
    optimizer_scATAC_mi_net.step()

    # cpc_loss.backward()
    # optimizer_CPC_net.step()
    
    return optimizer_scRNA_mi_net, optimizer_scATAC_mi_net, lld_scRNA_loss, lld_scATAC__loss

def batch_cosine_similarity(x, y):
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    sim_mat_x = torch.mm(x_norm, x_norm.t())
    sim_mat_y = torch.mm(y_norm, y_norm.t())
    return F.mse_loss(sim_mat_x, sim_mat_y)


def vq_usage_regularizer(batch_stats, device, target_util=0.65):
    rna_util = batch_stats.get("rna_util", 0.0)
    atac_util = batch_stats.get("atac_util", 0.0)
    batch_util = batch_stats.get("batch_util", None)
    rna_util_t = torch.tensor(rna_util, device=device, dtype=torch.float32)
    atac_util_t = torch.tensor(atac_util, device=device, dtype=torch.float32)
    loss = (rna_util_t - target_util) ** 2 + (atac_util_t - target_util) ** 2
    if batch_util is not None:
        batch_util_t = torch.tensor(batch_util, device=device, dtype=torch.float32)
        loss = loss + (batch_util_t - target_util) ** 2
    return loss


def compute_proto_consistency(scRNA_semantic, scATAC_semantic, scRNA_vq, scATAC_vq):
    scRNA_semantic = F.normalize(scRNA_semantic, p=2, dim=1)
    scATAC_semantic = F.normalize(scATAC_semantic, p=2, dim=1)
    scRNA_vq = F.normalize(scRNA_vq, p=2, dim=1)
    scATAC_vq = F.normalize(scATAC_vq, p=2, dim=1)
    return 0.5 * (
        F.mse_loss(scRNA_semantic, scRNA_vq) + F.mse_loss(scATAC_semantic, scATAC_vq)
    )


def stochastic_view(z, drop_prob=INTRA_DROP_PROB, noise_scale=INTRA_NOISE_SCALE):
    if drop_prob > 0:
        z = F.dropout(z, p=drop_prob, training=True)
    if noise_scale > 0:
        z = z + noise_scale * torch.randn_like(z)
    return z


def info_nce_loss(view1, view2, tau=INTRA_CONTRASTIVE_TAU):
    view1 = F.normalize(view1, p=2, dim=1)
    view2 = F.normalize(view2, p=2, dim=1)
    logits = torch.mm(view1, view2.t()) / tau
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def intra_modal_contrastive(z):
    view1 = stochastic_view(z)
    view2 = stochastic_view(z)
    return info_nce_loss(view1, view2)

def Mutimodal_Decoupling_second_forward(CPC, scRNA_raw_embedding, scATAC_raw_embedding, Encoder,
                                        scRNA_mi_net, scATAC_mi_net, Decoder, flag, scRNA_ori_data, scATAC_ori_data, device):
    # scRNA_raw_embedding = scRNA_raw_embedding.to('cuda')
    # scATAC_raw_embedding = scATAC_raw_embedding.to('cuda')
    # scRNA_ori_data = scRNA_ori_data.to('cuda')
    # scATAC_ori_data = scATAC_ori_data.to('cuda')
    
    # print(f'scRNA_raw_embedding: {scRNA_raw_embedding.size()}')
    scRNA_semantic_result, scATAC_semantic_result, scRNA_encoder_result, scATAC_encoder_result, \
        scRNA_vq, scATAC_vq, scRNA_embedding_loss, scATAC_embedding_loss, \
        Alignment_scRNA_Semantic, Alignment_scATAC_Semantic, batch_stats \
        = Encoder(scRNA_raw_embedding, scATAC_raw_embedding, flag, CPC, device)

    mi_scRNA_loss = scRNA_mi_net.mi_est(scRNA_semantic_result, scRNA_encoder_result)
    mi_scATAC_loss = scATAC_mi_net.mi_est(scATAC_semantic_result, scATAC_encoder_result)

    # print(f'scRNA_semantic_result: {scRNA_semantic_result.size()}')
    cpc_loss, _, _ = CPC(scRNA_semantic_result, scATAC_semantic_result, device=device)

    scRNA_recon_loss, scATAC_recon_loss, scRNA_recon_result, scATAC_recon_result, cross_loss_rna, cross_loss_atac, \
    _, _ = Decoder(scRNA_ori_data, scATAC_ori_data, scRNA_vq, scATAC_vq)

    reg_loss = codebook_reg(
        scRNA_semantic_result,
        scATAC_semantic_result,
        scRNA_vq,
        scATAC_vq,
        batch_stats,
        device,
    )

    return scRNA_embedding_loss, scATAC_embedding_loss, mi_scRNA_loss, mi_scATAC_loss, \
        cpc_loss, scRNA_recon_loss, scATAC_recon_loss, cross_loss_rna, cross_loss_atac, reg_loss, batch_stats
    # return scRNA_embedding_loss, scATAC_embedding_loss, mi_scRNA_loss, mi_scATAC_loss, \


def validate_test_epoch(CPC, Encoder, validate_test_dataloader, flag, device, valiate_type, epoch, Decoder):
    eval_models = [CPC, Encoder, Decoder]
    to_eval_model(eval_models, device)
    scRNA_semantic_tmp = torch.Tensor()
    scATAC_semantic_tmp = torch.Tensor()
    scRNA_encoder_tmp = torch.Tensor()
    scATAC_encoder_tmp = torch.Tensor()
    scRNA_vq_tmp = torch.Tensor()
    scATAC_vq_tmp = torch.Tensor()
    label_tmp = torch.Tensor()
    validate_foscttm_sum = 0
    
    total_cpc_loss = 0.0
    total_scRNA_embedding_loss = 0.0
    total_scATAC_embedding_loss = 0.0
    total_scRNA_recon_loss = 0.0
    total_scATAC_recon_loss = 0.0
    total_cross_loss_rna = 0.0
    total_cross_loss_atac = 0.0
    total_reg_loss = 0.0

    length_data = len(validate_test_dataloader)
    validate_test_dataloader = tqdm(validate_test_dataloader)
    with torch.no_grad():
        for n_iter, batch_data in enumerate(validate_test_dataloader):

            scRNA_ori_data, scRNA_raw_embedding, scATAC_ori_data, scATAC_raw_embedding, labels = batch_data
            scRNA_ori_data = scRNA_ori_data.to(device)
            scATAC_ori_data = scATAC_ori_data.to(device)
            scRNA_raw_embedding = scRNA_raw_embedding.to(device)
            scATAC_raw_embedding = scATAC_raw_embedding.to(device)
            labels = labels.to(device)

            scRNA_semantic_result, scATAC_semantic_result, scRNA_encoder_result, scATAC_encoder_result, \
                scRNA_vq, scATAC_vq, scRNA_embedding_loss, scATAC_embedding_loss, \
                Alignment_scRNA_Semantic, Alignment_scATAC_Semantic, batch_stats \
                = Encoder(scRNA_raw_embedding, scATAC_raw_embedding, flag, CPC, device)

            cpc_loss, rna_embedding_norm, atac_embedding_norm = CPC(
                scRNA_semantic_result, scATAC_semantic_result, device=device
            )
            validate_foscttm_sum += metrics.foscttm(
                rna_embedding_norm.detach().cpu().numpy(),
                atac_embedding_norm.detach().cpu().numpy(),
            )

            scRNA_recon_loss, scATAC_recon_loss, scRNA_recon_result, scATAC_recon_result, cross_loss_rna, cross_loss_atac, \
            _, _    = Decoder(scRNA_ori_data, scATAC_ori_data, scRNA_vq, scATAC_vq)

            reg_loss = codebook_reg(
                scRNA_semantic_result,
                scATAC_semantic_result,
                scRNA_vq,
                scATAC_vq,
                batch_stats,
                device,
            )

            total_cpc_loss += cpc_loss.item()
            total_scRNA_embedding_loss += scRNA_embedding_loss.item()
            total_scATAC_embedding_loss += scATAC_embedding_loss.item()
            total_scRNA_recon_loss += scRNA_recon_loss.item()
            total_scATAC_recon_loss += scATAC_recon_loss.item()
            total_cross_loss_rna += cross_loss_rna.item()
            total_cross_loss_atac += cross_loss_atac.item()
            total_reg_loss += reg_loss.item()

        avg_cpc_loss = total_cpc_loss / length_data
        avg_scRNA_embedding_loss = total_scRNA_embedding_loss / length_data
        avg_scATAC_embedding_loss = total_scATAC_embedding_loss / length_data
        avg_scRNA_recon_loss = total_scRNA_recon_loss / length_data
        avg_scATAC_recon_loss = total_scATAC_recon_loss / length_data
        avg_cross_loss_rna = total_cross_loss_rna / length_data
        avg_cross_loss_atac = total_cross_loss_atac / length_data
        avg_reg_loss = total_reg_loss / length_data

        loss =  (0.8 * avg_cpc_loss + 0.8 * (avg_scRNA_embedding_loss + avg_scATAC_embedding_loss)
         + avg_scRNA_recon_loss + avg_scATAC_recon_loss + avg_cross_loss_rna + avg_cross_loss_atac
         + avg_reg_loss)
        
        avg_scRNA_recon_loss_tensor = torch.tensor(avg_scRNA_recon_loss)
        avg_scATAC_recon_loss_tensor = torch.tensor(avg_scATAC_recon_loss)
        avg_cpc_loss_tensor = torch.tensor(avg_cpc_loss)
        avg_scRNA_embedding_loss_tensor = torch.tensor(avg_scRNA_embedding_loss)
        avg_scATAC_embedding_loss_tensor = torch.tensor(avg_scATAC_embedding_loss)

    return (
        loss,
        validate_foscttm_sum / length_data,
        avg_cpc_loss_tensor,
        avg_scRNA_recon_loss_tensor,
        avg_scATAC_recon_loss_tensor,
        avg_scRNA_embedding_loss_tensor,
        avg_scATAC_embedding_loss_tensor,
        torch.tensor(avg_cross_loss_rna),
        torch.tensor(avg_cross_loss_atac),
    )





