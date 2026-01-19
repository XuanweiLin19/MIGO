#!/usr/bin/env python

import torch.nn.functional as F
import random
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Module, MultiheadAttention, Dropout, Linear, LayerNorm
from migo.model.utils import _get_clones, _get_activation_fn

# ---------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1-pt)**self.gamma * BCE_loss
        return loss.mean()
    
class scRNA_Encoder(nn.Module):
    def __init__(self, scRNA_dim, hidden_dim):#(256, 64)
        super(scRNA_Encoder, self).__init__()

        self.scRNA_dim = scRNA_dim
        self.hidden_dim = hidden_dim
        self.scRNA_linear = nn.Linear(scRNA_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.scRNA_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

    def forward(self, scRNA_feat):
        scRNA_feat = self.relu(self.scRNA_linear(scRNA_feat))
        out =  self.relu2(self.scRNA_linear2(scRNA_feat))
        return out #(N, 128)

class Ribo_Encoder(nn.Module):
    def __init__(self, ribo_dim, hidden_dim, dropout=0.1):
        super(Ribo_Encoder, self).__init__()
        
        self.ribo_linear = nn.Sequential(
            nn.Linear(ribo_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, ribo_feat):
        return self.ribo_linear(ribo_feat)

class Ribo_Reconstruction_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        
        
        self.decoder_net = nn.Sequential(
            nn.Linear(vq_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(input_dim, output_dim)
        )
        
        self.log_var = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, z_vq):
        reconstruction = self.decoder_net(z_vq)
        return reconstruction, self.log_var.expand(reconstruction.size(0), -1)

    def reconstruct(self, z_vq, add_noise=False):
        reconstruction, log_var = self.forward(z_vq)
        
        if add_noise:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std) * 0.1
            return reconstruction + eps
        
        return reconstruction

class Ribo_ReconstructionLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, predicted, target, log_var=None):
        mse_loss = self.mse_loss(predicted, target)
        l1_loss = self.l1_loss(predicted, target)
        
        recon_loss = self.alpha * mse_loss + self.beta * l1_loss
        
        if log_var is not None:
            var_penalty = torch.mean(torch.exp(log_var))
            recon_loss += 0.01 * var_penalty
            
        return recon_loss

class scRNA_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(vq_dim, input_dim), 
            # nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
            # nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, z_vq): 
        return self.fc(z_vq)


class RA_VQVAE_Encoder(nn.Module):# (256, 256, 128, 128, 128, 400)
    def __init__(self, scRNA_dim, ribo_dim, scRNA_output_dim, ribo_output_dim, embedding_dim, n_embed=400):
        super(RA_VQVAE_Encoder, self).__init__()
        self.scRNA_dim = scRNA_dim
        self.ribo_dim = ribo_dim
        self.h_dim = embedding_dim
        
        self.scRNA_encoder = scRNA_Encoder(scRNA_dim, scRNA_output_dim)
        self.ribo_encoder = Ribo_Encoder(ribo_dim, ribo_output_dim)
        self.Cross_quantizer = Cross_VQ_RA(n_embed, self.h_dim)
        
        self.scRNA_self_att = InternalRelationModule(input_dim=self.scRNA_dim, d_model=self.h_dim)
        self.ribo_self_att = InternalRelationModule(input_dim=self.ribo_dim, d_model=self.h_dim)

    def forward(self, scRNA_feature, ribo_feature, flag, CPC, device):
        scRNA_semantic_result = self.scRNA_self_att(scRNA_feature)
        ribo_semantic_result = self.ribo_self_att(ribo_feature)

        scRNA_encoder_result = self.scRNA_encoder(scRNA_feature)
        ribo_encoder_result = self.ribo_encoder(ribo_feature)

        _, Alignment_scRNA_Semantic, Alignment_Ribo_Semantic = CPC(
            scRNA_semantic_result, ribo_semantic_result, device=device
        )

        scRNA_quantized, ribo_quantized, scRNA_loss, ribo_loss, scRNA_perplexity, ribo_perplexity, \
        batch_stats = self.Cross_quantizer(Alignment_scRNA_Semantic, Alignment_Ribo_Semantic, flag)

        return scRNA_semantic_result, ribo_semantic_result, scRNA_encoder_result, ribo_encoder_result, \
            scRNA_quantized, ribo_quantized, scRNA_loss, ribo_loss, \
            Alignment_scRNA_Semantic, Alignment_Ribo_Semantic, batch_stats
    
class InternalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model):
        super(InternalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)
        return feature

class Encoder(Module):
    """Args:
        encoder_layer: an instance of the EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):

        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output)

        if self.norm:
            output = self.norm(output)

        return output

class Ribo_GaussianLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, model_output, target, beta=0.001):

        mean, log_var = model_output
        target_log = torch.log1p(target)
        
        nll = 0.5 * torch.exp(-log_var) * (target_log - mean)**2
        nll += 0.5 * log_var
        nll += 0.5 * torch.log(torch.tensor(2 * torch.pi))
        
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)

        return torch.mean(nll) + beta * torch.mean(kl_div)


class EncoderLayer(Module):
    """Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Cross_VQ_RA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(Cross_VQ_RA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._commitment_cost = commitment_cost
        
        self.register_buffer("usage_count", torch.zeros(num_embeddings))
        self.register_buffer("rna_usage", torch.zeros(num_embeddings))
        self.register_buffer("ribo_usage", torch.zeros(num_embeddings))
        self.register_buffer("batch_count", torch.tensor(0.0))
        self.register_buffer("epoch_rna_usage", torch.zeros(num_embeddings))
        self.register_buffer("epoch_ribo_usage", torch.zeros(num_embeddings))
        
        nn.init.normal_(self._embedding.weight, mean=0, std=self._embedding_dim ** -0.5)

        for p in self._embedding.parameters():
            p.requires_grad = False

        self.embedding_proj = nn.Linear(self._embedding_dim, self._embedding_dim)

    def forward(self, scRNA_semantic, ribo_semantic, flag):
        scR_flat = scRNA_semantic.detach()
        scRibo_flat = ribo_semantic.detach()

        quant_codebook = self.embedding_proj(self._embedding.weight)
        
        scR_distances = (torch.sum(scR_flat ** 2, dim=1, keepdim=True) + 
                        torch.sum(quant_codebook ** 2, dim=1) -
                        2 * torch.matmul(scR_flat, quant_codebook.t()))
        
        scRibo_distances = (torch.sum(scRibo_flat ** 2, dim=1, keepdim=True) + 
                           torch.sum(quant_codebook ** 2, dim=1) -
                           2 * torch.matmul(scRibo_flat, quant_codebook.t()))

        scRNA_encoding_indices = torch.argmin(scR_distances, dim=1).unsqueeze(1)
        scRNA_encodings = torch.zeros(scRNA_encoding_indices.shape[0], self._num_embeddings,
                                      device=scR_distances.device).double()
        scRNA_encodings.scatter_(1, scRNA_encoding_indices, 1)
        scRNA_quantized = torch.matmul(scRNA_encodings.double(), quant_codebook).view(scRNA_semantic.shape)

        ribo_encoding_indices = torch.argmin(scRibo_distances, dim=1).unsqueeze(1)
        ribo_encodings = torch.zeros(ribo_encoding_indices.shape[0], self._num_embeddings,
                                    device=scR_distances.device).double()
        ribo_encodings.scatter_(1, ribo_encoding_indices, 1)
        ribo_quantized = torch.matmul(ribo_encodings.double(), quant_codebook).view(ribo_semantic.shape)

        with torch.no_grad():
            scRNA_indices = scRNA_encoding_indices.squeeze(-1)
            ribo_indices = ribo_encoding_indices.squeeze(-1)
            
            self.batch_count += 1
            
            rna_mask = torch.zeros(self._num_embeddings, device=scRNA_indices.device, dtype=torch.bool)
            ribo_mask = torch.zeros(self._num_embeddings, device=ribo_indices.device, dtype=torch.bool)
            
            rna_mask[scRNA_indices] = True
            ribo_mask[ribo_indices] = True
            
            self.rna_usage += rna_mask.float()
            self.ribo_usage += ribo_mask.float()
            self.usage_count += (rna_mask | ribo_mask).float()
            
            self.epoch_rna_usage += rna_mask.float()
            self.epoch_ribo_usage += ribo_mask.float()

        scRNA_e_latent_loss = F.mse_loss(scRNA_semantic, scRNA_quantized.detach())
        scRNA_loss = 2 * self._commitment_cost * scRNA_e_latent_loss

        ribo_scRNA_vq_forward_loss = ((F.mse_loss(ribo_quantized, ribo_semantic.detach())
                                      + F.mse_loss(scRNA_quantized, scRNA_semantic.detach())
                                      + 0.5 * F.mse_loss(ribo_quantized, scRNA_semantic.detach()))
                                     + 0.5 * F.mse_loss(scRNA_quantized, ribo_semantic.detach()))

        ribo_e_latent_loss = F.mse_loss(ribo_semantic, ribo_quantized.detach())
        ribo_loss = 2 * self._commitment_cost * ribo_e_latent_loss + self._commitment_cost * ribo_scRNA_vq_forward_loss

        scRNA_quantized = scRNA_semantic + (scRNA_quantized - scRNA_semantic).detach()
        ribo_quantized = ribo_semantic + (ribo_quantized - ribo_semantic).detach()

        scRNA_avg_probs = torch.mean(scRNA_encodings, dim=0)
        scRNA_perplexity = torch.exp(-torch.sum(scRNA_avg_probs * torch.log(scRNA_avg_probs + 1e-10)))
        ribo_avg_probs = torch.mean(ribo_encodings, dim=0)
        ribo_perplexity = torch.exp(-torch.sum(ribo_avg_probs * torch.log(ribo_avg_probs + 1e-10)))

        batch_stats = self.get_batch_stats(scRNA_indices, ribo_indices)
        
        return scRNA_quantized, ribo_quantized, scRNA_loss, ribo_loss, scRNA_perplexity, ribo_perplexity, batch_stats
    
    def get_batch_stats(self, scRNA_indices, ribo_indices):
        if self.batch_count == 0:
            return {
                'global_util': 0.0,
                'rna_util': 0.0,
                'ribo_util': 0.0,
                'cross_util': 0.0,
                'active_codes': 0
            }
        
        global_util = (self.usage_count > 0).float().sum() / self._num_embeddings
        rna_util = (self.rna_usage > 0).float().sum() / self._num_embeddings
        ribo_util = (self.ribo_usage > 0).float().sum() / self._num_embeddings
        cross_util = ((self.rna_usage > 0) & (self.ribo_usage > 0)).float().sum() / self._num_embeddings
        batch_util = torch.unique(torch.cat([scRNA_indices, ribo_indices])).size(0) / self._num_embeddings
        active_codes = (self.usage_count > 0).float().sum().item()
        
        return {
            'global_util': float(global_util.item()),
            'rna_util': float(rna_util.item()),
            'ribo_util': float(ribo_util.item()),
            'cross_util': float(cross_util.item()),
            'batch_util': float(batch_util),
            'active_codes': int(active_codes)
        }

    def get_codebook_utilization(self):
        if self.batch_count == 0:
            return {
                'global_util': 0.0,
                'rna_util': 0.0,
                'ribo_util': 0.0,
                'cross_util': 0.0,
                'active_codes': 0
            }
        global_util = (self.usage_count > 0).float().sum() / self._num_embeddings
        rna_util = (self.rna_usage > 0).float().sum() / self._num_embeddings
        ribo_util = (self.ribo_usage > 0).float().sum() / self._num_embeddings
        cross_util = ((self.rna_usage > 0) & (self.ribo_usage > 0)).float().sum() / self._num_embeddings
        active_codes = (self.usage_count > 0).float().sum().item()
        return {
            'global_util': float(global_util.item()),
            'rna_util': float(rna_util.item()),
            'ribo_util': float(ribo_util.item()),
            'cross_util': float(cross_util.item()),
            'active_codes': int(active_codes)
        }
    
    def reset_epoch_stats(self):
        epoch_rna_util = (self.epoch_rna_usage > 0).float().sum() / self._num_embeddings
        epoch_ribo_util = (self.epoch_ribo_usage > 0).float().sum() / self._num_embeddings
        epoch_cross_util = ((self.epoch_rna_usage > 0) & (self.epoch_ribo_usage > 0)).float().sum() / self._num_embeddings
        
        return {
            'epoch_rna_util': float(epoch_rna_util.item()),
            'epoch_ribo_util': float(epoch_ribo_util.item()),
            'epoch_cross_util': float(epoch_cross_util.item())
        }

class RA_VQVAE_Decoder(nn.Module):
    def __init__(self, scRNA_dim, ribo_dim, scRNA_output_dim, ribo_output_dim):
        super(RA_VQVAE_Decoder, self).__init__()
        self.hidden_dim = 64
        
        self.scRNA_decoder = scRNA_Decoder(scRNA_dim, scRNA_output_dim, self.hidden_dim)
        self.ribo_decoder = Ribo_Reconstruction_Decoder(ribo_dim, ribo_output_dim, self.hidden_dim)

        self.rna_loss = nn.MSELoss()
        self.ribo_loss = Ribo_ReconstructionLoss()

    def forward(self, scRNA_ori, ribo_ori, scRNA_vq, ribo_vq):
        scRNA_recon_result = self.scRNA_decoder(scRNA_vq)
        ribo_recon_result, ribo_log_var = self.ribo_decoder(ribo_vq)

        cross_scRNA_result = self.scRNA_decoder(ribo_vq)
        cross_ribo_result, cross_ribo_log_var = self.ribo_decoder(scRNA_vq)

        scRNA_recon_loss = self.rna_loss(scRNA_recon_result, scRNA_ori)
        ribo_recon_loss = self.ribo_loss(ribo_recon_result, ribo_ori, ribo_log_var)

        cross_loss_rna = self.rna_loss(cross_scRNA_result, scRNA_ori)
        cross_loss_ribo = self.ribo_loss(cross_ribo_result, ribo_ori, cross_ribo_log_var)

        return scRNA_recon_loss, ribo_recon_loss, scRNA_recon_result, \
               ribo_recon_result, cross_loss_rna, cross_loss_ribo, \
               cross_scRNA_result, cross_ribo_result





