#!/usr/bin/env python

import torch.nn.functional as F
import random
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Module, MultiheadAttention, Dropout, Linear, LayerNorm
from migo.model.utils import _get_clones, _get_activation_fn
from torch.distributions import NegativeBinomial
import math
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

class scATAC_Encoder(nn.Module):
    def __init__(self, scATAC_dim, hidden_dim):
        super(scATAC_Encoder, self).__init__()
        self.scATAC_dim = scATAC_dim
        self.hidden_dim = hidden_dim
        self.scATAC_linear = nn.Linear(scATAC_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.scATAC_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

    def forward(self, scATAC_feat):
        scATAC_feat = self.relu1(self.scATAC_linear(scATAC_feat))
        out = self.relu2(self.scATAC_linear2(scATAC_feat))
        return out



class NegativeBinomialLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, mu, theta, y_true):
        mu = torch.clamp(mu, min=self.eps, max=1e6)
        theta = torch.clamp(theta, min=self.eps, max=1e3)

        
        log_theta_mu_theta = theta * torch.log(theta + self.eps) - (y_true + theta) * torch.log(mu + theta + self.eps)
        log_y_term = y_true * torch.log(mu + self.eps)
        
        log_gamma_term = torch.lgamma(y_true + theta + self.eps) - torch.lgamma(theta + self.eps) - torch.lgamma(y_true + 1.0 + self.eps)
        
        log_prob = log_gamma_term + log_theta_mu_theta + log_y_term
        
        return -log_prob.mean()


class GaussianLogLikelihoodLoss(nn.Module):

    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, mean, logvar, target):
        var = torch.exp(torch.clamp(logvar, min=-10.0, max=10.0)) + self.eps
        loss = 0.5 * (torch.log(var + self.eps) + (target - mean) ** 2 / (var + self.eps))
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


class ZeroInflatedGaussianLoss(nn.Module):
    def __init__(self, zero_threshold=1e-6, eps=1e-8):
        super().__init__()
        self.zero_threshold = zero_threshold
        self.eps = eps
        self.log_two_pi = math.log(2 * math.pi)

    def forward(self, mean, logvar, logit_pi, target):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        var = torch.exp(logvar) + self.eps
        log_prob_normal = -0.5 * (self.log_two_pi + logvar + (target - mean) ** 2 / (var + self.eps))

        log_pi = -F.softplus(-logit_pi)
        log_one_minus_pi = -F.softplus(logit_pi)

        zero_mask = (target <= self.zero_threshold).float()
        log_prob_zero = torch.logaddexp(log_pi, log_one_minus_pi + log_prob_normal)
        log_prob_nonzero = log_one_minus_pi + log_prob_normal

        log_prob = zero_mask * log_prob_zero + (1 - zero_mask) * log_prob_nonzero
        return -log_prob.mean()



class scRNA_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim, hidden_dim=512, distribution="nb"):
        super().__init__()
        self.distribution = distribution.lower()
        assert self.distribution in {"nb", "gaussian", "zin"}, "distribution must be 'nb', 'gaussian', or 'zin'"

        self.decoder_network = nn.Sequential(
            nn.Linear(vq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        if self.distribution == "nb":
            self.scale_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, output_dim)
            )
        else:
            self.logvar_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, output_dim)
            )
            if self.distribution == "zin":
                self.zi_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, output_dim)
                )

    def forward(self, z_vq):
        x = self.decoder_network(z_vq)

        if self.distribution == "nb":
            mu = F.softplus(self.mu_head(x)) + 1e-6
            theta = F.softplus(self.scale_head(x)) + 1e-4
            mu = torch.clamp(mu, min=1e-6, max=1e6)
            theta = torch.clamp(theta, min=1e-4, max=1e3)
            return mu, theta

        mean = self.mu_head(x)
        mean = torch.clamp(mean, min=-10.0, max=12.0)
        logvar = self.logvar_head(x)
        logvar = torch.clamp(logvar, min=-6.0, max=4.0)

        if self.distribution == "gaussian":
            return mean, logvar

        logit_pi = self.zi_head(x)
        return mean, logvar, logit_pi

class scATAC_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(vq_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim), 
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z_vq):
        return self.fc(z_vq)

class RA_VQVAE_Encoder(nn.Module):
    def __init__(self, scRNA_dim, scATAC_dim, scRNA_output_dim, scATAC_output_dim, embedding_dim, n_embed=400):
        super(RA_VQVAE_Encoder, self).__init__()
        self.scRNA_dim = scRNA_dim
        self.scATAC_dim = scATAC_dim 
        self.h_dim = embedding_dim


        self.scRNA_semantic_encoder = InternalRelationModule(input_dim=scRNA_dim, d_model=embedding_dim)
        self.scATAC_semantic_encoder = InternalRelationModule(input_dim=scATAC_dim, d_model=embedding_dim)
        
        self.scRNA_specific_encoder = scRNA_Encoder(scRNA_dim, scRNA_output_dim)
        self.scATAC_specific_encoder = scATAC_Encoder(scATAC_dim, scATAC_output_dim)
        
        self.Cross_quantizer = Cross_VQ_RA(n_embed, self.h_dim)


    def forward(self, scRNA_feature, scATAC_feature, flag, CPC, device):

        scRNA_semantic_result = self.scRNA_semantic_encoder(scRNA_feature)
        scATAC_semantic_result = self.scATAC_semantic_encoder(scATAC_feature)

        scRNA_specific_result = self.scRNA_specific_encoder(scRNA_feature)
        scATAC_specific_result = self.scATAC_specific_encoder(scATAC_feature)
        
        _, Alignment_scRNA_Semantic, Alignment_scATAC_Semantic = CPC(
            scRNA_semantic_result, scATAC_semantic_result, device=device
        )
        
        scRNA_quantized, scATAC_quantized, scRNA_loss, scATAC_loss, scRNA_perplexity, scATAC_perplexity, batch_stats = \
            self.Cross_quantizer(Alignment_scRNA_Semantic, Alignment_scATAC_Semantic, flag)

        return (scRNA_semantic_result, scATAC_semantic_result, 
                scRNA_specific_result, scATAC_specific_result,
                scRNA_quantized, scATAC_quantized, 
                scRNA_loss, scATAC_loss, 
                Alignment_scRNA_Semantic, Alignment_scATAC_Semantic, 
                batch_stats)
    
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
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25,
                 ema_decay=0.99, ema_eps=1e-5, softmax_temperature=0.5):
        super(Cross_VQ_RA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.softmax_temperature = softmax_temperature
        
        self.register_buffer("unactivated_count", -torch.ones(num_embeddings)) #unactivated:-1
        self.register_buffer("usage_count", torch.zeros(num_embeddings))
        self.register_buffer("rna_usage", torch.zeros(num_embeddings))
        self.register_buffer("atac_usage", torch.zeros(num_embeddings))
        self.register_buffer("batch_count", torch.tensor(0))
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.zeros(num_embeddings, embedding_dim))
        
        nn.init.normal_(self._embedding.weight, mean=0, std=self._embedding_dim ** -0.5)

        for p in self._embedding.parameters():
            p.requires_grad = False

        self.embedding_proj = nn.Linear(self._embedding_dim, self._embedding_dim)

    def scRNA_vq_embedding(self, scRNA_semantic):
        quant_codebook = self.embedding_proj(self._embedding.weight)
        distances = (torch.sum(scRNA_semantic ** 2, dim=1, keepdim=True) + torch.sum(quant_codebook ** 2, dim=1)
                     - 2 * torch.matmul(scRNA_semantic, quant_codebook.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=scRNA_semantic.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, quant_codebook).view(scRNA_semantic.shape)
        scRNA_quantized = scRNA_semantic + (quantized - scRNA_semantic).detach()
        return scRNA_quantized

    def scATAC_vq_embedding(self, scATAC_semantic):
        quant_codebook = self.embedding_proj(self._embedding.weight)
        distances = (
                    torch.sum(scATAC_semantic ** 2, dim=1, keepdim=True) + torch.sum(quant_codebook ** 2, dim=1)
                    - 2 * torch.matmul(scATAC_semantic, quant_codebook.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=scATAC_semantic.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, quant_codebook).view(scATAC_semantic.shape)
        scATAC_quantized = scATAC_semantic + (quantized - scATAC_semantic).detach()
        return scATAC_quantized

    def forward(self, scRNA_semantic, scATAC_semantic, flag):

        # B, D = scRNA_semantic.size()

        scR_flat = scRNA_semantic.detach()
        scA_flat = scATAC_semantic.detach()

        quant_codebook = self.embedding_proj(self._embedding.weight).to(scRNA_semantic.dtype)
        # print('The weight of self.embedding_proj: ', self.embedding_proj.weight)
        scR_distances = (torch.sum(scR_flat ** 2, dim=1, keepdim=True) + torch.sum(quant_codebook ** 2, dim=1)
                         - 2 * torch.matmul(scR_flat, quant_codebook.t()))
        scA_distances = (torch.sum(scA_flat ** 2, dim=1, keepdim=True) + torch.sum(quant_codebook ** 2, dim=1)
                         - 2 * torch.matmul(scA_flat, quant_codebook.t()))

        temperature = max(self.softmax_temperature, 1e-3)
        scRNA_soft_assign = torch.softmax(-scR_distances / temperature, dim=1)
        scATAC_soft_assign = torch.softmax(-scA_distances / temperature, dim=1)

        scRNA_encoding_indices = torch.argmin(scR_distances, dim=1).unsqueeze(1)
        scRNA_encodings = torch.zeros(scRNA_encoding_indices.shape[0], self._num_embeddings,
                                      device=scR_distances.device, dtype=scRNA_semantic.dtype)
        scRNA_encodings.scatter_(1, scRNA_encoding_indices, 1)
        scRNA_quantized_hard = torch.matmul(scRNA_encodings, quant_codebook).view(scRNA_semantic.shape)

        scATAC_encoding_indices = torch.argmin(scA_distances, dim=1).unsqueeze(1)
        scATAC_encodings = torch.zeros(scATAC_encoding_indices.shape[0], self._num_embeddings,
                                       device=scR_distances.device, dtype=scATAC_semantic.dtype)
        scATAC_encodings.scatter_(1, scATAC_encoding_indices, 1)
        scATAC_quantized_hard = torch.matmul(scATAC_encodings, quant_codebook).view(scATAC_semantic.shape)

        scRNA_quantized_soft = torch.matmul(scRNA_soft_assign, quant_codebook).view(scRNA_semantic.shape)
        scATAC_quantized_soft = torch.matmul(scATAC_soft_assign, quant_codebook).view(scATAC_semantic.shape)


        scRNA_indices = scRNA_encoding_indices.squeeze(-1)  # [B, 1] -> [B]
        scATAC_indices = scATAC_encoding_indices.squeeze(-1)
        
        all_indices = torch.cat([scRNA_indices, scATAC_indices])
        
        with torch.no_grad():
            rna_counts = torch.bincount(scRNA_indices, minlength=self._num_embeddings)
            atac_counts = torch.bincount(scATAC_indices, minlength=self._num_embeddings)
            total_counts = torch.bincount(all_indices, minlength=self._num_embeddings)
            
            self.usage_count += (total_counts > 0).float()
            self.rna_usage += (rna_counts > 0).float()
            self.atac_usage += (atac_counts > 0).float()
            
            self.unactivated_count = torch.where(
                total_counts > 0,
                torch.zeros_like(self.unactivated_count),
                self.unactivated_count
            )
            
            self.batch_count += 1
            self._update_codebook_ema(scR_flat, scA_flat, scRNA_encodings, scATAC_encodings)
            

        scRNA_e_latent_loss = F.mse_loss(scRNA_semantic, scRNA_quantized_hard.detach())
        scRNA_loss = 2 * self._commitment_cost * scRNA_e_latent_loss

        scATAC_scRNA_vq_forward_loss = ((F.mse_loss(scATAC_quantized_hard, scATAC_semantic.detach())
                                         + F.mse_loss(scRNA_quantized_hard, scRNA_semantic.detach())
                                        + 0.5 * F.mse_loss(scATAC_quantized_hard, scRNA_semantic.detach()))
                                        + 0.5 * F.mse_loss(scRNA_quantized_hard, scATAC_semantic.detach()))

        scATAC_e_latent_loss = F.mse_loss(scATAC_semantic, scATAC_quantized_hard.detach())
        scATAC_loss = 2 * self._commitment_cost * scATAC_e_latent_loss + self._commitment_cost * scATAC_scRNA_vq_forward_loss

        scRNA_quantized = scRNA_semantic + (scRNA_quantized_soft - scRNA_semantic).detach()
        scATAC_quantized = scATAC_semantic + (scATAC_quantized_soft - scATAC_semantic).detach()

        scRNA_avg_probs = torch.mean(scRNA_encodings, dim=0)
        scRNA_perplexity = torch.exp(-torch.sum(scRNA_avg_probs * torch.log(scRNA_avg_probs + 1e-10)))
        scATAC_avg_probs = torch.mean(scATAC_encodings, dim=0)
        scATAC_perplexity = torch.exp(-torch.sum(scATAC_avg_probs * torch.log(scATAC_avg_probs + 1e-10)))

        batch_stats = self.get_batch_stats(scRNA_indices, scATAC_indices)
        
        return scRNA_quantized, scATAC_quantized, scRNA_loss, scATAC_loss, scRNA_perplexity, scATAC_perplexity, batch_stats
    
    def _update_codebook_ema(self, scR_flat, scA_flat, scRNA_encodings, scATAC_encodings):
        if not self.training:
            return

        with torch.no_grad():
            combined_encodings = torch.cat([scRNA_encodings, scATAC_encodings], dim=0)
            combined_vectors = torch.cat([scR_flat, scA_flat], dim=0).to(self.ema_w.dtype)
            combined_encodings = combined_encodings.to(self.ema_cluster_size.dtype)

            cluster_size = torch.sum(combined_encodings, dim=0)
            embed_sum = torch.matmul(combined_encodings.t(), combined_vectors)

            self.ema_cluster_size.mul_(self.ema_decay).add_((1 - self.ema_decay) * cluster_size)
            self.ema_w.mul_(self.ema_decay).add_((1 - self.ema_decay) * embed_sum)

            denom = torch.clamp(self.ema_cluster_size, min=self.ema_eps).unsqueeze(1)
            normalized_embed = self.ema_w / denom
            self._embedding.weight.data.copy_(normalized_embed)

    def get_batch_stats(self, scRNA_indices, scATAC_indices):
        if self.batch_count == 0:
            return {
                'global_util': 0.0,
                'rna_util': 0.0,
                'atac_util': 0.0,
                'cross_util': 0.0,
                'active_codes': 0
            }
        
        global_util = (self.usage_count > 0).float().sum() / self._num_embeddings
        rna_util = (self.rna_usage > 0).float().sum() / self._num_embeddings
        atac_util = (self.atac_usage > 0).float().sum() / self._num_embeddings
        
        cross_util = ((self.rna_usage > 0) & (self.atac_usage > 0)).float().sum() / self._num_embeddings
        batch_util = torch.unique(torch.cat([scRNA_indices, scATAC_indices])).size(0) / self._num_embeddings
        active_codes = (self.usage_count > 0).float().sum().item()

        return {
            'global_util': global_util.item(),
            'rna_util': rna_util.item(),
            'atac_util': atac_util.item(),
            'cross_util': cross_util.item(),
            'batch_util': float(batch_util),
            'active_codes': active_codes
        }

    def get_codebook_utilization(self):
        if self.batch_count == 0:
            return {
                'global_util': 0.0,
                'rna_util': 0.0,
                'atac_util': 0.0,
                'cross_util': 0.0,
                'active_codes': 0
            }
        global_util = (self.usage_count > 0).float().sum() / self._num_embeddings
        rna_util = (self.rna_usage > 0).float().sum() / self._num_embeddings
        atac_util = (self.atac_usage > 0).float().sum() / self._num_embeddings
        cross_util = ((self.rna_usage > 0) & (self.atac_usage > 0)).float().sum() / self._num_embeddings
        active_codes = (self.usage_count > 0).float().sum().item()
        return {
            'global_util': global_util.item(),
            'rna_util': rna_util.item(),
            'atac_util': atac_util.item(),
            'cross_util': cross_util.item(),
            'active_codes': active_codes
        }


class RA_VQVAE_Decoder(nn.Module):
    def __init__(self, scRNA_dim, scATAC_dim, scRNA_output_dim, scATAC_output_dim,
                 rna_distribution="nb", rna_recon_objective="auto"):
        super(RA_VQVAE_Decoder, self).__init__()
        self.hidden_dim = 64 # embedding_dim
        self.rna_distribution = rna_distribution.lower()
        self.rna_recon_objective = self._resolve_rna_objective(rna_recon_objective)

        self.scRNA_decoder = scRNA_Decoder(
            input_dim=scRNA_dim,
            output_dim=scRNA_output_dim,
            vq_dim=self.hidden_dim,
            distribution=self.rna_distribution
        )
        self.scATAC_decoder = scATAC_Decoder(scATAC_dim, scATAC_output_dim, self.hidden_dim)

        self.nb_loss = None
        self.gaussian_loss = None
        self.zin_loss = None

        if self.rna_distribution == "nb":
            self.nb_loss = NegativeBinomialLoss()
        elif self.rna_distribution == "gaussian":
            self.gaussian_loss = GaussianLogLikelihoodLoss()
        elif self.rna_distribution == "zin":
            self.zin_loss = ZeroInflatedGaussianLoss()
        else:
            raise ValueError("rna_distribution must be 'nb', 'gaussian', or 'zin'")

        self.mse_loss = nn.MSELoss()

        self.foc_loss = FocalLoss()

    def _resolve_rna_objective(self, objective):
        if objective is None:
            return "nll"
        objective = objective.lower()
        if objective == "auto":
            if self.rna_distribution in {"nb", "zin"}:
                return "nll"
            return "mse"
        if objective not in {"nll", "mse"}:
            raise ValueError("rna_recon_objective must be 'auto', 'nll', or 'mse'")
        if self.rna_distribution in {"nb", "zin"} and objective != "nll":
            raise ValueError("Negative Binomial and ZIN decoders only support 'nll' objective")
        return objective

    def _decode_rna(self, latent):
        outputs = self.scRNA_decoder(latent)
        if self.rna_distribution == "nb":
            mu, theta = outputs
            return {"mean": mu, "scale": theta}
        if self.rna_distribution == "gaussian":
            mean, logvar = outputs
            return {"mean": mean, "logvar": logvar}
        mean, logvar, logit_pi = outputs
        return {"mean": mean, "logvar": logvar, "logit_pi": logit_pi}

    def _rna_loss(self, outputs, target):
        if self.rna_distribution == "nb":
            return self.nb_loss(outputs["mean"], outputs["scale"], target)
        if self.rna_distribution == "gaussian":
            if self.rna_recon_objective == "mse":
                return self.mse_loss(outputs["mean"], target)
            return self.gaussian_loss(outputs["mean"], outputs["logvar"], target)
        return self.zin_loss(outputs["mean"], outputs["logvar"], outputs["logit_pi"], target)

    def forward(self, scRNA_ori, scATAC_ori, scRNA_vq, scATAC_vq):
        rna_self = self._decode_rna(scRNA_vq)
        scATAC_recon_result = self.scATAC_decoder(scATAC_vq)

        rna_cross = self._decode_rna(scATAC_vq)
        cross_scATAC = self.scATAC_decoder(scRNA_vq)

        scRNA_recon_loss = self._rna_loss(rna_self, scRNA_ori)
        scATAC_recon_loss = self.foc_loss(scATAC_recon_result, scATAC_ori)

        cross_loss_rna = self._rna_loss(rna_cross, scRNA_ori)
        cross_loss_atac = self.foc_loss(cross_scATAC, scATAC_ori)

        scRNA_recon_result = rna_self["mean"]
        cross_scRNA = rna_cross["mean"]

        return scRNA_recon_loss, scATAC_recon_loss, scRNA_recon_result, \
                scATAC_recon_result, cross_loss_rna, cross_loss_atac, cross_scRNA, cross_scATAC





