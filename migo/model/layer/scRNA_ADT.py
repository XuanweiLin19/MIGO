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

class ADT_Encoder(nn.Module):
    def __init__(self, ADT_dim, hidden_dim):#(m, 64)
        super(ADT_Encoder, self).__init__()

        self.ADT_linear = nn.Linear(ADT_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.ADT_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

    def forward(self, ADT_feat):
        ADT_feat = self.relu(self.ADT_linear(ADT_feat))
        out =  self.relu2(self.ADT_linear2(ADT_feat))
        return out #(N, 128)

class ADT_Reconstruction_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim):
        super().__init__()
        self.output_dim = output_dim
        self.vq_dim = vq_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(vq_dim, vq_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(vq_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(vq_dim * 2, vq_dim),
            nn.ReLU(),
            nn.LayerNorm(vq_dim),
            nn.Linear(vq_dim, output_dim),
            nn.Softplus()
        )
        
        self.scale_factor = nn.Parameter(torch.ones(output_dim))

    def forward(self, z_vq):
        reconstructed = self.decoder(z_vq)
        return reconstructed * self.scale_factor

    def reconstruct(self, z_vq, method="mean", deterministic=True):
        return self.forward(z_vq)

class scRNA_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(vq_dim, input_dim), 
            nn.GELU(),
            nn.LayerNorm(input_dim),
            nn.Dropout(0.1),
            nn.Linear(input_dim, output_dim),
            nn.GELU()
        )

    def forward(self, z_vq): 
        return self.fc(z_vq)

class RA_VQVAE_Encoder(nn.Module):# (256, 256, 128, 128, 128, 400)
    def __init__(self, scRNA_dim, ADT_dim, scRNA_output_dim, ADT_output_dim, embedding_dim, n_embed=400):
        super(RA_VQVAE_Encoder, self).__init__()
        self.scRNA_dim = scRNA_dim
        self.ADT_dim = ADT_dim
        self.h_dim = embedding_dim
        self.scRNA_encoder = scRNA_Encoder(scRNA_dim, scRNA_output_dim)#(256, 128)
        self.ADT_encoder = ADT_Encoder(ADT_dim, ADT_output_dim)
        self.Cross_quantizer = Cross_VQ_RA(n_embed, self.h_dim)
        # Using multi-head attention to obtain featur internal relation.
        self.scRNA_self_att = InternalRelationModule(input_dim=self.scRNA_dim, d_model=self.h_dim)
        self.ADT_self_att = InternalRelationModule(input_dim=self.ADT_dim, d_model=self.h_dim)

    def forward(self, scRNA_feature, ADT_feature, flag, CPC, device):

        scRNA_semantic_result = self.scRNA_self_att(scRNA_feature)  # [batch, hidden_dim]
        ADT_semantic_result = self.ADT_self_att(ADT_feature)  # [length, batch, hidden_dim]

        scRNA_encoder_result = self.scRNA_encoder(scRNA_feature)  # [batch, length, output_dim]
        ADT_encoder_result = self.ADT_encoder(ADT_feature)  # [batch, length, output_dim]

        _, Alignment_scRNA_Semantic, Alignment_ADT_Semantic = CPC(
            scRNA_semantic_result, ADT_semantic_result, device=device
        )

        scRNA_quantized, ADT_quantized, scRNA_loss, ADT_loss, scRNA_perplexity, ADT_perplexity,\
        batch_stats = self.Cross_quantizer(Alignment_scRNA_Semantic, Alignment_ADT_Semantic, flag)

        return scRNA_semantic_result, ADT_semantic_result, scRNA_encoder_result, ADT_encoder_result, \
            scRNA_quantized, ADT_quantized, scRNA_loss, ADT_loss, Alignment_scRNA_Semantic, Alignment_ADT_Semantic, batch_stats
    
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

class ADT_Loss(nn.Module):
    def __init__(self, loss_type='mse', alpha=0.5):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        
    def forward(self, pred, target):
        if self.loss_type == 'mse':
            return F.mse_loss(pred, target)
        
        elif self.loss_type == 'poisson':
            pred_safe = torch.clamp(pred, min=1e-8)
            return F.poisson_nll_loss(pred_safe, target, log_input=False, reduction='mean')
        
        elif self.loss_type == 'combined':
            mse_loss = F.mse_loss(pred, target)
            pred_safe = torch.clamp(pred, min=1e-8)
            poisson_loss = F.poisson_nll_loss(pred_safe, target, log_input=False, reduction='mean')
            return self.alpha * mse_loss + (1 - self.alpha) * poisson_loss
        
        elif self.loss_type == 'huber':
            return F.huber_loss(pred, target, reduction='mean', delta=1.0)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

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
        self.softmax_temperature = 0.5
        self.register_buffer("unactivated_count", -torch.ones(num_embeddings)) #unactivated:-1
        self.register_buffer("usage_count", torch.zeros(num_embeddings))
        self.register_buffer("rna_usage", torch.zeros(num_embeddings))
        self.register_buffer("ADT_usage", torch.zeros(num_embeddings))
        self.register_buffer("batch_count", torch.tensor(0))
        
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

    def ADT_vq_embedding(self, ADT_semantic):
        quant_codebook = self.embedding_proj(self._embedding.weight)
        distances = (
                    torch.sum(ADT_semantic ** 2, dim=1, keepdim=True) + torch.sum(quant_codebook ** 2, dim=1)
                    - 2 * torch.matmul(ADT_semantic, quant_codebook.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=ADT_semantic.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, quant_codebook).view(ADT_semantic.shape)
        ADT_quantized = ADT_semantic + (quantized - ADT_semantic).detach()
        return ADT_quantized

    def forward(self, scRNA_semantic, ADT_semantic, flag):

        # B, D = scRNA_semantic.size()

        scR_flat = scRNA_semantic.detach()
        scA_flat = ADT_semantic.detach()

        quant_codebook = self.embedding_proj(self._embedding.weight)
        # print('The weight of self.embedding_proj: ', self.embedding_proj.weight)
        scR_distances = (torch.sum(scR_flat ** 2, dim=1, keepdim=True) + torch.sum(quant_codebook ** 2, dim=1)
                         - 2 * torch.matmul(scR_flat, quant_codebook.t()))
        scA_distances = (torch.sum(scA_flat ** 2, dim=1, keepdim=True) + torch.sum(quant_codebook ** 2, dim=1)
                         - 2 * torch.matmul(scA_flat, quant_codebook.t()))

        temperature = max(self.softmax_temperature, 1e-3)
        scRNA_soft_assign = torch.softmax(-scR_distances / temperature, dim=1)
        ADT_soft_assign = torch.softmax(-scA_distances / temperature, dim=1)

        scRNA_encoding_indices = torch.argmin(scR_distances, dim=1).unsqueeze(1)
        scRNA_encodings = torch.zeros(scRNA_encoding_indices.shape[0], self._num_embeddings,
                                      device=scR_distances.device).double()
        scRNA_encodings.scatter_(1, scRNA_encoding_indices, 1)
        scRNA_quantized = torch.matmul(scRNA_encodings.double(), quant_codebook).view(scRNA_semantic.shape)

        ADT_encoding_indices = torch.argmin(scA_distances, dim=1).unsqueeze(1)
        ADT_encodings = torch.zeros(ADT_encoding_indices.shape[0], self._num_embeddings,
                                       device=scR_distances.device).double()
        ADT_encodings.scatter_(1, ADT_encoding_indices, 1)
        ADT_quantized = torch.matmul(ADT_encodings.double(), quant_codebook).view(ADT_semantic.shape)


        scRNA_quantized_soft = torch.matmul(scRNA_soft_assign, quant_codebook).view(scRNA_semantic.shape)
        ADT_quantized_soft = torch.matmul(ADT_soft_assign, quant_codebook).view(ADT_semantic.shape)

        scRNA_indices = scRNA_encoding_indices.squeeze(-1)  # [B, 1] -> [B]
        ADT_indices = ADT_encoding_indices.squeeze(-1)
        
        all_indices = torch.cat([scRNA_indices, ADT_indices])
        
        with torch.no_grad():
            rna_counts = torch.bincount(scRNA_indices, minlength=self._num_embeddings)
            ADT_counts = torch.bincount(ADT_indices, minlength=self._num_embeddings)
            total_counts = torch.bincount(all_indices, minlength=self._num_embeddings)
            
            self.usage_count += (total_counts > 0).float()
            self.rna_usage += (rna_counts > 0).float()
            self.ADT_usage += (ADT_counts > 0).float()
            
            self.unactivated_count = torch.where(
                total_counts > 0,
                torch.zeros_like(self.unactivated_count),
                self.unactivated_count
            )
            
            self.batch_count += 1

        scRNA_e_latent_loss = F.mse_loss(scRNA_semantic, scRNA_quantized.detach())
  
        scRNA_loss = 2 * self._commitment_cost * scRNA_e_latent_loss

        ADT_scRNA_vq_forward_loss = ((F.mse_loss(ADT_quantized, ADT_semantic.detach())
                                         + F.mse_loss(scRNA_quantized, scRNA_semantic.detach())
                                        + 0.5 * F.mse_loss(ADT_quantized, scRNA_semantic.detach()))
                                        + 0.5 * F.mse_loss(scRNA_quantized, ADT_semantic.detach()))

        ADT_e_latent_loss = F.mse_loss(ADT_semantic, ADT_quantized.detach())

        ADT_loss = 2 * self._commitment_cost * ADT_e_latent_loss + self._commitment_cost * ADT_scRNA_vq_forward_loss

        scRNA_quantized = scRNA_semantic + (scRNA_quantized_soft - scRNA_semantic).detach()
        ADT_quantized = ADT_semantic + (ADT_quantized_soft - ADT_semantic).detach()

        scRNA_avg_probs = torch.mean(scRNA_encodings, dim=0)
        scRNA_perplexity = torch.exp(-torch.sum(scRNA_avg_probs * torch.log(scRNA_avg_probs + 1e-10)))
        ADT_avg_probs = torch.mean(ADT_encodings, dim=0)
        ADT_perplexity = torch.exp(-torch.sum(ADT_avg_probs * torch.log(ADT_avg_probs + 1e-10)))

        batch_stats = self.get_batch_stats(scRNA_indices, ADT_indices)
        
        return scRNA_quantized, ADT_quantized, scRNA_loss, ADT_loss, scRNA_perplexity, ADT_perplexity, batch_stats
    
    def get_batch_stats(self, scRNA_indices, ADT_indices):
        if self.batch_count == 0:
            return {
                'global_util': 0.0,
                'rna_util': 0.0,
                'ADT_util': 0.0,
                'cross_util': 0.0,
                'active_codes': 0
            }
        
        global_util = (self.usage_count > 0).float().sum() / self._num_embeddings
        rna_util = (self.rna_usage > 0).float().sum() / self._num_embeddings
        ADT_util = (self.ADT_usage > 0).float().sum() / self._num_embeddings
        
        cross_util = ((self.rna_usage > 0) & (self.ADT_usage > 0)).float().sum() / self._num_embeddings
        batch_util = torch.unique(torch.cat([scRNA_indices, ADT_indices])).size(0) / self._num_embeddings
        active_codes = (self.usage_count > 0).float().sum().item()

        return {
            'global_util': global_util.item(),
            'rna_util': rna_util.item(),
            'ADT_util': ADT_util.item(),
            'cross_util': cross_util.item(),
            'batch_util': float(batch_util),
            'active_codes': active_codes
        }

    def get_codebook_utilization(self):
        if self.batch_count == 0:
            return {
                'global_util': 0.0,
                'rna_util': 0.0,
                'ADT_util': 0.0,
                'cross_util': 0.0,
                'active_codes': 0
            }
        global_util = (self.usage_count > 0).float().sum() / self._num_embeddings
        rna_util = (self.rna_usage > 0).float().sum() / self._num_embeddings
        ADT_util = (self.ADT_usage > 0).float().sum() / self._num_embeddings
        cross_util = ((self.rna_usage > 0) & (self.ADT_usage > 0)).float().sum() / self._num_embeddings
        active_codes = (self.usage_count > 0).float().sum().item()
        return {
            'global_util': global_util.item(),
            'rna_util': rna_util.item(),
            'ADT_util': ADT_util.item(),
            'cross_util': cross_util.item(),
            'active_codes': active_codes
        }


class RA_VQVAE_Decoder(nn.Module):
    def __init__(self, scRNA_dim, ADT_dim, scRNA_output_dim, ADT_output_dim):
        super(RA_VQVAE_Decoder, self).__init__()
        self.hidden_dim = 64
        
        self.scRNA_decoder = scRNA_Decoder(scRNA_dim, scRNA_output_dim, self.hidden_dim) #(256, 128, 64)
        self.ADT_decoder = ADT_Reconstruction_Decoder(ADT_dim, ADT_output_dim, self.hidden_dim) #(256, ADT_output_dim, 64)

        self.rna_loss = nn.MSELoss()
        self.ADT_loss = ADT_Loss(loss_type='combined', alpha=0.7)

    def forward(self, scRNA_ori, ADT_ori, scRNA_vq, ADT_vq):

        scRNA_recon_result = self.scRNA_decoder(scRNA_vq)
        ADT_recon_result = self.ADT_decoder(ADT_vq)

        cross_scRNA_result = self.scRNA_decoder(ADT_vq)
        cross_ADT_result = self.ADT_decoder(scRNA_vq)

        scRNA_recon_loss = self.rna_loss(scRNA_recon_result, scRNA_ori)
        ADT_recon_loss = self.ADT_loss(ADT_recon_result, ADT_ori)

        cross_loss_rna = self.rna_loss(cross_scRNA_result, scRNA_ori) 
        cross_loss_ADT = self.ADT_loss(cross_ADT_result, ADT_ori)

        return scRNA_recon_loss, ADT_recon_loss, scRNA_recon_result,\
                ADT_recon_result, cross_loss_rna, cross_loss_ADT, cross_scRNA_result, cross_ADT_result
