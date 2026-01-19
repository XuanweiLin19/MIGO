import os
import numpy as np
import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, drop=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class Cross_CPC_RA(nn.Module):
    def __init__(self, embedding_dim, projection_dim, drop=0.2):
        super().__init__()
        self.CLIP_linear_rna = ProjectionHead(input_dim=128, output_dim=projection_dim, drop=drop)
        self.CLIP_linear_atac = ProjectionHead(input_dim=128, output_dim=projection_dim, drop=drop)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, rna_embedding, atac_embedding, device=None):
        if device is None:
            device = next(self.parameters()).device

        rna_embedding_linear = self.CLIP_linear_rna(rna_embedding)
        atac_embedding_linear = self.CLIP_linear_atac(atac_embedding)

        rna_embedding_norm = rna_embedding_linear / rna_embedding_linear.norm(dim=1, keepdim=True)
        atac_embedding_norm = atac_embedding_linear / atac_embedding_linear.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_rna = logit_scale * rna_embedding_norm @ atac_embedding_norm.t()
        logits_per_atac = logits_per_rna.t()

        label = torch.arange(logits_per_rna.shape[0]).to(device)
        loss_cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

        loss1 = loss_cross_entropy(logits_per_rna, label)
        loss2 = loss_cross_entropy(logits_per_atac, label)
        loss = (loss1 + loss2) / 2

        return loss, rna_embedding_norm, atac_embedding_norm


def _unpack_cpc_embeddings(cpc_output):
    if not isinstance(cpc_output, (tuple, list)):
        return None, None
    if len(cpc_output) >= 3:
        return cpc_output[1], cpc_output[2]
    if len(cpc_output) == 2:
        return cpc_output[1], None
    return None, None

_RESOLUTION = os.environ.get("MIGO_RESOLUTION", "1M")

if _RESOLUTION == "1M":
    
    import torch.nn.functional as F
    import random
    import torch
    from torch.nn import functional as F
    import torch.nn as nn
    from torch.nn import Module, MultiheadAttention, Dropout, Linear, LayerNorm
    from migo.model.utils import _get_clones, _get_activation_fn
    from torch.distributions import NegativeBinomial
    from pytorch_msssim import ssim, ms_ssim
    import numpy as np
    
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
    
    class NB_Loss(nn.Module):
        def forward(self, params, counts):
            mean, disp = params
            
            eps = 1e-10
            t1 = torch.lgamma(disp + eps) + torch.lgamma(counts + 1.0) - torch.lgamma(counts + disp + eps)
            t2 = (disp + counts) * torch.log(1.0 + (mean/(disp + eps))) + (counts * (torch.log(disp + eps) - torch.log(mean + eps)))
            return torch.mean(t1 + t2)
    
        
    class scRNA_Encoder(nn.Module):
        def __init__(self, scRNA_dim, hidden_dim):
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
            return out
    
    class scHiC_Encoder(nn.Module):
        def __init__(self, scHiC_dim, hidden_dim):
            super(scHiC_Encoder, self).__init__()
            self.scHiC_dim = scHiC_dim
            self.hidden_dim = hidden_dim
            self.scHiC_linear = nn.Linear(scHiC_dim, hidden_dim)
            self.relu1 = nn.ReLU()
            self.scHiC_linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.relu2 = nn.ReLU()
    
        def forward(self, scHiC_feat):
            scHiC_feat = self.relu1(self.scHiC_linear(scHiC_feat))
            out = self.relu2(self.scHiC_linear2(scHiC_feat))
            return out
        
    class scRNA_Decoder(nn.Module):
        def __init__(self, input_dim, output_dim, vq_dim):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(vq_dim, input_dim), 
                nn.GELU(),
                nn.Linear(input_dim, output_dim),
                nn.GELU()
            )
    
        def forward(self, z_vq): 
            return self.fc(z_vq)
    
    
    def to_count_mean(model_output):
    
        mean, _ = model_output
        return mean
        
    def to_count_sample(model_output):
        mean, disp = model_output
        dist = NegativeBinomial(total_count=disp, probs=mean / (mean + disp))
        return dist.sample()
        
    def to_count_map(model_output):
        mean, disp = model_output
        mode = torch.where(disp > 1, (disp-1)/disp * mean, 0)
        return torch.round(torch.clamp(mode, min=0))
    
    class scRNA_NB_Decoder(nn.Module):
        def __init__(self, input_dim, output_dim, vq_dim):
            super().__init__()
            self.output_dim = output_dim
            
            self.mean_net = nn.Sequential(
                nn.Linear(vq_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, output_dim),
                nn.Softplus() 
            )     
            self.disp_net = nn.Sequential(
                nn.Linear(vq_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, output_dim),
                nn.Softplus()
            )
    
        def forward(self, z_vq):
            mean = self.mean_net(z_vq)
            disp = self.disp_net(z_vq) + 1e-5
            
            return (mean, disp)
    
        def to_count(self, model_output, method="mean"):
            if method == "mean":
                return to_count_mean(model_output)
            elif method == "sample":
                return to_count_sample(model_output)
            elif method == "map":
                return to_count_map(model_output)
            else:
                raise ValueError("无效的方�? 使用'mean', 'sample'�?map'")    
    class SymmetricTransposeConv(nn.Module):
        def __init__(self, in_c, out_c, kernel, stride, padding):
            super().__init__()
            self.conv = nn.ConvTranspose2d(in_c, out_c, kernel, stride, padding)

        def forward(self, x):
            x = self.conv(x)
            return (x + x.permute(0, 1, 3, 2)) / 2.0
    class SymmetrizeLayer(nn.Module):
        def forward(self, x):
            return (x + x.permute(0, 1, 3, 2)) / 2.0
            
    
    class UNetDownBlock(nn.Module):
        def __init__(self, in_channels, out_channels, norm=True, dropout=0.0):
            super().__init__()
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.model = nn.Sequential(*layers)
    
        def forward(self, x):
            return self.model(x)
    
    class UNetUpBlock(nn.Module):
        def __init__(self, in_channels, out_channels, dropout=0.0, cond_dim=None):
            super().__init__()
            
            self.conv_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True)
            )
            
            if dropout > 0:
                self.dropout = nn.Dropout(dropout)
            else:
                self.dropout = None
    
            self.use_cond = cond_dim is not None
            if self.use_cond:
                self.cond_projector = nn.Linear(cond_dim, out_channels * 2)
    
        def forward(self, x, skip_input, cond_info=None):
            diffY = skip_input.size()[2] - x.size()[2]
            diffX = skip_input.size()[3] - x.size()[3]
    
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            
            x = torch.cat([x, skip_input], 1)
    
            x = self.conv_block(x)
            
            if self.use_cond and cond_info is not None:
                style = self.cond_projector(cond_info)
                scale, bias = style.chunk(2, dim=1)
                scale = scale.unsqueeze(-1).unsqueeze(-1)
                bias = bias.unsqueeze(-1).unsqueeze(-1)
                x = x * scale + bias
            
            if self.dropout is not None:
                x = self.dropout(x)
                
            return x
        
    
    class HiC_UNet_VQVAE(nn.Module):
        def __init__(self, embedding_dim=128, n_embed=1024, input_size=196, combined_rna_dim=None):
            super().__init__()
            
            self.input_size = input_size
            
            self.bottleneck_size = input_size // 8
            
            self.down1 = UNetDownBlock(1, 32, norm=False)
            self.down2 = UNetDownBlock(32, 64)
            self.down3 = UNetDownBlock(64, 128)
            self.bottleneck_conv = nn.Conv2d(128, embedding_dim, kernel_size=1)
            
            self.up_conv = nn.Conv2d(embedding_dim * 2, 128, kernel_size=1)
            
            self.up1 = UNetUpBlock(in_channels=256, out_channels=64, dropout=0.5)
            self.up2 = UNetUpBlock(in_channels=128, out_channels=32, dropout=0.5)
            self.up3 = UNetUpBlock(in_channels=64, out_channels=64)
            
            self.final_up = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
            
            self.final_resize = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=True)
            self.symmetrize = SymmetrizeLayer()
        
        def get_bottleneck_size(self):
            return self.bottleneck_size
    
        def encode(self, x):
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            d1 = self.down1(x)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            bottleneck = self.bottleneck_conv(d3)
            skips = (d1, d2, d3)
            return bottleneck, skips
    
        def decode(self, z, skips, cond_info=None):
            d1, d2, d3 = skips
            
            x = self.up_conv(z)
            
            x = self.up1(x, skip_input=d3, cond_info=cond_info)
            x = self.up2(x, skip_input=d2, cond_info=cond_info)
            x = self.up3(x, skip_input=d1, cond_info=cond_info)
            
            recon_x = self.final_up(x)
            recon_x = self.final_resize(recon_x)
            recon_x = self.symmetrize(recon_x)
            return recon_x
    
        def forward(self, x):
            bottleneck, skips = self.encode(x)
            recon_x = self.decode(bottleneck, skips)
            return recon_x

    class EnhancedCrossModalGenerator(nn.Module):
        def __init__(self, combined_rna_dim, embedding_dim, hic_input_size):
            super().__init__()

            self.hic_input_size = hic_input_size
            self.embedding_dim = embedding_dim

            self.d3_size = hic_input_size // 8
            self.d2_size = hic_input_size // 4
            self.d1_size = hic_input_size // 2

            self.register_buffer(
                "distance_prior",
                self._create_hic_distance_prior(embedding_dim, self.d3_size),
            )

            self.prior_scale = nn.Parameter(torch.tensor(0.05))
            self.gate_net = nn.Sequential(
                nn.Linear(combined_rna_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

            self.feature_extractor = nn.Sequential(
                nn.Linear(combined_rna_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 1024),
                nn.LayerNorm(1024),
                nn.ReLU(),
            )

            self.bottleneck_generator = nn.Sequential(
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, embedding_dim * self.d3_size * self.d3_size),
            )

            self.skip_feature_generator = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.GELU(),
                nn.Dropout(0.1),
            )

            self.skip_heads = nn.ModuleDict({
                "d1": nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, 32 * self.d1_size * self.d1_size),
                    nn.Tanh(),
                ),
                "d2": nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, 64 * self.d2_size * self.d2_size),
                    nn.Tanh(),
                ),
                "d3": nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 128 * self.d3_size * self.d3_size),
                    nn.Tanh(),
                ),
            })

            self.spatial_aware_conv = nn.ModuleDict({
                "d1": self._make_spatial_conv(32, 32),
                "d2": self._make_spatial_conv(64, 64),
                "d3": self._make_spatial_conv(128, 128),
            })

        def _create_hic_distance_prior(self, channels, size):
            i = torch.arange(size).float().view(-1, 1)
            j = torch.arange(size).float().view(1, -1)
            dist = torch.abs(i - j)
            dist.fill_diagonal_(1.0)

            alpha = 1.0
            epsilon = 1e-8
            decay_weight = 1.0 / (torch.pow(dist, alpha) + epsilon)
            decay_weight = decay_weight / decay_weight.max()

            encoding = torch.zeros(1, channels, size, size)
            num_freq_groups = 4
            channels_per_group = max(1, channels // num_freq_groups)
            for c in range(channels):
                freq_group = min(c // channels_per_group, num_freq_groups - 1)
                freq = 2.0 ** freq_group
                if c % 2 == 0:
                    encoding[0, c] = decay_weight * torch.sin(freq * dist / size * np.pi)
                else:
                    encoding[0, c] = decay_weight * torch.cos(freq * dist / size * np.pi)

            encoding = (encoding + encoding.transpose(-1, -2)) / 2.0
            encoding_std = encoding.std()
            if encoding_std > 1e-8:
                encoding = encoding / encoding_std
            return encoding

        def _make_spatial_conv(self, in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        def forward(self, combined_rna_info, B, device):
            shared_features = self.feature_extractor(combined_rna_info)

            bottleneck_flat = self.bottleneck_generator(shared_features)
            bottleneck = bottleneck_flat.view(B, self.embedding_dim, self.d3_size, self.d3_size)

            gate_weight = self.gate_net(combined_rna_info).view(B, 1, 1, 1)
            prior = self.distance_prior.expand(B, -1, -1, -1).to(device)
            bottleneck = bottleneck + self.prior_scale * gate_weight * prior
            bottleneck = (bottleneck + bottleneck.transpose(-1, -2)) / 2.0

            skip_features = self.skip_feature_generator(shared_features)

            d1 = self.skip_heads["d1"](skip_features).view(B, 32, self.d1_size, self.d1_size)
            d1 = self.spatial_aware_conv["d1"](d1)
            d1 = (d1 + d1.transpose(-1, -2)) / 2.0

            d2 = self.skip_heads["d2"](skip_features).view(B, 64, self.d2_size, self.d2_size)
            d2 = self.spatial_aware_conv["d2"](d2)
            d2 = (d2 + d2.transpose(-1, -2)) / 2.0

            d3 = self.skip_heads["d3"](skip_features).view(B, 128, self.d3_size, self.d3_size)
            d3 = self.spatial_aware_conv["d3"](d3)
            d3 = (d3 + d3.transpose(-1, -2)) / 2.0

            return bottleneck, (d1, d2, d3)
    
    class RA_VQVAE_Encoder(nn.Module):
        def __init__(self, scRNA_dim, scRNA_output_dim, HiC_output_dim, embedding_dim, n_embed=1024, hic_input_size=196):
            super().__init__()
            self.h_dim = embedding_dim
            self.hic_input_size = hic_input_size
            
            self.scRNA_encoder = scRNA_Encoder(scRNA_dim, scRNA_output_dim)
            self.scRNA_self_att = InternalRelationModule(input_dim=scRNA_dim, d_model=self.h_dim)
            
            combined_rna_dim = embedding_dim * 2 + scRNA_output_dim
            
            self.hic_unet = HiC_UNet_VQVAE(
                embedding_dim=embedding_dim, 
                n_embed=n_embed, 
                input_size=hic_input_size,
                combined_rna_dim=combined_rna_dim
            )
            
            self.hic_semantic_head = nn.Linear(embedding_dim, embedding_dim)
            self.hic_specific_head = nn.Linear(embedding_dim, HiC_output_dim)
    
            self.Cross_quantizer = Cross_VQEmbeddingEMA_RA(n_embed, self.h_dim)
    
        def forward(self, scRNA_feature, HiC_2d_data, flag, CPC, device):
    
            scRNA_semantic_result = self.scRNA_self_att(scRNA_feature)
            scRNA_encoder_result = self.scRNA_encoder(scRNA_feature)
    
            hic_bottleneck, hic_skips = self.hic_unet.encode(HiC_2d_data)
            hic_bottleneck_pooled = hic_bottleneck.mean(dim=[2,3])
            scHiC_semantic_result = self.hic_semantic_head(hic_bottleneck_pooled)
            scHiC_encoder_result = self.hic_specific_head(hic_bottleneck_pooled)
    
            cpc_output = CPC(scRNA_semantic_result, scHiC_semantic_result, device=device)
            Alignment_scRNA_Semantic, Alignment_HiC_Semantic = _unpack_cpc_embeddings(cpc_output)
    
            (scRNA_quantized,
             scHiC_quantized_semantic,
             scRNA_loss,
             scHiC_loss,
             _,
             _,
             batch_stats) = self.Cross_quantizer(Alignment_scRNA_Semantic, Alignment_HiC_Semantic, flag)
    
            B, Cb, Hb, Wb = hic_bottleneck.shape
            semantic_spatial = scHiC_quantized_semantic.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Hb, Wb)
            decoder_input_bottleneck = torch.cat([hic_bottleneck, semantic_spatial], dim=1)
    
            return (scRNA_semantic_result, scHiC_semantic_result,
                    scRNA_encoder_result, scHiC_encoder_result,
                    scRNA_quantized, scHiC_quantized_semantic,
                    scRNA_loss, scHiC_loss,
                    batch_stats,
                    hic_skips,
                    decoder_input_bottleneck)
        
    class InternalRelationModule(nn.Module):
        def __init__(self, input_dim, d_model):
            super(InternalRelationModule, self).__init__()
            self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4)
            self.encoder = Encoder(self.encoder_layer, num_layers=1)
    
            self.affine_matrix = nn.Linear(input_dim, d_model)
            self.relu = nn.ReLU(inplace=False)
    
        def forward(self, feature):
            feature = self.affine_matrix(feature)
            feature = self.encoder(feature)
            return feature
    
    class Encoder(Module):
    
    
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
    
    
        def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="relu"):
            super(EncoderLayer, self).__init__()
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
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
            else:
                src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
            return src
    
    class Cross_VQEmbeddingEMA_RA(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
            super(Cross_VQEmbeddingEMA_RA, self).__init__()
            self._embedding_dim = embedding_dim
            self._num_embeddings = num_embeddings
            self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
            self._commitment_cost = 0.05
            self._decay = decay
            self._epsilon = epsilon
            
            self._init_embeddings()
            
            self.register_buffer("unactivated_count", -torch.ones(num_embeddings))
            self.register_buffer("usage_count", torch.zeros(num_embeddings))
            self.register_buffer("rna_usage", torch.zeros(num_embeddings))
            self.register_buffer("HiC_usage", torch.zeros(num_embeddings))
            self.register_buffer("batch_count", torch.tensor(0))

            for p in self._embedding.parameters():
                p.requires_grad = False

            self.embedding_proj = nn.Sequential(
                nn.Linear(self._embedding_dim, self._embedding_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self._embedding_dim * 2, self._embedding_dim),
                nn.LayerNorm(self._embedding_dim)
            )
            
            self.diversity_loss_weight = 0.01
    
        def _init_embeddings(self):
            nn.init.orthogonal_(self._embedding.weight[:min(self._num_embeddings, self._embedding_dim)])
            if self._num_embeddings > self._embedding_dim:
                nn.init.normal_(self._embedding.weight[self._embedding_dim:], mean=0, std=self._embedding_dim ** -0.5)

        def forward(self, scRNA_semantic, HiC_semantic, flag):
            device = scRNA_semantic.device

            scR_flat = scRNA_semantic.detach()
            scA_flat = HiC_semantic.detach()

            quant_codebook = self.embedding_proj(self._embedding.weight)
            
            temperature = 0.5 if self.training else 1.0
            
            scR_distances = (torch.sum(scR_flat ** 2, dim=1, keepdim=True) + 
                            torch.sum(quant_codebook ** 2, dim=1) -
                            2 * torch.matmul(scR_flat, quant_codebook.t())) / temperature
            
            scA_distances = (torch.sum(scA_flat ** 2, dim=1, keepdim=True) + 
                            torch.sum(quant_codebook ** 2, dim=1) -
                            2 * torch.matmul(scA_flat, quant_codebook.t())) / temperature

            scRNA_encoding_indices = torch.argmin(scR_distances, dim=1).unsqueeze(1)
            scRNA_encodings = torch.zeros(scRNA_encoding_indices.shape[0], self._num_embeddings,
                                          device=device, dtype=scRNA_semantic.dtype)
            scRNA_encodings.scatter_(1, scRNA_encoding_indices, 1)
            scRNA_quantized = torch.matmul(scRNA_encodings.float(), quant_codebook).view(scRNA_semantic.shape)

            HiC_encoding_indices = torch.argmin(scA_distances, dim=1).unsqueeze(1)
            HiC_encodings = torch.zeros(HiC_encoding_indices.shape[0], self._num_embeddings,
                                       device=device, dtype=HiC_semantic.dtype)
            HiC_encodings.scatter_(1, HiC_encoding_indices, 1)
            HiC_quantized = torch.matmul(HiC_encodings.float(), quant_codebook).view(HiC_semantic.shape)

            scRNA_indices = scRNA_encoding_indices.squeeze(-1)
            HiC_indices = HiC_encoding_indices.squeeze(-1)
            
            with torch.no_grad():
                self._update_usage_stats(scRNA_indices, HiC_indices)

            scRNA_e_latent_loss = F.mse_loss(scRNA_semantic, scRNA_quantized.detach())
            scRNA_loss = self._commitment_cost * scRNA_e_latent_loss

            HiC_e_latent_loss = F.mse_loss(HiC_semantic, HiC_quantized.detach())

            cross_modal_loss = (F.mse_loss(HiC_quantized, scRNA_semantic.detach()) +
                               F.mse_loss(scRNA_quantized, HiC_semantic.detach()) +
                               0.5 * F.mse_loss(HiC_quantized, scRNA_quantized.detach()))
            
            diversity_loss = self._compute_diversity_loss(quant_codebook)
            
            HiC_loss = (self._commitment_cost * HiC_e_latent_loss + 
                       0.5 * cross_modal_loss +
                       self.diversity_loss_weight * diversity_loss)

            scRNA_quantized = scRNA_semantic + (scRNA_quantized - scRNA_semantic).detach()
            HiC_quantized = HiC_semantic + (HiC_quantized - HiC_semantic).detach()

            scRNA_avg_probs = torch.mean(scRNA_encodings, dim=0)
            scRNA_perplexity = torch.exp(-torch.sum(scRNA_avg_probs * torch.log(scRNA_avg_probs + 1e-10)))
            HiC_avg_probs = torch.mean(HiC_encodings, dim=0)
            HiC_perplexity = torch.exp(-torch.sum(HiC_avg_probs * torch.log(HiC_avg_probs + 1e-10)))

            batch_stats = self.get_batch_stats(scRNA_indices, HiC_indices)
            
            return scRNA_quantized, HiC_quantized, scRNA_loss, HiC_loss, scRNA_perplexity, HiC_perplexity, batch_stats

        def _compute_diversity_loss(self, codebook):
            normalized_codebook = F.normalize(codebook, dim=1)
            similarity_matrix = torch.mm(normalized_codebook, normalized_codebook.t())
            mask = ~torch.eye(self._num_embeddings, device=codebook.device, dtype=torch.bool)
            return torch.mean(similarity_matrix[mask] ** 2)

        def _update_usage_stats(self, scRNA_indices, HiC_indices):
            all_indices = torch.cat([scRNA_indices, HiC_indices])
            total_counts = torch.bincount(all_indices, minlength=self._num_embeddings)
            self.usage_count += (total_counts > 0).float()
            self.rna_usage += (torch.bincount(scRNA_indices, minlength=self._num_embeddings) > 0).float()
            self.HiC_usage += (torch.bincount(HiC_indices, minlength=self._num_embeddings) > 0).float()
            self.batch_count += 1

        def get_batch_stats(self, scRNA_indices, HiC_indices):
            return {
                'rna_util': torch.unique(scRNA_indices).size(0) / self._num_embeddings,
                'HiC_util': torch.unique(HiC_indices).size(0) / self._num_embeddings,
                'batch_util': torch.unique(torch.cat([scRNA_indices, HiC_indices])).size(0) / self._num_embeddings
            }

        def get_codebook_utilization(self):
            if self.batch_count == 0:
                return {'global_util': 0.0, 'rna_util': 0.0, 'HiC_util': 0.0, 'cross_util': 0.0, 'active_codes': 0}
            return {
                'global_util': ((self.usage_count > 0).float().sum() / self._num_embeddings).item(),
                'rna_util': ((self.rna_usage > 0).float().sum() / self._num_embeddings).item(),
                'HiC_util': ((self.HiC_usage > 0).float().sum() / self._num_embeddings).item(),
                'cross_util': (((self.rna_usage > 0) & (self.HiC_usage > 0)).float().sum() / self._num_embeddings).item(),
                'active_codes': (self.usage_count > 0).float().sum().item()
            }

    class RA_VQVAE_Decoder(nn.Module):
        def __init__(self, scRNA_intermediate_dim, scRNA_output_dim, HiC_output_dim, latent_dim, 
                     scRNA_specific_dim, perceptual_weight=1.0, hic_input_size=196):
            super().__init__()
            self.hic_input_size = hic_input_size
            self.scRNA_decoder = scRNA_NB_Decoder(scRNA_intermediate_dim, scRNA_output_dim, latent_dim)
            self.hic_unet_decoder = None
            self.rna_loss = NB_Loss()
            self.combined_latent_dim = latent_dim * 2
            self.hic_embedding_dim = latent_dim
            
            self.cross_modal_generator = EnhancedCrossModalGenerator(
                self.combined_latent_dim,
                self.hic_embedding_dim,
                hic_input_size,
            )

        def enhanced_hic_loss(self, pred_img, target_img):
            if pred_img.dim() == 3:
                pred_img = pred_img.unsqueeze(1)
            if target_img.dim() == 3:
                target_img = target_img.unsqueeze(1)
            mse_loss = F.mse_loss(pred_img, target_img)
            ssim_loss = 1.0 - ssim(pred_img, target_img, data_range=target_img.max(), size_average=True)
            symmetry_loss = F.mse_loss(pred_img.squeeze(1), pred_img.squeeze(1).transpose(-1, -2))
            distance_decay_loss = self.distance_decay_constraint(pred_img, target_img)
            sparsity_loss = self.sparsity_constraint(pred_img, target_img)
            local_structure_loss = self.local_structure_consistency(pred_img, target_img)
            return (0.3 * mse_loss + 0.4 * ssim_loss + 0.1 * symmetry_loss + 
                    0.1 * distance_decay_loss + 0.05 * sparsity_loss + 0.05 * local_structure_loss)

        def distance_decay_constraint(self, pred, target):
            B, C, H, W = pred.shape
            if not hasattr(self, 'dist_matrix') or self.dist_matrix.shape[0] != H:
                i_indices = torch.arange(H, device=pred.device).float().view(-1, 1)
                j_indices = torch.arange(W, device=pred.device).float().view(1, -1)
                dist = torch.abs(i_indices - j_indices)
                dist.fill_diagonal_(1.0)
                self.dist_matrix = dist
            weight_matrix = 1.0 / (torch.pow(self.dist_matrix, 1.0) + 1e-8)
            weight_matrix = weight_matrix / weight_matrix.max()
            return torch.mean(weight_matrix.unsqueeze(0).unsqueeze(0) * F.mse_loss(pred, target, reduction='none'))
        
        def sparsity_constraint(self, pred, target):
            pred_nonzero_ratio = (pred > 0.01).float().mean()
            target_nonzero_ratio = (target > 0.01).float().mean()
            return 0.1 * torch.mean(torch.abs(pred)) + F.mse_loss(pred_nonzero_ratio, target_nonzero_ratio)

        def local_structure_consistency(self, pred, target):
            return F.mse_loss(F.avg_pool2d(pred, 5, 1, 2), F.avg_pool2d(target, 5, 1, 2))

        def forward(self, scRNA_ori, HiC_ori, scRNA_vq, scRNA_semantic, scRNA_encoder_result,
                    hic_quantized_bottleneck, HiC_semantic_vq, hic_skips, hic_unet_module):
            if self.hic_unet_decoder is None:
                self.hic_unet_decoder = hic_unet_module.decode

            B = HiC_ori.size(0)
            hic_size = int(np.sqrt(HiC_ori.shape[1]))
            HiC_ori_matrix = HiC_ori.reshape(B, hic_size, hic_size)

            scRNA_recon_params = self.scRNA_decoder(scRNA_vq)
            scHiC_recon_result = self.hic_unet_decoder(hic_quantized_bottleneck, hic_skips, None)
            scHiC_recon_result = torch.clamp(scHiC_recon_result, min=0.0)

            cross_scRNA_params = self.scRNA_decoder(HiC_semantic_vq)
            combined_rna_info = torch.cat([scRNA_vq, scRNA_semantic], dim=1)
            
            Hb, Wb = hic_quantized_bottleneck.shape[2], hic_quantized_bottleneck.shape[3]
            gen_bottleneck, gen_skips = self.cross_modal_generator(
                combined_rna_info, B, scRNA_vq.device
            )
            
            semantic_spatial_expanded = scRNA_vq.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Hb, Wb)
            generated_bottleneck = torch.cat([gen_bottleneck, semantic_spatial_expanded], dim=1)

            cross_scHiC = self.hic_unet_decoder(generated_bottleneck, gen_skips, combined_rna_info)
            cross_scHiC = torch.clamp(cross_scHiC, min=0.0)

            scRNA_recon_loss = self.rna_loss(scRNA_recon_params, scRNA_ori)
            scHiC_recon_loss = self.enhanced_hic_loss(scHiC_recon_result, HiC_ori_matrix)
            cross_loss_rna = self.rna_loss(cross_scRNA_params, scRNA_ori)
            cross_loss_HiC = self.enhanced_hic_loss(cross_scHiC, HiC_ori_matrix)

            scRNA_recon_result = to_count_mean(scRNA_recon_params)
            cross_scRNA = to_count_mean(cross_scRNA_params)
            
            bottleneck_imitation_loss = F.mse_loss(generated_bottleneck, hic_quantized_bottleneck.detach())
            skips_imitation_loss = 0.0
            for gs, rs in zip(gen_skips, hic_skips):
                if gs.shape != rs.shape:
                    gs = F.interpolate(gs, size=rs.shape[2:], mode='bilinear', align_corners=False)
                    if gs.shape[1] != rs.shape[1]:
                        m = min(gs.shape[1], rs.shape[1])
                        gs, rs = gs[:, :m], rs[:, :m]
                skips_imitation_loss += F.mse_loss(gs, rs.detach())
            latent_imitation_loss = bottleneck_imitation_loss + skips_imitation_loss

            return (scRNA_recon_loss, scHiC_recon_loss, scRNA_recon_result, scHiC_recon_result.squeeze(1),
                    cross_loss_rna, cross_loss_HiC, cross_scRNA, cross_scHiC.squeeze(1), latent_imitation_loss)

elif _RESOLUTION == "50K":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import Module, MultiheadAttention, Dropout, Linear, LayerNorm
    from migo.model.utils import _get_clones, _get_activation_fn
    from pytorch_msssim import ssim
    import numpy as np
    
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
    
    class NB_Loss(nn.Module):
        def forward(self, params, counts):
            mean, disp = params
            eps = 1e-10
            t1 = torch.lgamma(disp + eps) + torch.lgamma(counts + 1.0) - torch.lgamma(counts + disp + eps)
            t2 = (disp + counts) * torch.log(1.0 + (mean/(disp + eps))) + (counts * (torch.log(disp + eps) - torch.log(mean + eps)))
            return torch.mean(t1 + t2)
    
    class scRNA_Encoder(nn.Module):
        def __init__(self, scRNA_dim, hidden_dim):
            super(scRNA_Encoder, self).__init__()
            self.scRNA_linear = nn.Linear(scRNA_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.scRNA_linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.relu2 = nn.ReLU()
    
        def forward(self, scRNA_feat):
            scRNA_feat = self.relu(self.scRNA_linear(scRNA_feat))
            return self.relu2(self.scRNA_linear2(scRNA_feat))
    
    class scHiC_Encoder(nn.Module):
        def __init__(self, scHiC_dim, hidden_dim):
            super(scHiC_Encoder, self).__init__()
            self.scHiC_linear = nn.Linear(scHiC_dim, hidden_dim)
            self.relu1 = nn.ReLU()
            self.scHiC_linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.relu2 = nn.ReLU()
    
        def forward(self, scHiC_feat):
            return self.relu2(self.scHiC_linear2(self.relu1(self.scHiC_linear(scHiC_feat))))
    
    class scRNA_NB_Decoder(nn.Module):
        def __init__(self, input_dim, output_dim, vq_dim):
            super().__init__()
            self.mean_net = nn.Sequential(
                nn.Linear(vq_dim, input_dim), nn.ReLU(),
                nn.Linear(input_dim, output_dim), nn.Softplus() 
            )     
            self.disp_net = nn.Sequential(
                nn.Linear(vq_dim, input_dim), nn.ReLU(),
                nn.Linear(input_dim, output_dim), nn.Softplus()
            )
    
        def forward(self, z_vq):
            return (self.mean_net(z_vq), self.disp_net(z_vq) + 1e-5)
    
    def to_count_mean(model_output):
        return model_output[0]
    
    class SymmetrizeLayer(nn.Module):
        def forward(self, x):
            return (x + x.permute(0, 1, 3, 2)) / 2.0

    class UNetDownBlock(nn.Module):
        def __init__(self, in_channels, out_channels, norm=True, dropout=0.0):
            super().__init__()
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.model = nn.Sequential(*layers)
    
        def forward(self, x):
            return self.model(x)
    
    class UNetUpBlock(nn.Module):
        def __init__(self, in_channels, out_channels, dropout=0.0, cond_dim=None):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_channels), nn.ReLU(True)
            )
            self.dropout = nn.Dropout(dropout) if dropout > 0 else None
            self.use_cond = cond_dim is not None
            if self.use_cond:
                self.cond_projector = nn.Linear(cond_dim, out_channels * 2)
    
        def forward(self, x, skip_input, cond_info=None):
            diffY = skip_input.size()[2] - x.size()[2]
            diffX = skip_input.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x, skip_input], 1)
            x = self.conv_block(x)
            if self.use_cond and cond_info is not None:
                style = self.cond_projector(cond_info)
                scale, bias = style.chunk(2, dim=1)
                x = x * scale.unsqueeze(-1).unsqueeze(-1) + bias.unsqueeze(-1).unsqueeze(-1)
            if self.dropout is not None:
                x = self.dropout(x)
            return x
    
    class HiC_UNet_VQVAE(nn.Module):
        def __init__(self, embedding_dim=128, n_embed=1024, input_size=196, combined_rna_dim=None):
            super().__init__()
            self.input_size = input_size
            self.bottleneck_size = input_size // 8
            
            self.down1 = UNetDownBlock(1, 32, norm=False)
            self.down2 = UNetDownBlock(32, 64)
            self.down3 = UNetDownBlock(64, 128)
            self.bottleneck_conv = nn.Conv2d(128, embedding_dim, kernel_size=1)
            self.up_conv = nn.Conv2d(embedding_dim * 2, 128, kernel_size=1)
            self.up1 = UNetUpBlock(256, 64, dropout=0.5)
            self.up2 = UNetUpBlock(128, 32, dropout=0.5)
            self.up3 = UNetUpBlock(64, 64)
            self.final_up = nn.ConvTranspose2d(64, 1, 4, 2, 1)
            self.final_resize = nn.Upsample(size=(input_size, input_size), mode='bilinear', align_corners=True)
            self.symmetrize = SymmetrizeLayer()
        
        def get_bottleneck_size(self):
            return self.bottleneck_size
    
        def encode(self, x):
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            d1, d2, d3 = self.down1(x), self.down2(self.down1(x)), self.down3(self.down2(self.down1(x)))
            return self.bottleneck_conv(d3), (d1, d2, d3)
    
        def decode(self, z, skips, cond_info=None):
            d1, d2, d3 = skips
            x = self.up_conv(z)
            x = self.up3(self.up2(self.up1(x, d3, cond_info), d2, cond_info), d1, cond_info)
            return self.symmetrize(self.final_resize(self.final_up(x)))
    
        def forward(self, x):
            bottleneck, skips = self.encode(x)
            return self.decode(bottleneck, skips)
    

    class RA_VQVAE_Encoder(nn.Module):
        def __init__(self, scRNA_dim, scRNA_output_dim, HiC_output_dim, embedding_dim, n_embed=1024, hic_input_size=196):
            super().__init__()
            self.h_dim = embedding_dim
            self.hic_input_size = hic_input_size
            self.scRNA_encoder = scRNA_Encoder(scRNA_dim, scRNA_output_dim)
            self.scRNA_self_att = InternalRelationModule(scRNA_dim, self.h_dim)
            self.hic_unet = HiC_UNet_VQVAE(embedding_dim, n_embed, hic_input_size)
            self.hic_semantic_head = nn.Linear(embedding_dim, embedding_dim)
            self.hic_specific_head = nn.Linear(embedding_dim, HiC_output_dim)
            self.Cross_quantizer = Cross_VQEmbeddingEMA_RA(n_embed, self.h_dim)
    
        def forward(self, scRNA_feature, HiC_2d_data, flag, CPC, device):
            scRNA_semantic_result = self.scRNA_self_att(scRNA_feature)
            scRNA_encoder_result = self.scRNA_encoder(scRNA_feature)
            hic_bottleneck, hic_skips = self.hic_unet.encode(HiC_2d_data)
            hic_bottleneck_pooled = hic_bottleneck.mean(dim=[2,3])
            scHiC_semantic_result = self.hic_semantic_head(hic_bottleneck_pooled)
            scHiC_encoder_result = self.hic_specific_head(hic_bottleneck_pooled)
            cpc_output = CPC(scRNA_semantic_result, scHiC_semantic_result, device)
            Alignment_scRNA_Semantic, Alignment_HiC_Semantic = _unpack_cpc_embeddings(cpc_output)
            scRNA_quantized, scHiC_quantized_semantic, scRNA_loss, scHiC_loss, _, _, batch_stats = \
                self.Cross_quantizer(Alignment_scRNA_Semantic, Alignment_HiC_Semantic, flag)
            B, Cb, Hb, Wb = hic_bottleneck.shape
            semantic_spatial = scHiC_quantized_semantic.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Hb, Wb)
            decoder_input_bottleneck = torch.cat([hic_bottleneck, semantic_spatial], dim=1)
            return (scRNA_semantic_result, scHiC_semantic_result, scRNA_encoder_result, scHiC_encoder_result,
                    scRNA_quantized, scHiC_quantized_semantic, scRNA_loss, scHiC_loss, batch_stats,
                    hic_skips, decoder_input_bottleneck)
    
    class InternalRelationModule(nn.Module):
        def __init__(self, input_dim, d_model):
            super().__init__()
            self.encoder_layer = EncoderLayer(d_model, 4)
            self.encoder = Encoder(self.encoder_layer, 1)
            self.affine_matrix = nn.Linear(input_dim, d_model)
    
        def forward(self, feature):
            return self.encoder(self.affine_matrix(feature))
    
    class Encoder(Module):
        def __init__(self, encoder_layer, num_layers, norm=None):
            super().__init__()
            self.layers = _get_clones(encoder_layer, num_layers)
            self.num_layers = num_layers
            self.norm = norm
    
        def forward(self, src):
            output = src
            for layer in self.layers:
                output = layer(output)
            return self.norm(output) if self.norm else output
    
    class EncoderLayer(Module):
        def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="relu"):
            super().__init__()
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
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
            src = self.norm1(src + self.dropout1(src2))
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            return self.norm2(src + self.dropout2(src2))
    
    class Cross_VQEmbeddingEMA_RA(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
            super(Cross_VQEmbeddingEMA_RA, self).__init__()
            self._embedding_dim = embedding_dim
            self._num_embeddings = num_embeddings
            self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
            self._commitment_cost = 0.05
            self._decay = decay
            self._epsilon = epsilon

            self._init_embeddings()

            self.register_buffer("unactivated_count", -torch.ones(num_embeddings))
            self.register_buffer("usage_count", torch.zeros(num_embeddings))
            self.register_buffer("rna_usage", torch.zeros(num_embeddings))
            self.register_buffer("HiC_usage", torch.zeros(num_embeddings))
            self.register_buffer("batch_count", torch.tensor(0))

            for p in self._embedding.parameters():
                p.requires_grad = False

            self.embedding_proj = nn.Sequential(
                nn.Linear(self._embedding_dim, self._embedding_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self._embedding_dim * 2, self._embedding_dim),
                nn.LayerNorm(self._embedding_dim)
            )

            self.diversity_loss_weight = 0.01

        def _init_embeddings(self):
            nn.init.orthogonal_(self._embedding.weight[:min(self._num_embeddings, self._embedding_dim)])
            if self._num_embeddings > self._embedding_dim:
                nn.init.normal_(self._embedding.weight[self._embedding_dim:], 
                              mean=0, std=self._embedding_dim ** -0.5)

        def forward(self, scRNA_semantic, HiC_semantic, flag):
            device = scRNA_semantic.device

            scR_flat = scRNA_semantic.detach()
            scA_flat = HiC_semantic.detach()

            quant_codebook = self.embedding_proj(self._embedding.weight)

            temperature = 0.5 if self.training else 1.0

            scR_distances = (torch.sum(scR_flat ** 2, dim=1, keepdim=True) + 
                            torch.sum(quant_codebook ** 2, dim=1) -
                            2 * torch.matmul(scR_flat, quant_codebook.t())) / temperature

            scA_distances = (torch.sum(scA_flat ** 2, dim=1, keepdim=True) + 
                            torch.sum(quant_codebook ** 2, dim=1) -
                            2 * torch.matmul(scA_flat, quant_codebook.t())) / temperature

            scRNA_encoding_indices = torch.argmin(scR_distances, dim=1).unsqueeze(1)
            scRNA_encodings = torch.zeros(scRNA_encoding_indices.shape[0], self._num_embeddings,
                                          device=device, dtype=scRNA_semantic.dtype)
            scRNA_encodings.scatter_(1, scRNA_encoding_indices, 1)
            scRNA_quantized = torch.matmul(scRNA_encodings.float(), quant_codebook).view(scRNA_semantic.shape)

            HiC_encoding_indices = torch.argmin(scA_distances, dim=1).unsqueeze(1)
            HiC_encodings = torch.zeros(HiC_encoding_indices.shape[0], self._num_embeddings,
                                       device=device, dtype=HiC_semantic.dtype)
            HiC_encodings.scatter_(1, HiC_encoding_indices, 1)
            HiC_quantized = torch.matmul(HiC_encodings.float(), quant_codebook).view(HiC_semantic.shape)

            scRNA_indices = scRNA_encoding_indices.squeeze(-1)
            HiC_indices = HiC_encoding_indices.squeeze(-1)

            with torch.no_grad():
                self._update_usage_stats(scRNA_indices, HiC_indices)

            scRNA_e_latent_loss = F.mse_loss(scRNA_semantic, scRNA_quantized.detach())
            scRNA_loss = self._commitment_cost * scRNA_e_latent_loss

            HiC_e_latent_loss = F.mse_loss(HiC_semantic, HiC_quantized.detach())
            cross_modal_loss = (F.mse_loss(HiC_quantized, scRNA_semantic.detach()) +
                               F.mse_loss(scRNA_quantized, HiC_semantic.detach()) +
                               0.5 * F.mse_loss(HiC_quantized, scRNA_quantized.detach()))

            diversity_loss = self._compute_diversity_loss(quant_codebook)

            HiC_loss = (self._commitment_cost * HiC_e_latent_loss + 
                       0.5 * cross_modal_loss +
                       self.diversity_loss_weight * diversity_loss)

            scRNA_quantized = scRNA_semantic + (scRNA_quantized - scRNA_semantic).detach()
            HiC_quantized = HiC_semantic + (HiC_quantized - HiC_semantic).detach()

            scRNA_avg_probs = torch.mean(scRNA_encodings, dim=0)
            scRNA_perplexity = torch.exp(-torch.sum(scRNA_avg_probs * torch.log(scRNA_avg_probs + 1e-10)))
            HiC_avg_probs = torch.mean(HiC_encodings, dim=0)
            HiC_perplexity = torch.exp(-torch.sum(HiC_avg_probs * torch.log(HiC_avg_probs + 1e-10)))

            batch_stats = self.get_batch_stats(scRNA_indices, HiC_indices)

            return scRNA_quantized, HiC_quantized, scRNA_loss, HiC_loss, scRNA_perplexity, HiC_perplexity, batch_stats

        def _compute_diversity_loss(self, codebook):
            normalized_codebook = F.normalize(codebook, dim=1)
            similarity_matrix = torch.mm(normalized_codebook, normalized_codebook.t())
            mask = ~torch.eye(self._num_embeddings, device=codebook.device, dtype=torch.bool)
            off_diagonal_sim = similarity_matrix[mask]
            diversity_loss = torch.mean(off_diagonal_sim ** 2)
            return diversity_loss

        def _update_usage_stats(self, scRNA_indices, HiC_indices):
            all_indices = torch.cat([scRNA_indices, HiC_indices])

            rna_counts = torch.bincount(scRNA_indices, minlength=self._num_embeddings)
            HiC_counts = torch.bincount(HiC_indices, minlength=self._num_embeddings)
            total_counts = torch.bincount(all_indices, minlength=self._num_embeddings)

            self.usage_count += (total_counts > 0).float()
            self.rna_usage += (rna_counts > 0).float()
            self.HiC_usage += (HiC_counts > 0).float()

            self.unactivated_count = torch.where(
                total_counts > 0,
                torch.zeros_like(self.unactivated_count),
                self.unactivated_count
            )

            self.batch_count += 1

        def get_batch_stats(self, scRNA_indices, HiC_indices):
            unique_rna = torch.unique(scRNA_indices).size(0)
            unique_HiC = torch.unique(HiC_indices).size(0)
            unique_total = torch.unique(torch.cat([scRNA_indices, HiC_indices])).size(0)

            rna_util = unique_rna / self._num_embeddings
            HiC_util = unique_HiC / self._num_embeddings
            batch_util = unique_total / self._num_embeddings

            return {
                'rna_util': rna_util,
                'HiC_util': HiC_util,
                'batch_util': batch_util
            }

        def get_codebook_utilization(self):
            if self.batch_count == 0:
                return {
                    'global_util': 0.0,
                    'rna_util': 0.0,
                    'HiC_util': 0.0,
                    'cross_util': 0.0,
                    'active_codes': 0
                }

            global_util = (self.usage_count > 0).float().sum() / self._num_embeddings
            rna_util = (self.rna_usage > 0).float().sum() / self._num_embeddings
            HiC_util = (self.HiC_usage > 0).float().sum() / self._num_embeddings
            cross_util = ((self.rna_usage > 0) & (self.HiC_usage > 0)).float().sum() / self._num_embeddings
            active_codes = (self.usage_count > 0).float().sum().item()

            return {
                'global_util': global_util.item(),
                'rna_util': rna_util.item(),
                'HiC_util': HiC_util.item(),
                'cross_util': cross_util.item(),
                'active_codes': active_codes
            }

    class EnhancedCrossModalGenerator(nn.Module):

        def __init__(self, combined_rna_dim, embedding_dim, hic_input_size):
            super().__init__()

            self.hic_input_size = hic_input_size
            self.embedding_dim = embedding_dim


            self.d3_size = hic_input_size // 8
            self.d2_size = hic_input_size // 4
            self.d1_size = hic_input_size // 2


            self.register_buffer(
                'distance_prior', 
                self._create_hic_distance_prior(embedding_dim, self.d3_size)
            )
            self.prior_scale = nn.Parameter(torch.tensor(0.05))


            self.gate_net = nn.Sequential(
                nn.Linear(combined_rna_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )


            self.feature_extractor = nn.Sequential(
                nn.Linear(combined_rna_dim, 512),
                nn.LayerNorm(512),  
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 1024),
                nn.LayerNorm(1024), 
                nn.ReLU()
            )


            self.bottleneck_generator = ConvBottleneckGenerator(
                in_dim=1024,
                out_channels=embedding_dim,
                target_size=self.d3_size
            )


            self.skip_feature_generator = nn.Sequential(
                nn.Linear(1024, 2048),
                nn.GELU(),
                nn.Dropout(0.1)
            )

            self.skip_heads = nn.ModuleDict({
                'd1': ConvSkipGenerator(2048, 32, self.d1_size),
                'd2': ConvSkipGenerator(2048, 64, self.d2_size),
                'd3': ConvSkipGenerator(2048, 128, self.d3_size)
            })

            self.spatial_aware_conv = nn.ModuleDict({
                'd1': self._make_spatial_conv(32, 32),
                'd2': self._make_spatial_conv(64, 64), 
                'd3': self._make_spatial_conv(128, 128)
            })


            self.register_buffer('target_mean', torch.zeros(1))
            self.register_buffer('target_std', torch.ones(1))
            self.register_buffer('momentum', torch.tensor(0.9))

        def _create_hic_distance_prior(self, channels, size):

            i = torch.arange(size).float().view(-1, 1)
            j = torch.arange(size).float().view(1, -1)
            dist = torch.abs(i - j)
            dist.fill_diagonal_(1.0)

            alpha = 1.0
            epsilon = 1e-8
            decay_weight = 1.0 / (torch.pow(dist, alpha) + epsilon)
            decay_weight = decay_weight / decay_weight.max()

            encoding = torch.zeros(1, channels, size, size)
            num_freq_groups = 4
            channels_per_group = max(1, channels // num_freq_groups)

            for c in range(channels):
                freq_group = min(c // channels_per_group, num_freq_groups - 1)
                freq = 2.0 ** freq_group

                if c % 2 == 0:
                    encoding[0, c] = decay_weight * torch.sin(freq * dist / size * np.pi)
                else:
                    encoding[0, c] = decay_weight * torch.cos(freq * dist / size * np.pi)

            encoding = (encoding + encoding.transpose(-1, -2)) / 2.0

            encoding_std = encoding.std()
            if encoding_std > 1e-8:
                encoding = encoding / encoding_std

            return encoding

        def _make_spatial_conv(self, in_channels, out_channels):

            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        def update_target_statistics(self, real_bottleneck):

            with torch.no_grad():
                batch_mean = real_bottleneck.mean()
                batch_std = real_bottleneck.std()


                self.target_mean = self.momentum * self.target_mean + (1 - self.momentum) * batch_mean
                self.target_std = self.momentum * self.target_std + (1 - self.momentum) * batch_std

        def align_distribution(self, generated_feature):

            gen_mean = generated_feature.mean()
            gen_std = generated_feature.std() + 1e-8
            normalized = (generated_feature - gen_mean) / gen_std

            aligned = normalized * self.target_std + self.target_mean

            return aligned

        def forward(self, combined_rna_info, B, device):

            shared_features = self.feature_extractor(combined_rna_info)


            bottleneck = self.bottleneck_generator(shared_features, B)

            if self.training:
                bottleneck = self.align_distribution(bottleneck)


            gate_weight = self.gate_net(combined_rna_info).view(B, 1, 1, 1)
            prior = self.distance_prior.expand(B, -1, -1, -1).to(device)
            bottleneck = bottleneck + self.prior_scale * gate_weight * prior


            bottleneck = (bottleneck + bottleneck.transpose(-1, -2)) / 2.0


            skip_features = self.skip_feature_generator(shared_features)

            d1 = self.skip_heads['d1'](skip_features, B)
            d1 = self.spatial_aware_conv['d1'](d1)
            d1 = (d1 + d1.transpose(-1, -2)) / 2.0

            d2 = self.skip_heads['d2'](skip_features, B)
            d2 = self.spatial_aware_conv['d2'](d2)
            d2 = (d2 + d2.transpose(-1, -2)) / 2.0

            d3 = self.skip_heads['d3'](skip_features, B)
            d3 = self.spatial_aware_conv['d3'](d3)
            d3 = (d3 + d3.transpose(-1, -2)) / 2.0

            return bottleneck, (d1, d2, d3)

    class ConvBottleneckGenerator(nn.Module):

        def __init__(self, in_dim, out_channels, target_size):
            super().__init__()


            self.init_size = max(8, target_size // 16) 
            self.target_size = target_size

            self.fc_init = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, out_channels * self.init_size * self.init_size)
            )


            upsample_layers = []
            current_size = self.init_size

            while current_size < target_size:
                next_size = min(current_size * 2, target_size)

                if next_size == target_size and current_size * 2 > target_size:

                    upsample_layers.append(
                        nn.AdaptiveAvgPool2d((target_size, target_size))
                    )
                    break
                else:
                    upsample_layers.extend([
                        nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, 3, 1, 1), 
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    ])
                    current_size = next_size

            self.upsample = nn.Sequential(*upsample_layers)

        def forward(self, x, B):
            x = self.fc_init(x)
            x = x.view(B, -1, self.init_size, self.init_size)
            x = self.upsample(x)
            return x

    class ConvSkipGenerator(nn.Module):

        def __init__(self, in_dim, out_channels, target_size):
            super().__init__()

            self.init_size = max(8, target_size // 16)
            self.target_size = target_size


            self.fc_init = nn.Sequential(
                nn.Linear(in_dim, out_channels * self.init_size * self.init_size)
            )


            upsample_layers = []
            current_size = self.init_size

            while current_size < target_size:
                next_size = min(current_size * 2, target_size)

                if next_size == target_size and current_size * 2 > target_size:
                    upsample_layers.append(
                        nn.AdaptiveAvgPool2d((target_size, target_size))
                    )
                    break
                else:
                    upsample_layers.extend([
                        nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    ])
                    current_size = next_size

            self.upsample = nn.Sequential(*upsample_layers)

        def forward(self, x, B):
            x = self.fc_init(x)
            x = x.view(B, -1, self.init_size, self.init_size)
            x = self.upsample(x)
            return x


    class RA_VQVAE_Decoder(nn.Module):
        def __init__(self, scRNA_intermediate_dim, scRNA_output_dim, HiC_output_dim, latent_dim, 
                     scRNA_specific_dim, perceptual_weight=1.0, hic_input_size=196):
            super().__init__()

            self.hic_input_size = hic_input_size

            self.scRNA_decoder = scRNA_NB_Decoder(
                input_dim=scRNA_intermediate_dim, 
                output_dim=scRNA_output_dim, 
                vq_dim=latent_dim
            )

            self.hic_unet_decoder = None
            self.rna_loss = NB_Loss()
            self.perceptual_weight = perceptual_weight

            self.combined_latent_dim = latent_dim * 2
            self.hic_embedding_dim = latent_dim

            self.cross_modal_generator = EnhancedCrossModalGenerator(
                self.combined_latent_dim, 
                self.hic_embedding_dim,
                hic_input_size
            )

        def enhanced_hic_loss(self, pred_img, target_img):

            if pred_img.dim() == 3:
                pred_img = pred_img.unsqueeze(1)
            if target_img.dim() == 3:
                target_img = target_img.unsqueeze(1)

            mse_loss = F.mse_loss(pred_img, target_img)
            ssim_loss = 1.0 - ssim(pred_img, target_img, data_range=target_img.max(), size_average=True)
            symmetry_loss = F.mse_loss(pred_img.squeeze(1), pred_img.squeeze(1).transpose(-1, -2))
            distance_decay_loss = self.distance_decay_constraint(pred_img, target_img)
            sparsity_loss = self.sparsity_constraint(pred_img, target_img)
            local_structure_loss = self.local_structure_consistency(pred_img, target_img)

            total_loss = (0.3 * mse_loss + 0.4 * ssim_loss + 0.1 * symmetry_loss + 
                          0.1 * distance_decay_loss + 0.05 * sparsity_loss + 0.05 * local_structure_loss)

            return total_loss

        def distance_decay_constraint(self, pred, target):

            B, C, H, W = pred.shape
            device = pred.device

            if not hasattr(self, 'dist_matrix') or self.dist_matrix.shape[0] != H:
                i_indices = torch.arange(H, device=device).float().view(-1, 1)
                j_indices = torch.arange(W, device=device).float().view(1, -1)
                dist = torch.abs(i_indices - j_indices)
                dist.fill_diagonal_(1.0)
                self.dist_matrix = dist

            alpha = 1.0
            epsilon = 1e-8
            weight_matrix = 1.0 / (torch.pow(self.dist_matrix, alpha) + epsilon)
            weight_matrix = weight_matrix / torch.max(weight_matrix)

            weighted_loss = weight_matrix.unsqueeze(0).unsqueeze(0) * F.mse_loss(pred, target, reduction='none')
            return torch.mean(weighted_loss)

        def sparsity_constraint(self, pred, target):

            pred_nonzero_ratio = (pred > 0.01).float().mean()
            target_nonzero_ratio = (target > 0.01).float().mean()
            l1_reg = torch.mean(torch.abs(pred))
            ratio_loss = F.mse_loss(pred_nonzero_ratio, target_nonzero_ratio)
            return 0.1 * l1_reg + ratio_loss

        def local_structure_consistency(self, pred, target):

            kernel_size = 5
            padding = kernel_size // 2
            pred_local = F.avg_pool2d(pred, kernel_size, stride=1, padding=padding)
            target_local = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)
            return F.mse_loss(pred_local, target_local)

        def forward(self, scRNA_ori, HiC_ori, scRNA_vq, scRNA_semantic, scRNA_encoder_result,
                    hic_quantized_bottleneck, HiC_semantic_vq, hic_skips, hic_unet_module):
            if self.hic_unet_decoder is None:
                self.hic_unet_decoder = hic_unet_module.decode

            B = HiC_ori.size(0)
            hic_size = int(np.sqrt(HiC_ori.shape[1]))
            HiC_ori_matrix = HiC_ori.reshape(B, hic_size, hic_size)


            scRNA_recon_params = self.scRNA_decoder(scRNA_vq)
            scHiC_recon_result = self.hic_unet_decoder(hic_quantized_bottleneck, skips=hic_skips, cond_info=None)
            scHiC_recon_result = torch.clamp(scHiC_recon_result, min=0.0)


            if self.training:

                real_semantic_bottleneck = hic_quantized_bottleneck[:, :self.hic_embedding_dim, :, :]
                self.cross_modal_generator.update_target_statistics(real_semantic_bottleneck)

            cross_scRNA_params = self.scRNA_decoder(HiC_semantic_vq)
            combined_rna_info = torch.cat([scRNA_vq, scRNA_semantic], dim=1)

            Hb, Wb = hic_quantized_bottleneck.shape[2], hic_quantized_bottleneck.shape[3]
            gen_bottleneck, gen_skips = self.cross_modal_generator(combined_rna_info, B, scRNA_vq.device)

            semantic_spatial_expanded = scRNA_vq.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, Hb, Wb)
            generated_bottleneck = torch.cat([gen_bottleneck, semantic_spatial_expanded], dim=1)

            cross_scHiC = self.hic_unet_decoder(generated_bottleneck, skips=gen_skips, cond_info=combined_rna_info)
            cross_scHiC = torch.clamp(cross_scHiC, min=0.0)

            scRNA_recon_loss = self.rna_loss(scRNA_recon_params, scRNA_ori)
            scHiC_recon_loss = self.enhanced_hic_loss(scHiC_recon_result, HiC_ori_matrix)
            cross_loss_rna = self.rna_loss(cross_scRNA_params, scRNA_ori)
            cross_loss_HiC = self.enhanced_hic_loss(cross_scHiC, HiC_ori_matrix)

            scRNA_recon_result = to_count_mean(scRNA_recon_params)
            cross_scRNA = to_count_mean(cross_scRNA_params)

            latent_imitation_loss = 0.0

            return (scRNA_recon_loss, scHiC_recon_loss, scRNA_recon_result, scHiC_recon_result.squeeze(1),
                    cross_loss_rna, cross_loss_HiC, cross_scRNA, cross_scHiC.squeeze(1), latent_imitation_loss)


else:
    raise ValueError(f"Unsupported MIGO_RESOLUTION: {_RESOLUTION}")


