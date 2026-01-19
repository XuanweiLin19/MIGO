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
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        drop=0.1,
        input_dim_rna=None,
        input_dim_atac=None,
    ):
        super().__init__()
        input_dim_rna = input_dim_rna or embedding_dim
        input_dim_atac = input_dim_atac or embedding_dim

        self.CLIP_linear_rna = ProjectionHead(input_dim=input_dim_rna, output_dim=projection_dim, drop=drop)
        self.CLIP_linear_atac = ProjectionHead(input_dim=input_dim_atac, output_dim=projection_dim, drop=drop)
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

        out_atac = torch.nn.functional.softmax(logits_per_atac, dim=-1)
        out_rna = torch.nn.functional.softmax(logits_per_rna, dim=-1)

        label = torch.arange(out_rna.shape[0]).to(device)
        loss_cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

        loss1 = loss_cross_entropy(logits_per_rna, label)
        loss2 = loss_cross_entropy(logits_per_atac, label)
        loss = (loss1 + loss2) / 2

        return loss, rna_embedding_norm, atac_embedding_norm
