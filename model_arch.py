# model_arch.py
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet  # ✅ Import EfficientNet

class SimpleEfficientTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1, dim=256, depth=4, heads=4, mlp_dim=512, dropout=0.1):
        super().__init__()

        assert image_size % patch_size == 0, 'Image must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 1280  # Output channels of EfficientNet-B0

        # EfficientNet-B0 backbone
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.backbone._fc = nn.Identity()  # Remove classifier

        # Patch projection
        self.to_patch_embedding = nn.Linear(self.patch_dim, dim)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # Extract EfficientNet features
        with torch.no_grad():
            feats = self.backbone.extract_features(img)  # [B, 1280, H/32, W/32]

        # Flatten spatial dims
        b, c, h, w = feats.shape
        x = feats.view(b, c, h * w).permute(0, 2, 1)  # [B, N, C]

        # Project to transformer dim
        x = self.to_patch_embedding(x)
        x += self.pos_embedding[:, :x.size(1)]
        x = self.dropout(x)

        # Transformer encoder
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        return self.mlp_head(x)
