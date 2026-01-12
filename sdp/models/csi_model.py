import torch.nn as nn

class CSIModel(nn.Module):
    def __init__(self, num_classes=10, base_channels=32, latent_dim=128):
        super().__init__()

        # Spatial Encoder: cope with dimension F and A
        # input shape (B*T, 1, F, A)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels*2, 3, groups=base_channels),
            nn.Conv2d(base_channels*2, base_channels*2, 1),
            nn.AdaptiveAvgPool2d((4, 4)), # -> (B*T, base_channels*2, 4, 4)
            nn.Flatten() # -> (B*T, base_channels*2 * 16)
        )

        # Temporal Processor: deal with dimension T
        # input shape: [B, T, latent_dim]
        self.temporal_processor = nn.Sequential(
            nn.Linear(base_channels*2*16, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=4,
                dim_feedforward=latent_dim*2,
                dropout=0.2,
                batch_first=True
            )
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        self.output_layer = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        B, T, F, A = x.shape


        # shape: [B, T, F, A] -> [B*T, 1, F, A] '1' is the channel
        spatial_input = x.view(B*T, 1, F, A)

        # spatial_feat shape: [B*T, base_channels*2 * 16]
        spatial_feat = self.spatial_encoder(spatial_input)
        
        # shape: [B, T, base_channels*2 * 16]
        spatial_feat = spatial_feat.view(B, T, -1)

        # temporal_feat shape: [B, T, latent_dim]
        temporal_feat = self.temporal_processor(spatial_feat)
        
        # transpose(1, 2) -> shape: [B, latent_dim, T]
        # adaptive_pool -> shape: [B, latent_dim, 1]
        pooled = self.adaptive_pool(temporal_feat.transpose(1, 2))

        # flatten -> shape: [B, latent_dim]
        features = self.flatten(pooled)
        
        return self.output_layer(features)