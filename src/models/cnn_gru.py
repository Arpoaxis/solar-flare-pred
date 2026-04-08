import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """
    CNN feature extractor for one timestep of solar active-region imagery.

    This backbone converts a single multi-channel image into a compact
    feature vector for downstream temporal modeling.

    Input shape:
        [B, 10, 256, 256]

    Output shape:
        [B, feat_dim]
    """

    def __init__(self, in_channels: int = 10, feat_dim: int = 128) -> None:
        """
        Initialize the CNN backbone.

        Args:
            in_channels (int): Number of input image channels.
            feat_dim (int): Dimension of the output feature vector.
        """
        super().__init__()

        # Convolutional feature extractor with progressive channel expansion
        # and spatial downsampling.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 256 -> 128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(64, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
        )

        # Global average pooling converts the final feature map into one
        # feature vector per sample.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the CNN backbone.

        Args:
            x (torch.Tensor): Input tensor with shape [B, 10, 256, 256].

        Returns:
            torch.Tensor: Feature tensor with shape [B, feat_dim].
        """
        x = self.features(x)        # [B, feat_dim, 32, 32]
        x = self.pool(x).flatten(1) # [B, feat_dim]
        return x


class CNNGRU(nn.Module):
    """
    CNN-GRU model for spatiotemporal solar flare prediction.

    The model extracts spatial features from each timestep using a CNN
    backbone, then models temporal evolution across timesteps with a GRU.

    Input shape:
        [B, T, C, H, W] = [B, 4, 10, 256, 256]

    Output shape:
        [B, 1]
    """

    def __init__(
        self,
        in_channels: int = 10,
        feat_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        """
        Initialize the CNN-GRU model.

        Args:
            in_channels (int): Number of input channels per timestep.
            feat_dim (int): Dimension of the CNN feature vector per timestep.
            hidden_dim (int): GRU hidden-state dimension.
            num_layers (int): Number of GRU layers.
        """
        super().__init__()

        self.backbone = CNNBackbone(
            in_channels=in_channels,
            feat_dim=feat_dim,
        )

        # The GRU processes one feature vector per timestep.
        # With batch_first=True, the expected input shape is [B, T, feat_dim].
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the CNN-GRU model.

        Processing steps:
            1. Reshape [B, T, C, H, W] -> [B*T, C, H, W]
            2. Extract per-timestep CNN features
            3. Reshape features to [B, T, feat_dim]
            4. Model temporal evolution with the GRU
            5. Use the final hidden state for binary classification

        Args:
            x (torch.Tensor): Input tensor with shape [B, 4, 10, 256, 256].

        Returns:
            torch.Tensor: Output logits with shape [B, 1].
        """
        batch_size, timesteps, channels, height, width = x.shape

        x = x.reshape(batch_size * timesteps, channels, height, width)
        features = self.backbone(x)                               # [B*T, feat_dim]
        features = features.reshape(batch_size, timesteps, -1)   # [B, T, feat_dim]

        _, hidden_states = self.gru(features)                     # [L, B, hidden_dim]
        last_hidden = hidden_states[-1]                           # [B, hidden_dim]

        return self.head(last_hidden)                             # [B, 1]