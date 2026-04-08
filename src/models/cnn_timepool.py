import torch
import torch.nn as nn


class CNNTimeMeanPool(nn.Module):
    """
    CNN baseline with temporal mean pooling for solar flare prediction.

    This model extracts a feature vector from each timestep using a shared
    CNN encoder, then averages those feature vectors across time before
    applying a final linear classification head.

    Input shape:
        [B, T, C, H, W] = [B, 4, 10, 256, 256]

    Output shape:
        [B, 1]
    """

    def __init__(self, in_channels: int = 10, feat_dim: int = 128) -> None:
        """
        Initialize the CNN time-mean-pooling model.

        Args:
            in_channels (int): Number of input channels per timestep.
            feat_dim (int): Dimension of the per-timestep feature vector.
        """
        super().__init__()

        # Shared CNN encoder applied independently to each timestep.
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

        # Global average pooling converts each timestep feature map into one
        # feature vector of length feat_dim.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the temporal mean-pooling model.

        Processing steps:
            1. Reshape [B, T, C, H, W] -> [B*T, C, H, W]
            2. Extract CNN features for each timestep
            3. Pool to one feature vector per timestep
            4. Reshape to [B, T, feat_dim]
            5. Average over time
            6. Apply the classification head

        Args:
            x (torch.Tensor): Input tensor with shape [B, 4, 10, 256, 256].

        Returns:
            torch.Tensor: Output logits with shape [B, 1].
        """
        batch_size, timesteps, channels, height, width = x.shape

        x = x.reshape(batch_size * timesteps, channels, height, width)
        x = self.features(x)                                  # [B*T, feat_dim, 32, 32]
        x = self.pool(x).flatten(1)                           # [B*T, feat_dim]

        x = x.reshape(batch_size, timesteps, -1)              # [B, T, feat_dim]
        x = x.mean(dim=1)                                     # [B, feat_dim]

        return self.head(x)                                   # [B, 1]