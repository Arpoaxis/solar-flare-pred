import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    CNN-only baseline for binary solar flare prediction.

    This model consumes a single timestep with 10 input channels and
    produces one logit per sample for binary classification.

    Input shape:
        [B, 10, 256, 256]

    Output shape:
        [B, 1]
    """

    def __init__(self, in_channels: int = 10) -> None:
        """
        Initialize the CNN baseline.

        Args:
            in_channels (int): Number of input channels per image.
        """
        super().__init__()

        # Convolutional feature extractor with progressive channel expansion
        # and spatial downsampling after each block.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Global average pooling reduces the spatial feature map to one
        # feature vector per sample before the final linear classifier.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor with shape [B, 10, 256, 256].

        Returns:
            torch.Tensor: Output logits with shape [B, 1].
        """
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)