"""TCN--BiLSTM architecture used by the released Material A--E weights."""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig, get_material


class DilatedResidualBlock(nn.Module):
    """Length-preserving residual dilated-convolution block.

    The operation order and layer names intentionally match the training
    implementation so that its state dictionaries load without conversion.
    """

    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
        )
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()
        self.pw = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        self.proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform ``(batch, channels, time)`` while preserving time length."""

        y = self.pw(self.act(self.norm(self.conv(x))))
        return self.drop(y) + self.proj(x)


class MagnetSeq2Seq(nn.Module):
    """One-pass non-autoregressive estimator of a complete H sequence."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.stem = nn.Conv1d(config.in_ch, config.stem, kernel_size=5, padding=2)
        blocks = []
        channels = config.stem
        for dilation in config.tcn_dilations:
            blocks.append(
                DilatedResidualBlock(
                    channels, config.tcn_ch, dilation, config.dropout
                )
            )
            channels = config.tcn_ch
        self.tcn = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(
            channels,
            config.lstm_hidden,
            config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * config.lstm_hidden, config.head_hidden),
            nn.GELU(),
            nn.Linear(config.head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map material-specific input channels to normalized H."""

        features = self.tcn(self.stem(x.transpose(1, 2))).transpose(1, 2)
        features, _ = self.lstm(features)
        return self.head(features).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model(config: ModelConfig | str | None = None) -> MagnetSeq2Seq:
    """Construct a model from a config or a Material A--E identifier."""

    if isinstance(config, str):
        config = get_material(config).model
    return MagnetSeq2Seq(config or ModelConfig())


# Alias retained because the original training scripts used this spelling.
count_params = count_parameters
