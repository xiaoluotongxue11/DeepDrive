"""
Policy network model for DeepDrive ContractToControl.

Architecture:
  - Visual encoder (ResNet-based) for camera input
  - State encoder for speed/waypoints/sensors
  - Fusion module combining visual and state features
  - Policy head outputting [steering, throttle/brake]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class VisualEncoder(nn.Module):
    """Encodes camera observations into a feature vector."""

    def __init__(self, obs_dim: int, pretrained: bool = True):
        super().__init__()
        # Use pretrained ResNet-18 backbone (lighter than ResNet-50)
        import torchvision.models as models
        backbone = models.resnet18(pretrained=pretrained)
        # Remove final classification layer
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Projection to desired feature dim
        self.proj = nn.Linear(512, obs_dim)
        self.norm = nn.LayerNorm(obs_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = self.pool(feat).flatten(1)
        feat = self.norm(self.proj(feat))
        return feat


class StateEncoder(nn.Module):
    """Encodes non-visual state (speed, waypoints, etc.) into features."""

    def __init__(self, state_dim: int, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, obs_dim),
            nn.LayerNorm(obs_dim),
            nn.ReLU(inplace=True),
            nn.Linear(obs_dim, obs_dim),
            nn.LayerNorm(obs_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    """
    Main policy network for DeepDrive tasks.

    Fuses visual and state observations and outputs continuous control actions.
    Supports both task1 (simple) and task3 (complex) through configuration.
    """

    def __init__(
        self,
        obs_dim: int = 256,
        hidden_dim: int = 512,
        action_dim: int = 2,
        num_layers: int = 3,
        dropout: float = 0.1,
        state_dim: int = 24,  # speed + waypoints + optional sensors
        pretrained_encoder: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Encoders
        self.visual_encoder = VisualEncoder(obs_dim, pretrained=pretrained_encoder)
        self.state_encoder = StateEncoder(state_dim, obs_dim)

        # Attention-based fusion (better than simple concat for multi-modal input)
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=obs_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Policy MLP layers
        layers = []
        in_dim = obs_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim // 2
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        self.policy_mlp = nn.Sequential(*layers)

        # Action head: [steering, throttle/brake]
        self.action_head = nn.Linear(hidden_dim // 2, action_dim)

        # Value head for auxiliary value estimation (improves training stability)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize action and value heads with small weights for stable start."""
        nn.init.uniform_(self.action_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.action_head.bias)
        nn.init.uniform_(self.value_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.value_head.bias)

    def encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observations into fused feature vector."""
        visual_feat = self.visual_encoder(obs["image"])       # [B, obs_dim]
        state_feat = self.state_encoder(obs["state"])         # [B, obs_dim]

        # Stack as sequence for attention-based fusion [B, 2, obs_dim]
        seq = torch.stack([visual_feat, state_feat], dim=1)
        fused, _ = self.fusion_attn(seq, seq, seq)
        # Pool fused sequence
        return fused.mean(dim=1)                              # [B, obs_dim]

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: Dictionary with keys:
                - "image": [B, 3, H, W] camera input
                - "state": [B, state_dim] speed/waypoints

        Returns:
            actions: [B, action_dim] continuous control actions
            values:  [B, 1] value estimates
        """
        feat = self.encode(obs)
        hidden = self.policy_mlp(feat)

        actions = torch.tanh(self.action_head(hidden))  # Bounded in [-1, 1]
        values = self.value_head(hidden)
        return actions, values
