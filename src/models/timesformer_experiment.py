from typing import Optional

import torch.nn as nn
from transformers import AutoConfig, TimesformerForVideoClassification


class TimeSformerExperiment(nn.Module):
    """Wrapper for fast architecture experiments on top of HF TimeSformer."""

    def __init__(
        self,
        pretrained_name: str,
        num_frames: int,
        image_size: int,
        num_labels: Optional[int],
        dropout: float,
        attention_dropout: float,
        freeze_backbone: bool,
        use_custom_head: bool,
        custom_head_hidden_dim: int,
    ) -> None:
        super().__init__()

        # Build model config before loading weights.
        config = AutoConfig.from_pretrained(pretrained_name)
        config.num_frames = num_frames
        config.image_size = image_size
        if num_labels is not None:
            config.num_labels = num_labels
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = attention_dropout

        # Load base model and allow classifier size adaptation.
        self.model = TimesformerForVideoClassification.from_pretrained(
            pretrained_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        if use_custom_head:
            # Replace the original classifier with a small MLP head.
            out_features = num_labels if num_labels is not None else config.num_labels
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, custom_head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(custom_head_hidden_dim, out_features),
            )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        # Freeze TimeSformer blocks while keeping classifier trainable.
        for param in self.model.timesformer.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        # Restore gradient update for all backbone params.
        for param in self.model.timesformer.parameters():
            param.requires_grad = True

    def replace_classifier(self, new_head: nn.Module) -> None:
        # Hot-swap custom head for experiments.
        self.model.classifier = new_head

    def forward(self, pixel_values, labels: Optional[object] = None):
        # Add custom experiment logic around this call as needed.
        return self.model(pixel_values=pixel_values, labels=labels)
