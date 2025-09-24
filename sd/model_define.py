"""High level model builders for the Stable Diffusion implementation.

This module gathers the core neural networks defined under :mod:`sd`
and exposes a beginner friendly API that keeps the class names close
to the original paper:

``CLIPTextEncoder``
    Text encoder that turns token ids into contextual embeddings.
``AutoencoderKL``
    Variational autoencoder used to move between RGB pixels and the
    latent space (4 channels at 64x64 when training 512x512 images).
``UNetModel``
    The denoising U-Net that predicts the noise component during the
    diffusion process.

The helpers below simply instantiate the PyTorch modules from this
repository and optionally move them to the desired device.  Keeping the
logic in a separate file makes it straightforward to reuse when writing
training or inference scripts from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict

import torch
from clip import CLIP  # type: ignore  # pylint: disable=wrong-import-position
from decoder import VAE_Decoder  # type: ignore  # pylint: disable=wrong-import-position
from diffusion import Diffusion  # type: ignore  # pylint: disable=wrong-import-position
from encoder import VAE_Encoder  # type: ignore  # pylint: disable=wrong-import-position
import model_converter  # type: ignore  # pylint: disable=wrong-import-position


@dataclass
class StableDiffusionModels:
    """Container that holds every neural network used in the pipeline."""

    text_encoder: CLIP
    vae_encoder: VAE_Encoder
    vae_decoder: VAE_Decoder
    unet: Diffusion

    def to(self, device: torch.device | str) -> "StableDiffusionModels":
        """Move all models to the provided device and return ``self``."""
        self.text_encoder.to(device)
        self.vae_encoder.to(device)
        self.vae_decoder.to(device)
        self.unet.to(device)
        return self

    def as_dict(self) -> Dict[str, torch.nn.Module]:
        """Return the mapping expected by :func:`sd.pipeline.generate`."""
        return {
            "clip": self.text_encoder,
            "encoder": self.vae_encoder,
            "decoder": self.vae_decoder,
            "diffusion": self.unet,
        }


def create_untrained_components(device: torch.device | str | None = None) -> StableDiffusionModels:
    """Instantiate brand-new (randomly initialised) model components."""
    models = StableDiffusionModels(
        text_encoder=CLIP(),
        vae_encoder=VAE_Encoder(),
        vae_decoder=VAE_Decoder(),
        unet=Diffusion(),
    )
    if device is not None:
        models.to(device)
    return models


def load_pretrained_components(
    ckpt_path: str,
    device: torch.device | str = "cuda"
) -> StableDiffusionModels:
    """Load the official Stable Diffusion v1.x weights.

    Parameters
    ----------
    ckpt_path:
        Path to the original ``.ckpt`` file (the same format released by
        CompVis / Stability AI).  The helper relies on
        :mod:`sd.model_converter` to adapt the parameter names before
        loading them into this repository's modules.
    device:
        Device that should hold the resulting models.  ``"cuda"`` is used
        by default because the weights do not fit in CPU memory once the
        training loop starts.
    """
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)
    models = create_untrained_components(device)
    models.text_encoder.load_state_dict(state_dict["clip"], strict=True)
    models.vae_encoder.load_state_dict(state_dict["encoder"], strict=True)
    models.vae_decoder.load_state_dict(state_dict["decoder"], strict=True)
    models.unet.load_state_dict(state_dict["diffusion"], strict=True)
    return models


__all__ = [
    "StableDiffusionModels",
    "create_untrained_components",
    "load_pretrained_components",
]
