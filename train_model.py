"""Reusable text-to-image training utilities built on top of PyTorch."""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

from model_define import (
    StableDiffusionModels,
    create_untrained_components,
    load_pretrained_components,
)

# ``sd`` lives as a simple folder, therefore we add it to the module
# search path before importing its utilities.
SD_DIR = Path(__file__).resolve().parent / "sd"
if str(SD_DIR) not in sys.path:
    sys.path.append(str(SD_DIR))

from ddpm import DDPMSampler  # type: ignore  # pylint: disable=wrong-import-position
from pipeline import get_time_embedding  # type: ignore  # pylint: disable=wrong-import-position


class ImageTextDataset(Dataset):
    """Dataset that expects ``image_path`` and ``text`` columns in a CSV file."""

    def __init__(self, metadata_path: str, images_root: str, resolution: int = 512):
        super().__init__()
        self.metadata_path = Path(metadata_path)
        self.images_root = Path(images_root)
        self.resolution = resolution
        self.entries = self._load_metadata()

    def _load_metadata(self) -> List[dict]:
        rows: List[dict] = []
        with self.metadata_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_path = row.get("image_path")
                text = row.get("text")
                if not image_path or not text:
                    continue
                rows.append({"image_path": image_path, "text": text})
        if not rows:
            raise ValueError(
                f"No usable rows found in {self.metadata_path}. Expected 'image_path' and 'text' columns."
            )
        return rows

    def __len__(self) -> int:  # pragma: no cover - trivial container
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        image_path = self.images_root / entry["image_path"]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize((self.resolution, self.resolution), resample=Image.BICUBIC)
            np_img = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_img).permute(2, 0, 1)
        tensor = tensor * 2.0 - 1.0  # Normalize to [-1, 1]
        return {
            "pixel_values": tensor,
            "text": entry["text"],
        }


@dataclass
class TrainConfig:
    """Hyper-parameters shared by the pretraining and scratch scripts."""

    output_dir: str
    tokenizer_path: str
    metadata_path: str
    images_root: str
    pretrained_ckpt: Optional[str] = None
    device: str = "cuda"
    resolution: int = 512
    batch_size: int = 4
    gradient_accumulation: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    max_train_steps: int = 10_000
    num_epochs: int = 1
    num_workers: int = 4
    mixed_precision: bool = True
    save_every: int = 1_000
    train_text_encoder: bool = False
    train_vae: bool = False


def _prepare_models(config: TrainConfig) -> StableDiffusionModels:
    if config.pretrained_ckpt:
        models = load_pretrained_components(config.pretrained_ckpt, config.device)
    else:
        models = create_untrained_components(config.device)
    models.text_encoder.train(config.train_text_encoder)
    models.vae_encoder.train(config.train_vae)
    models.vae_decoder.train(config.train_vae)
    models.unet.train(True)

    if not config.train_text_encoder:
        for param in models.text_encoder.parameters():
            param.requires_grad_(False)
    if not config.train_vae:
        for param in models.vae_encoder.parameters():
            param.requires_grad_(False)
            param.grad = None
        for param in models.vae_decoder.parameters():
            param.requires_grad_(False)
            param.grad = None
    return models


def _gather_trainable_parameters(models: StableDiffusionModels) -> Iterable[nn.Parameter]:
    params: List[nn.Parameter] = []
    for module in (models.text_encoder, models.vae_encoder, models.vae_decoder, models.unet):
        for param in module.parameters():
            if param.requires_grad:
                params.append(param)
    return params


def _encode_text(
    tokenizer: CLIPTokenizer, prompts: List[str], device: torch.device, text_encoder: nn.Module
) -> torch.Tensor:
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).input_ids
    tokens = tokens.to(device)
    return text_encoder(tokens)


def _encode_images(pixels: torch.Tensor, vae_encoder: nn.Module, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(
        (pixels.shape[0], 4, pixels.shape[2] // 8, pixels.shape[3] // 8),
        generator=generator,
        device=pixels.device,
    )
    return vae_encoder(pixels, noise)


def train_text_to_image(config: TrainConfig) -> None:
    device = torch.device(config.device)
    dataset = ImageTextDataset(config.metadata_path, config.images_root, config.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    tokenizer = CLIPTokenizer.from_pretrained(config.tokenizer_path)
    models = _prepare_models(config)
    models.to(device)

    params = _gather_trainable_parameters(models)
    optimizer = AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision and device.type == "cuda")
    generator = torch.Generator(device=device)
    sampler = DDPMSampler(generator)

    global_step = 0
    total_steps = min(config.max_train_steps, config.num_epochs * math.ceil(len(dataset) / config.batch_size))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(total=total_steps, desc="training")

    for epoch in range(config.num_epochs):
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            prompts: List[str] = batch["text"]  # type: ignore[assignment]

            with torch.no_grad():
                text_embeddings = _encode_text(tokenizer, prompts, device, models.text_encoder)
                latents = _encode_images(pixel_values, models.vae_encoder, generator)
                noise = torch.randn_like(latents, generator=generator)
                timesteps = torch.randint(
                    0,
                    sampler.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                    generator=generator,
                ).long()
                noisy_latents = sampler.add_noise(latents, timesteps)

            time_embeddings = get_time_embedding(timesteps).to(device)

            with torch.cuda.amp.autocast(enabled=config.mixed_precision and device.type == "cuda"):
                noise_pred = models.unet(noisy_latents, text_embeddings, time_embeddings)
                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss / config.gradient_accumulation).backward()

            if (global_step + 1) % config.gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % config.save_every == 0:
                _save_checkpoint(models, optimizer, global_step, output_dir)

            if global_step >= config.max_train_steps:
                break

        if global_step >= config.max_train_steps:
            break

    progress_bar.close()
    _save_checkpoint(models, optimizer, global_step, output_dir)


def _save_checkpoint(models: StableDiffusionModels, optimizer: AdamW, step: int, output_dir: Path) -> None:
    checkpoint = {
        "text_encoder": models.text_encoder.state_dict(),
        "vae_encoder": models.vae_encoder.state_dict(),
        "vae_decoder": models.vae_decoder.state_dict(),
        "unet": models.unet.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
    }
    torch.save(checkpoint, output_dir / f"checkpoint-{step:06d}.pt")


__all__ = ["ImageTextDataset", "TrainConfig", "train_text_to_image"]
