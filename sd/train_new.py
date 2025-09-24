"""Train Stable Diffusion from scratch on a compact demonstration dataset."""
from __future__ import annotations

from pathlib import Path

from train_model import TrainConfig, train_text_to_image

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "laion_aesthetic6"
METADATA_PATH = DATA_ROOT / "metadata.jsonl"
IMAGES_ROOT = DATA_ROOT / "images"
TOKENIZER_NAME = "openai/clip-vit-large-patch14"
OUTPUT_DIR = Path("experiments/toy_run")


def main() -> None:
    config = TrainConfig(
        output_dir=str(OUTPUT_DIR),
        tokenizer_path=TOKENIZER_NAME,
        metadata_path=str(METADATA_PATH),
        images_root=str(IMAGES_ROOT),
        pretrained_ckpt=None,
        batch_size=2,
        gradient_accumulation=4,
        learning_rate=5e-5,
        weight_decay=0.0,
        num_epochs=20,
        max_train_steps=5_000,
        save_every=500,
        train_text_encoder=True,
        train_vae=True,
        mixed_precision=False,
        resolution=256,
    )
    train_text_to_image(config)


if __name__ == "__main__":
    main()
