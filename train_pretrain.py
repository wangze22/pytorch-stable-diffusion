"""Fine-tune Stable Diffusion on a large professional dataset."""
from __future__ import annotations

from pathlib import Path

from train_model import TrainConfig, train_text_to_image

DATA_ROOT = Path("/datasets/laion-aesthetic-v2")
METADATA_PATH = DATA_ROOT / "metadata/laion-aesthetic-6.5plus.csv"
IMAGES_ROOT = DATA_ROOT / "images"
PRETRAINED_WEIGHTS = Path("/path/to/v1-5-pruned-emaonly.ckpt")
TOKENIZER_NAME = "openai/clip-vit-large-patch14"
OUTPUT_DIR = Path("experiments/laion_finetune")


def main() -> None:
    config = TrainConfig(
        output_dir=str(OUTPUT_DIR),
        tokenizer_path=TOKENIZER_NAME,
        metadata_path=str(METADATA_PATH),
        images_root=str(IMAGES_ROOT),
        pretrained_ckpt=str(PRETRAINED_WEIGHTS),
        batch_size=6,
        gradient_accumulation=2,
        learning_rate=1e-5,
        weight_decay=0.01,
        num_epochs=4,
        max_train_steps=40_000,
        save_every=2_000,
        train_text_encoder=True,
        train_vae=False,
        mixed_precision=True,
    )
    train_text_to_image(config)


if __name__ == "__main__":
    main()
