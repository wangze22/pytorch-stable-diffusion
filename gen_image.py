"""Generate a handful of sample images using pretrained Stable Diffusion weights."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPTokenizer

from model_define import load_pretrained_components

BASE_DIR = Path(__file__).resolve().parent
SD_DIR = BASE_DIR / "sd"
if str(SD_DIR) not in sys.path:
    sys.path.append(str(SD_DIR))

from pipeline import generate  # type: ignore  # pylint: disable=wrong-import-position

WEIGHTS_PATH = Path("./data/v1-5-pruned-emaonly.ckpt")
TOKENIZER_NAME = "openai/clip-vit-large-patch14"
OUTPUT_DIR = Path("generated_samples")

PROMPTS = [
    "Ultra detailed studio portrait of a jazz saxophonist, 85mm photograph, warm rim lighting",
    "Impressionist oil painting of a rainy Paris boulevard at dusk, soft brush strokes",
    "Golden hour landscape photograph of an alpine lake with snow-capped mountains and pine forests",
    "Macro photograph of dewy succulent leaves, shallow depth of field, glistening bokeh",
    "Vintage still life of a ceramic teapot with citrus fruit and velvet tablecloth, dramatic chiaroscuro",
]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_pretrained_components(str(WEIGHTS_PATH), device)
    tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_NAME)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, prompt in enumerate(PROMPTS, start=1):
        image_array = generate(
            prompt=prompt,
            uncond_prompt="",
            models=models.as_dict(),
            n_inference_steps=50,
            seed=idx * 12345,
            device=device,
            tokenizer=tokenizer,
        )
        image = Image.fromarray(image_array)
        image.save(OUTPUT_DIR / f"sample_{idx:02d}.png")
        print(f"Saved sample_{idx:02d}.png for prompt: {prompt}")


if __name__ == "__main__":
    main()
