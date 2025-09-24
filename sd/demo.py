import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import re
from pathlib import Path

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

## TEXT TO IMAGE

# prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
prompt = "A close-up shot of a futuristic red supercar, glossy reflections, carbon fiber details, cinematic lighting, ultra sharp, 8k resolution"
prompt = "A breathtaking landscape with snowy mountains, a crystal-clear lake, pine trees along the shore, under a sunset sky, ultra sharp, 8k resolution"
prompt = "A close-up portrait of a 25-year-old European female model, natural skin texture, cinematic lighting, ultra sharp, 8k resolution"
prompt = "A half-body portrait of a 25-year-old European female model, natural skin texture, fashionable outfit, cinematic lighting, ultra sharp, 8k resolution"
prompt = "Hyper-realistic 8K portrait of a young woman with wet ginger hair, glowing sun-kissed skin, coral blush, and a subtle lip bite. She wears a black heart choker. Golden hour light adds soft highlights, with shallow depth of field. Captured in 35mm film style on a Canon AE-1, with cinematic tones, grain."
prompt = 'Realistic 8K image of Daenerys Targaryen in a spider-inspired costume, standing on the edge of a tall city building at dusk. She looks back with a focused expression, city lights and sunset glow softly illuminating her face and curvy silhouette. Cinematic tones, shallow depth of field, intimate atmosphere.'


uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

input_image = None
# Comment to disable image to image

# 把 prompt 变成安全的文件名：只保留字母数字和少量符号，其它替换成 "_"
safe_prompt = re.sub(r'[^a-zA-Z0-9_\- ]', '', prompt)
# 把空格换成 "_"
safe_prompt = safe_prompt.strip().replace(" ", "_")
# 防止太长（Windows 文件名长度限制通常是 255）
safe_prompt = safe_prompt[:50]

image_path = fr"../images/{safe_prompt}.jpg"
# input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9

## SAMPLER

sampler = "ddpm"
num_inference_steps = 50
seed = 5

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# Combine the input image and the output image into a single image.
Image.fromarray(output_image).save(image_path)