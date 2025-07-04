import os
from datetime import datetime
from diffusers import StableDiffusionPipeline
import torch

OUTPUT_DIR = "/workspace/outputs"
UPLOAD_DIR = "/workspace/ComfyUI/models/Lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load base Stable Diffusion pipeline (real model)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

def run_wan_generate(prompt, input_image, lora_pairs, width, height):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"wan_render_{timestamp}_{width}x{height}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Actually generate
    image = pipe(prompt).images[0]
    image = image.resize((width, height))
    image.save(output_path)

    return output_path

def get_lora_list():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".safetensors")]

def purge_outputs():
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".png"):
            os.remove(os.path.join(OUTPUT_DIR, f))
    return []
