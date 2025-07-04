import os
from datetime import datetime
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
import torch

OUTPUT_DIR = "/workspace/outputs"
UPLOAD_DIR = "/workspace/ComfyUI/models/Lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load base SD pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

def apply_lora(lora_path):
    print(f"Applying LoRA: {lora_path}")
    lora_state = load_file(lora_path)
    unet = pipe.unet

    # Simple example: apply weights if keys match (basic)
    for key in lora_state:
        if key in unet.state_dict():
            unet.state_dict()[key].copy_(lora_state[key])

def run_wan_generate(prompt, input_image, lora_pairs, width, height):
    if lora_pairs:
        for lora in lora_pairs:
            lora_path = os.path.join(UPLOAD_DIR, lora)
            apply_lora(lora_path)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"wan_render_{timestamp}_{width}x{height}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)

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
