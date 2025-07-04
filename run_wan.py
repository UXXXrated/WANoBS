import os
from datetime import datetime
from diffusers import StableVideoDiffusionPipeline
import torch
import imageio
from PIL import Image
import numpy as np

OUTPUT_DIR = "/workspace/outputs"
UPLOAD_DIR = "/workspace/ComfyUI/models/Lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load base Stable Video Diffusion pipeline
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16
).to("cuda")

def run_wan_generate(prompt, input_image, lora_pairs, width, height):
    # Use input image if given, else make blank white image for pure prompt generation
    if input_image:
        input_image = Image.open(input_image).convert("RGB").resize((width, height))
    else:
        input_image = Image.new("RGB", (width, height), (255, 255, 255))

    # Actually generate video frames
    result = pipe(prompt=prompt, image=input_image)
    frames = result.frames  # list of PIL Images

    # Save as mp4
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"wan_render_{timestamp}_{width}x{height}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)

    frame_list = [np.array(f) for f in frames]
    imageio.mimsave(output_path, frame_list, fps=8)  # 8 FPS

    return output_path

def get_lora_list():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".safetensors")]

def purge_outputs():
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".mp4"):
            os.remove(os.path.join(OUTPUT_DIR, f))
    return []
