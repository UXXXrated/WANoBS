import os
from datetime import datetime
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
import torch
from PIL import Image
import imageio

OUTPUT_DIR = "/workspace/outputs"
UPLOAD_DIR = "/workspace/ComfyUI/models/Lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load your pipelines once — adjust model IDs as needed.
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

img2vid_pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16
).to("cuda")

def run_wan_generate(prompt, input_image, lora_pairs, width, height):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ratio_tag = f"{width}x{height}"
    filename = f"wan_render_{timestamp}_{ratio_tag}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)

    if input_image is None:
        # TEXT TO IMAGE → IMAGE TO VIDEO
        image = text2img_pipe(prompt, height=height, width=width).images[0]
    else:
        # Use uploaded image as input
        image = Image.open(input_image).convert("RGB").resize((width, height))

    # IMAGE TO VIDEO FRAMES
    frames = img2vid_pipe(image=image, decode_chunk_size=8).frames[0]

    # Save frames as MP4
    imageio.mimsave(output_path, frames, fps=8)

    return output_path

def get_lora_list():
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".safetensors")]

def purge_outputs():
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".mp4"):
            os.remove(os.path.join(OUTPUT_DIR, f))
    return []
