import os
from datetime import datetime
import torch
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline

# === Config ===
OUTPUT_DIR = "/workspace/outputs"
UPLOAD_DIR = "/workspace/ComfyUI/models/Lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Load pipelines once ===
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

svid_pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16
).to("cuda")

# === Main function ===
def run_wan_generate(prompt, input_image, lora_pairs, width, height):
    """
    Makes a short video from:
    - Prompt only (makes an image, then makes video)
    - OR image + prompt (refine given image)
    """
    # If no input image, do text→image first
    if input_image is None:
        print("[WAN] No image given → making base image from prompt...")
        image = sd_pipe(prompt, width=width, height=height).images[0]
    else:
        print("[WAN] Using uploaded image.")
        image = input_image  # your uploaded image

    # Video generation: image → frames
    print("[WAN] Making video frames from image...")
    frames = svid_pipe(image).frames

    # === Save video ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ratio_tag = f"{width}x{height}"
    filename = f"wan_render_{timestamp}_{ratio_tag}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Save MP4 (this is an example, adapt as needed!)
    import imageio
    imageio.mimsave(output_path, frames, fps=8)

    retur
