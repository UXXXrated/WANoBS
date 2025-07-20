import os
import time
from PIL import Image
import torch
from diffusers import DiffusionPipeline
import imageio

# Load WAN 2.1 T2V 14B model (public, working)
print("Loading WAN 2.1 T2V 14B...")
wan_pipe = DiffusionPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    torch_dtype=torch.float16
).to("cuda")
print("Model loaded.")

def run_wan_generate(
    prompt: str,
    image_path: str,
    lora_paths: list,
    lora_weights: list,
    output_dir: str,
    width: int,
    height: int
) -> str:
    # Conditioning image is not supported by this model (T2V only)
    init_image = None  # kept for compatibility, ignored here

    # LoRA loading (not supported by this model — placeholder)
    if lora_paths:
        for path, weight in zip(lora_paths, lora_weights):
            print(f"Skipping LoRA {path} — not supported in this model.")

    print(f"Generating video with prompt: {prompt}")
    try:
        result = wan_pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=24
        )
        video_frames = result.get("frames") if isinstance(result, dict) else result.frames
    except Exception as e:
        print(f"Generation failed: {e}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    output_path = os.path.join(output_dir, f"wan_{timestamp}.mp4")

    try:
        imageio.mimsave(output_path, video_frames, fps=8)
        print(f"Video saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Could not save video: {e}")
        return None
