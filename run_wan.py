import os
import time
from PIL import Image
import torch
from diffusers import DiffusionPipeline
import imageio

# Define a reusable loader for base or user-supplied model
def load_wan_pipeline(model_path=None):
    if model_path:
        print(f"Loading user model from: {model_path}")
        return DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        print("Loading default WAN base model...")
        return DiffusionPipeline.from_pretrained(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            torch_dtype=torch.float16
        ).to("cuda")

# Load default model at startup
wan_pipe = load_wan_pipeline()

def run_wan_generate(
    prompt: str,
    image_path: str,
    lora_paths: list,
    lora_weights: list,
    output_dir: str,
    width: int,
    height: int
) -> str:
    # We'll add dynamic model swapping in Step 2
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
