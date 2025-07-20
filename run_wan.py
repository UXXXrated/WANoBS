import os
import time
from PIL import Image
import torch
from diffusers import DiffusionPipeline
import imageio

# Load base or user-supplied WAN model
def load_wan_pipeline(model_path=None):
    try:
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
    except Exception as e:
        print(f"Model loading failed: {e}")
        print("Falling back to default WAN base model...")
        return DiffusionPipeline.from_pretrained(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            torch_dtype=torch.float16
        ).to("cuda")

# Load default at startup
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
    global wan_pipe

    # Check for model override file
    override_model_path = None
    for path in lora_paths:
        filename = os.path.basename(path).lower()
        if filename.endswith(".ckpt") or filename.endswith(".safetensors") or "model" in filename:
            override_model_path = path
            break

    # Try to reload model if override file is found
    if override_model_path:
        print(f"Reloading WAN pipeline from uploaded model: {override_model_path}")
        wan_pipe = load_wan_pipeline(override_model_path)

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
