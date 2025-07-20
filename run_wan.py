import os
import time
from PIL import Image
import torch
from diffusers import DiffusionPipeline
import imageio

# Load WAN-14B once
print("Loading WAN-14B base model...")
wan_pipe = DiffusionPipeline.from_pretrained(
    "ali-vilab/WAN-14B",
    torch_dtype=torch.float16
).to("cuda")
print("WAN-14B loaded.")

def run_wan_generate(
    prompt: str,
    image_path: str,
    lora_paths: list,
    lora_weights: list,
    output_dir: str,
    width: int,
    height: int
) -> str:
    # Load conditioning image (optional)
    init_image = None
    if image_path:
        try:
            init_image = Image.open(image_path).convert("RGB")
            init_image = init_image.resize((width, height))
        except Exception as e:
            print(f"Could not load image: {e}")
            init_image = None

    # Load and apply LoRAs (optional)
    if lora_paths:
        for path, weight in zip(lora_paths, lora_weights):
            if path and os.path.exists(path):
                try:
                    wan_pipe.load_lora_weights(path, weight=weight)  # Adjust if needed
                    print(f"Loaded LoRA: {path} (weight {weight})")
                except Exception as e:
                    print(f"Failed to load LoRA {path}: {e}")

    # Run WAN-14B generation
    print(f"Generating video with prompt: {prompt}")
    try:
        result = wan_pipe(
            prompt=prompt,
            image=init_image,
            width=width,
            height=height,
            num_frames=24
        )
        video_frames = result.frames  # Adjust if different attribute
    except Exception as e:
        print(f"Generation failed: {e}")
        return None

    # Save as .mp4 video
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
