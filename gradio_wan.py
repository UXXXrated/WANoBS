import os
import shutil
import gradio as gr
from run_wan import run_wan_generate

UPLOAD_DIR = "/workspace/uploads"
OUTPUT_DIR = "/workspace/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def submit(prompt, image, uploaded_files, width, height, weights):
    lora_paths = []
    lora_weights = []

    # Flatten weights from dataframe
    flat_weights = [w[0] if isinstance(w, list) else w for w in weights]

    for file_obj, weight in zip(uploaded_files, flat_weights):
        filename = os.path.basename(file_obj.name)
        dest_path = os.path.join(UPLOAD_DIR, filename)
        shutil.copyfile(file_obj.name, dest_path)

        lora_paths.append(dest_path)
        lora_weights.append(float(weight) if weight is not None else 1.0)

    # Save conditioning image if provided
    image_path = None
    if image:
        image_path = os.path.join(UPLOAD_DIR, "input_image.png")
        image.save(image_path)

    # Run WAN generation
    output_path = run_wan_generate(
        prompt,
        image_path,
        lora_paths,
        lora_weights,
        OUTPUT_DIR,
        int(width),
        int(height)
    )

    return output_path

with gr.Blocks() as iface:
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt")
        image = gr.Image(type="pil", label="Optional Image (Conditioning)")

    with gr.Row():
        uploaded_files = gr.File(
            label="Upload LoRA/Checkpoint Files",
            file_types=[".safetensors", ".ckpt", ".pt", ".lora"],
            file_count="multiple"
        )

    with gr.Row():
        weights = gr.Dataframe(
            headers=["LoRA Weight"],
            datatype=["number"],
            row_count=3,
            label="Set LoRA Weights (match file order)"
        )

    with gr.Row():
        width = gr.Number(value=512, label="Width")
        height = gr.Number(value=512, label="Height")

    with gr.Row():
        submit_btn = gr.Button("Generate Video")
        output_video = gr.Video(label="Output Video")

    submit_btn.click(
        fn=submit,
        inputs=[prompt, image, uploaded_files, width, height, weights],
        outputs=output_video
    )

iface.launch(server_name="0.0.0.0", server_port=7860)
