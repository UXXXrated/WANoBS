import os
import gradio as gr
from run_wan import run_wan_generate

UPLOAD_DIR = "./uploads"
OUTPUT_DIR = "./outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def submit(prompt, image, uploaded_files, width, height, weights):
    lora_paths = []
    lora_weights = []

    # Save uploaded files
    for file_obj, weight in zip(uploaded_files, weights):
        file_path = os.path.join(UPLOAD_DIR, os.path.basename(file_obj.name))
        with open(file_path, "wb") as f:
            f.write(file_obj.read())

        lora_paths.append(file_path)
        lora_weights.append(weight)

    # Save image if provided
    image_path = None
    if image:
        image_path = os.path.join(UPLOAD_DIR, "input_image.png")
        image.save(image_path)

    # Run generation
    output_path = run_wan_generate(
        prompt,
        image_path,
        lora_paths,
        lora_weights,
        OUTPUT_DIR,
        width,
        height
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
        weights = gr.Dataframe(headers=["LoRA Weight"], datatype=["number"], row_count=3, label="Set Weights (Match File Order)")

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
