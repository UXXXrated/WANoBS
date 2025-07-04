{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import gradio as gr\
import os\
from run_wan import run_wan_generate, get_lora_list, purge_outputs\
\
# Where LoRAs and renders live\
UPLOAD_DIR = "/workspace/ComfyUI/models/Lora"\
OUTPUT_DIR = "/workspace/outputs"\
\
os.makedirs(UPLOAD_DIR, exist_ok=True)\
os.makedirs(OUTPUT_DIR, exist_ok=True)\
\
def upload_lora(files):\
    for file in files:\
        dest = os.path.join(UPLOAD_DIR, file.name)\
        with open(dest, "wb") as f:\
            f.write(file.read())\
    return get_lora_list()\
\
def generate_video(selected_loras, lora_weights, prompt, input_image, aspect_ratio, custom_width, custom_height):\
    if not selected_loras:\
        return "Please select at least one LoRA.", None\
\
    height = 720\
    if aspect_ratio == "Custom":\
        width = int(custom_width or 720)\
        height = int(custom_height or 720)\
    else:\
        ratios = \{"16:9": 16/9, "4:3": 4/3, "3:4": 3/4, "1:1": 1/1\}\
        ratio = ratios.get(aspect_ratio, 1)\
        width = int(height * ratio)\
\
    lora_pairs = list(zip(selected_loras, lora_weights[:len(selected_loras)]))\
\
    out_path = run_wan_generate(prompt, input_image, lora_pairs, width, height)\
    return "Generation complete!", out_path\
\
def list_outputs():\
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".mp4")]\
    files.sort()\
    return [os.path.join(OUTPUT_DIR, f) for f in files]\
\
with gr.Blocks() as demo:\
    gr.Markdown("# \uc0\u55356 \u57088  WAN 14B Video Generator")\
\
    with gr.Row():\
        uploader = gr.File(label="Upload LoRAs (.safetensors)", file_types=[".safetensors"], file_count="multiple")\
        upload_btn = gr.Button("Upload LoRA(s)")\
\
    lora_list = gr.CheckboxGroup(choices=get_lora_list(), label="Select LoRAs")\
    lora_weights = gr.Dataframe(headers=["Weight"], datatype="number", row_count=5, col_count=1)\
\
    upload_btn.click(upload_lora, inputs=uploader, outputs=lora_list)\
\
    prompt = gr.Textbox(label="Prompt (optional)")\
    input_image = gr.Image(type="filepath", label="Input Image (optional)")\
\
    aspect_ratio = gr.Dropdown(["16:9", "4:3", "3:4", "1:1", "Custom"], value="4:3", label="Aspect Ratio")\
    custom_width = gr.Number(label="Custom Width (if Custom)")\
    custom_height = gr.Number(label="Custom Height (if Custom)")\
\
    generate_btn = gr.Button("Generate")\
    status = gr.Textbox(label="Status")\
    output_video = gr.Video()\
    gallery = gr.Gallery(label="Renders").style(grid=[2], height="auto")\
    purge_btn = gr.Button("Purge All Renders")\
\
    generate_btn.click(\
        generate_video,\
        inputs=[lora_list, lora_weights, prompt, input_image, aspect_ratio, custom_width, custom_height],\
        outputs=[status, output_video]\
    )\
\
    generate_btn.click(list_outputs, outputs=gallery)\
    purge_btn.click(purge_outputs, outputs=gallery)\
\
demo.launch(server_name="0.0.0.0", server_port=7860)}