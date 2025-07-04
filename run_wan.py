{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
from datetime import datetime\
\
OUTPUT_DIR = "/workspace/outputs"\
UPLOAD_DIR = "/workspace/ComfyUI/models/Lora"\
os.makedirs(OUTPUT_DIR, exist_ok=True)\
\
def run_wan_generate(prompt, input_image, lora_pairs, width, height):\
    """\
    This is where you plug in your real WAN logic.\
    Right now it just makes a dummy file to prove it works.\
    """\
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")\
    ratio_tag = f"\{width\}x\{height\}"\
    filename = f"wan_render_\{timestamp\}_\{ratio_tag\}.mp4"\
    output_path = os.path.join(OUTPUT_DIR, filename)\
\
    with open(output_path, "wb") as f:\
        f.write(b"")  # TODO: Replace with real video data!\
\
    return output_path\
\
def get_lora_list():\
    os.makedirs(UPLOAD_DIR, exist_ok=True)\
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".safetensors")]\
\
def purge_outputs():\
    for f in os.listdir(OUTPUT_DIR):\
        if f.endswith(".mp4"):\
            os.remove(os.path.join(OUTPUT_DIR, f))\
    return []}