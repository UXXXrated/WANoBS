{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # \uc0\u55356 \u57088  WAN 14B Video Generator\
\
This is your drag & drop Gradio interface for WAN Video:\
- Upload LoRAs\
- Pick which LoRAs to apply + set weights\
- Optional prompt + optional image\
- Pick aspect ratio or custom dimensions\
- Generate video with auto-named output\
- See all renders in a gallery \uc0\u8594  download or delete as needed\
\
---\
\
## \uc0\u9989  RunPod Recommended Settings\
\
- **Expose HTTP Port:** 7860  \
- **Volume Disk:** 150 GB  \
- **Container Disk:** 50 GB\
\
---\
\
## \uc0\u55357 \u56960  Deploy\
\
1. Push this repo to GitHub.\
2. In RunPod, make a new Pod:\
   - Use this repo as the custom container\
   - Expose port `7860`\
   - Set disks as above\
3. Open: `https://YOURPODID-7860.proxy.runpod.net/`\
\
---\
\
## \uc0\u9881 \u65039  WAN Backend\
\
- The WAN logic lives in `run_wan.py` \uc0\u8594  `run_wan_generate()`.  \
- Replace the dummy stub with your real WAN load + inference code.\
\
---\
\
## \uc0\u55357 \u56770 \u65039  Output\
\
- Renders saved to `/workspace/outputs`  \
- Auto-named with timestamp + aspect ratio  \
- Only download the ones you want  \
- `Purge All` wipes them all.\
\
---\
\
## \uc0\u9989  Enjoy!\
\
This is your *one-click* WAN video lab \'97 no node graphs, no terminals.}