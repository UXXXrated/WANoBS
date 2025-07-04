# 🌀 WAN 14B Video Generator

This is your drag & drop Gradio interface for WAN Video:  
- Upload LoRAs  
- Pick which LoRAs to apply + set weights  
- Optional prompt + optional image  
- Pick aspect ratio or custom dimensions  
- Generate video with auto-named output  
- See all renders in a gallery → download or delete as needed

---

## ✅ RunPod Recommended Settings

- **Expose HTTP Port:** 7860  
- **Volume Disk:** 150 GB  
- **Container Disk:** 50 GB

---

## 🚀 Deploy

1. Push this repo to GitHub.  
2. In RunPod, make a new Pod:
   - Use this repo as the custom container
   - Expose port `7860`
   - Set disks as above  
3. Open: `https://YOURPODID-7860.proxy.runpod.net/`

---

## ⚙️ WAN Backend

- The WAN logic lives in `run_wan.py` → `run_wan_generate()`.  
- Replace the dummy stub with your real WAN load + inference code.

---

## 🗂️ Output

- Renders saved to `/workspace/outputs`  
- Auto-named with timestamp + aspect ratio  
- Only download the ones you want  
- `Purge All` wipes them all.

---

## ✅ Enjoy!

This is your **one-click** WAN video lab — no node graphs, no terminals.
