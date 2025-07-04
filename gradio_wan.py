import gradio as gr
from run_wan import run_wan_generate, get_lora_list, purge_outputs

ASPECT_RATIOS = {
    "1280x960 (4:3)": (1280, 960),
    "720x960 (3:4)": (720, 960),
    "720x720 (1:1)": (720, 720)
}

with gr.Blocks() as demo:
    gr.Markdown("# WAN Video Generator")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="Describe what you want to render...")
        input_image = gr.Image(label="Optional Image", type="filepath")

    lora_files = gr.CheckboxGroup(choices=get_lora_list(), label="Select LoRA Files")

    aspect_ratio = gr.Radio(choices=list(ASPECT_RATIOS.keys()), value="1280x960 (4:3)", label="Aspect Ratio")

    run_button = gr.Button("Run WAN")
    clear_button = gr.Button("Clear All Renders")

    progress = gr.Textbox(label="Status")
    outputs = gr.File(label="Rendered Image")

    def generate(prompt, input_image, lora_files, aspect_ratio):
        width, height = ASPECT_RATIOS[aspect_ratio]
        progress_text = "Generating..."
        output_path = run_wan_generate(prompt, input_image, lora_files, width, height)
        progress_text = "Done!"
        return progress_text, output_path

    def clear():
        purge_outputs()
        return "", None

    run_button.click(
        fn=generate,
        inputs=[prompt, input_image, lora_files, aspect_ratio],
        outputs=[progress, outputs]
    )

    clear_button.click(
        fn=clear,
        inputs=[],
        outputs=[progress, outputs]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
