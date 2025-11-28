import os
import json
import gradio as gr
from pathlib import Path

# Load models from JSON files, create default if missing
def load_models():
    models_file = "models.json"
    defaults_file = "models.json.defaults"
    
    # Try to load custom models file first
    if os.path.exists(models_file):
        with open(models_file, 'r') as f:
            return json.load(f)
    # Fallback to defaults and create the file
    elif os.path.exists(defaults_file):
        with open(defaults_file, 'r') as f:
            default_models = json.load(f)
        
        # Create models.json with default models
        with open(models_file, 'w') as f:
            json.dump(default_models, f, indent=2)
        
        return default_models
    else:
        raise FileNotFoundError("No models.json or models.json.defaults found")

# Get model names for dropdown
def get_model_names():
    models = load_models()
    return [model["custom_name"] for model in models]

# Process images function
def process_images(image_folder, model_name, caption_extension, caption_separator, 
                  threshold, character_threshold, recursive, debug):

    result = None
    report = None
    return result, report

# Main Gradio app
def create_app():
    models = load_models()
    
    with gr.Blocks(title="WD14/E621 Tagger") as demo:
        gr.Markdown("# WD14/E621 Tagger")
        
        with gr.Row():
            with gr.Column():
                image_folder = gr.Textbox(label="Image Folder", placeholder="Path to folder containing images")
                model_dropdown = gr.Dropdown(choices=get_model_names(), label="Model", value=get_model_names()[0])
                caption_extension = gr.Textbox(label="Caption File Extension", value=".txt")
                caption_separator = gr.Textbox(label="Caption Separator", value=", ")
                threshold = gr.Slider(0, 1, value=0.35, label="General Threshold")
                character_threshold = gr.Slider(0, 1, value=0.85, label="Character Threshold")
                recursive = gr.Checkbox(label="Recursive", value=False)
                debug = gr.Checkbox(label="Debug Mode", value=False)
                
                submit_btn = gr.Button("Process Images")
            
            with gr.Column():
                output_text = gr.Textbox(label="Output", interactive=False, lines=20)
                frequency_report = gr.Textbox(label="Tags Frequency Report", interactive=False, lines=15)
        
        # Event handling
        submit_btn.click(
            fn=process_images,
            inputs=[image_folder, model_dropdown, caption_extension, caption_separator, 
                   threshold, character_threshold, recursive, debug],
            outputs=[output_text, frequency_report]
        )
    
    return demo

# Run the app
if __name__ == "__main__":
    app = create_app()
    app.launch()