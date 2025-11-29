import os
import json
import gradio as gr
from pathlib import Path
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
from PIL import Image

# Global variables
MODELS = None

# Load models from JSON files, create default if missing
def load_models():
    global MODELS
    if MODELS is not None:
        return MODELS
    
    models_file = "models.json"
    defaults_file = "models.json.defaults"
    
    # Try to load custom models file first
    if os.path.exists(models_file):
        with open(models_file, 'r') as f:
            MODELS = json.load(f)
    # Fallback to defaults and create the file
    elif os.path.exists(defaults_file):
        with open(defaults_file, 'r') as f:
            default_models = json.load(f)
        
        # Create models.json with default models
        with open(models_file, 'w') as f:
            json.dump(default_models, f, indent=2)
        
        MODELS = default_models
    else:
        raise FileNotFoundError("No models.json or models.json.defaults found")
    
    return MODELS

# Get model names for dropdown
def get_model_names():
    models = load_models()
    return [model["custom_name"] for model in models]

# Download model if it doesn't exist
def download_model(model_info):
    hf_repo = model_info["hf_repo"]
    model_file = model_info["model_file"]
    tags_file = model_info["tags_file"]
    local_folder = model_info.get("local_folder")
    
    # Determine where to download the model
    if local_folder:
        # Use specified local folder
        download_path = Path(local_folder)
        download_path.mkdir(parents=True, exist_ok=True)
    else:
        # Use Hugging Face cache
        download_path = Path.home() / ".cache" / "huggingface" / "hub"
    
    # Check if model files already exist
    model_path = download_path / model_file
    tags_path = download_path / tags_file
    
    if not model_path.exists() or not tags_path.exists():
        print(f"Downloading model {hf_repo} to {download_path}")
        # Download just the specific files we need
        hf_hub_download(
            repo_id=hf_repo,
            filename=model_file,
            local_dir=download_path,
        )
        hf_hub_download(
            repo_id=hf_repo,
            filename=tags_file,
            local_dir=download_path,
        )
    
    return str(download_path)

# Process images function
def process_images(image_folder, model_name, caption_extension, caption_separator, 
                  threshold, character_threshold, recursive, debug):
    
    # Find the selected model
    models = load_models()
    selected_model = None
    for model in models:
        if model["custom_name"] == model_name:
            selected_model = model
            break
    
    if not selected_model:
        return "Error: Model not found", "Error: Model not found"
    
    # Download model if needed
    try:
        model_path = download_model(selected_model)
    except Exception as e:
        return f"Error downloading model: {str(e)}", f"Error downloading model: {str(e)}"
    
    # Validate image folder
    if not os.path.exists(image_folder):
        return "Error: Image folder does not exist", "Error: Image folder does not exist"
    
    # Get all image files in the folder (and subfolders if recursive is enabled)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    image_files = []
    
    if recursive:
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(image_folder):
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(image_folder, file))
    
    if not image_files:
        return "Error: No images found in the folder", "Error: No images found in the folder"
    
    result = f"Processing {len(image_files)} images from: {image_folder}\n"
    result += f"Model: {model_name}\n"
    result += f"Extension: {caption_extension}\n"
    result += f"Separator: {caption_separator}\n"
    result += f"Threshold: {threshold}\n"
    result += f"Character Threshold: {character_threshold}\n"
    result += f"Recursive: {recursive}\n"
    result += f"Debug: {debug}\n\n"
    
    # Process each image
    processed_count = 0
    for image_path in image_files:
        try:
            # Load and preprocess image for ONNX model
            result += f"Processing: {os.path.basename(image_path)}\n"
            
            if debug:
                result += f"  Image path: {image_path}\n"
                
            processed_count += 1
            
        except Exception as e:
            result += f"Error processing {os.path.basename(image_path)}: {str(e)}\n"
    
    result += f"\nProcessed {processed_count} images successfully.\n"
    
    report = None
    
    return result, report

# Main Gradio app
def create_app():
    load_models()  # Load models once at startup
    
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