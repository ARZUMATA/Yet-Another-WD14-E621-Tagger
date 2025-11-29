import os
import json
import gradio as gr
from pathlib import Path
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
from PIL import Image
import csv
import onnxruntime as onnxrt

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

# Preprocess image for ONNX model
def preprocess_image(image_path):
    """
    Preprocess image for WD14:
    - Load as RGB
    - Pad original image to square (aspect preserved)
    - Resize square to target_size
    - Convert RGB -> BGR
    - Return BGR numpy array
    """

    # Load as RGB using PIL
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    h, w = image_rgb.shape[:2]

    # Pad to square (using max dimension)
    max_dim = max(h, w)
    
    # Calculate padding (centered)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2

    # Pad to square: (top, bottom), (left, right), (no padding on channels)
    padded_rgb = np.pad(
        image_rgb,
        ((pad_h, pad_h + (max_dim - h - pad_h)), 
         (pad_w, pad_w + (max_dim - w - pad_w)), 
         (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Resize to 448x448 with dynamic interpolation selection
    original_size = padded_rgb.shape[0]  # Since image is now square
    target_size = 448
    
    if original_size >= target_size:
        # Downscaling - use INTER_AREA
        interpolation = cv2.INTER_AREA
    else:
        # Upscaling - use INTER_LANCZOS4
        interpolation = cv2.INTER_LANCZOS4
    
    # Now resize square padded image to target size
    resized_rgb = cv2.resize(padded_rgb, (target_size, target_size), interpolation=interpolation)

    # Convert RGB to BGR
    image_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)

    image = image_bgr.astype(np.float32)
    return image

 # Load tags from CSV
def load_tags_from_csv(tags_file_path):
    """Load tag names from CSV file."""
    tags = []

    with open(tags_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)[1:]  # Skip header
        for row in rows:
            if len(row) >= 4:  # Ensure row has enough columns
                tag_data = {
                    'name': row[1],      # tag name
                    'category': row[2]   # tag category
                }
                tags.append(tag_data)
    return tags

# Process images function
def process_images(image_folder, model_name, caption_extension, caption_separator, 
                   general_threshold, character_threshold, recursive, debug):
    
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
    result += f"General Threshold: {general_threshold}\n"
    result += f"Character Threshold: {character_threshold}\n"
    result += f"Recursive: {recursive}\n"
    result += f"Debug: {debug}\n\n"
    
    # Process each image
    processed_count = 0
    
    try:
        # Get model file path
        model_file_path = os.path.join(model_path, selected_model["model_file"])
        tags_file_path = os.path.join(model_path, selected_model["tags_file"])
        
        # Select appropriate execution provider for ONNX Runtime
        available_providers = onnxrt.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            provider = ["CUDAExecutionProvider"]
        elif "ROCMExecutionProvider" in available_providers:
            provider = ["ROCMExecutionProvider"]
        else:
            provider = ["CPUExecutionProvider"]

        # Create ONNX session
        onnxrt_session = onnxrt.InferenceSession(model_file_path, providers=provider)
        
        # Prepare for inference
        input_name = onnxrt_session.get_inputs()[0].name

        # Load tags from CSV
        tags = load_tags_from_csv(tags_file_path)
        tag_frequency = {}

        # For each image, process it
        for image_path in image_files:
            try:
                # Load and preprocess image
                processed_image = preprocess_image(image_path)
                
                # Add batch dimension
                input_tensor = np.expand_dims(processed_image, axis=0)
                
                # Run inference
                outputs = onnxrt_session.run(None, {input_name: input_tensor})
                probs = outputs[0][0]  # Get the first (and only) output
                
                # Assign probabilities to tags
                for i, tag in enumerate(tags):
                    if i < len(probs):
                        tag['prob'] = probs[i]

                # Select tags based on categories and thresholds
                rating_tags = [tag for tag in tags if tag['category'] == '9']
                character_tags = [tag for tag in tags if tag['category'] == '4' and tag['prob'] >= character_threshold]
                general_tags = [tag for tag in tags if tag['category'] != '9' and tag['category'] != '4' and tag['prob'] >= general_threshold]

                # Select only one rating tag with max probability
                selected_rating_tag = None
                if rating_tags:
                    selected_rating_tag = max(rating_tags, key=lambda x: x['prob'])

                # Update tag frequencies
                for tag in character_tags:
                    tag_frequency[tag['name']] = tag_frequency.get(tag['name'], 0) + 1
                for tag in general_tags:
                    tag_frequency[tag['name']] = tag_frequency.get(tag['name'], 0) + 1

                # Combine selected tags
                combined_tags = []
                # if selected_rating_tag:
                #     combined_tags.append(selected_rating_tag['name'])
                combined_tags.extend([tag['name'] for tag in general_tags])
                combined_tags.extend([tag['name'] for tag in character_tags])

                # Save caption file
                caption_file = os.path.splitext(image_path)[0] + caption_extension
                tag_text = caption_separator.join(combined_tags)
                
                with open(caption_file, "wt", encoding="utf-8") as f:
                    f.write(tag_text + "\n")
                
                if debug:
                    result += f"  Tags: {tag_text}\n"
                
                processed_count += 1
                    
            except Exception as e:
                result += f"Error processing {os.path.basename(image_path)}: {str(e)}\n"
        
            result += f"\nProcessed {processed_count} images successfully.\n"
        
        # Generate frequency report
        sorted_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)
        report = "Tag frequencies:\n"
        for tag, freq in sorted_tags[:20]:  # Show top 20 tags
            report += f"{tag}: {freq}\n"
        
        if len(sorted_tags) > 20:
            report += f"... and {len(sorted_tags) - 20} more tags\n"
        
    except Exception as e:
        return f"Error during processing: {str(e)}", f"Error during processing: {str(e)}"
    
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
                general_threshold = gr.Slider(0, 1, value=0.35, step=0.05, label="General Threshold")
                character_threshold = gr.Slider(0, 1, value=0.35, step=0.05, label="Character Threshold")
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
                    general_threshold, character_threshold, recursive, debug],
            outputs=[output_text, frequency_report]
        )
    
    return demo

# Run the app
if __name__ == "__main__":
    app = create_app()
    app.launch()