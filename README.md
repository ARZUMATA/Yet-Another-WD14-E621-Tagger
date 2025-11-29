# WD14/E621 Tagger

Yes, yes.
Another tagger.
I'd like this to be a standalone tool with support for cascade models and, eventually, a tag sorting feature that is planned.

## Model Configuration

The application uses a `models.json` file to configure available models. Each model entry should include:
- `custom_name`: Display name in the dropdown
- `hf_repo`: Hugging Face repository ID
- `model_file`: ONNX model filename
- `tags_file`: Tags CSV filename
- `local_folder`: Optional local folder for model storage

## Inspired by

[Z3D-E621-Convnext-Tagger](https://huggingface.co/spaces/John6666/Z3D-E621-Convnext-Tagger)

[Kohya-SS sd-scripts](https://github.com/kohya-ss/sd-scripts)