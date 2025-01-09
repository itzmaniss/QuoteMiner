import pathlib
import torch

# Directories
INPUT_DIR = pathlib.Path("./videos")
OUTPUT_DIR = pathlib.Path("./clips")
TRANSCRIPTIONS_DIR = pathlib.Path("./transcriptions")

# Model settings
MODEL_PATH = None  # set if you have local whisperx model
CONSTRAINT = False  # Set as true if you have lesser than 16gb of ram

def get_pytorch_settings():
    """Configure PyTorch and WhisperX settings based on hardware availability."""
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        whisper_model_size = "large-v2"
    else:
        device = "cpu"
        compute_type = "int8"
        whisper_model_size = "small"
    
    if CONSTRAINT:
        whisper_model_size = "tiny"

    return device, compute_type, whisper_model_size