import pathlib
import torch

# Directories
INPUT_DIR = pathlib.Path("../data/videos")
OUTPUT_DIR = pathlib.Path("../data/")
TRANSCRIPTIONS_DIR = pathlib.Path("../data/transcriptions")

# Maximum number of workers for parallel processing
MAX_WORKERS = 4  # Adjust based on your system's capabilities

# Model settings
MODEL_PATH = None  # set if you have local whisperx model
CONSTRAINT = False  # Set as true if you have lesser than 16gb of ram


def get_pytorch_settings():
    """Configure PyTorch and whisper_model settings based on hardware availability."""
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        whisper_model_size = "large-v2"
    else:
        device = "cpu"
        compute_type = "float32"
        whisper_model_size = "base.en"

    if CONSTRAINT:
        device = "cpu"
        compute_type = "float32"
        whisper_model_size = "small.en"

    return device, compute_type, whisper_model_size
