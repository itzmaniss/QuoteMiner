from faster_whisper import WhisperModel
from config import get_pytorch_settings
from utils import Logger
from pathlib import Path
from typing import Optional
import json


logger = Logger(name="TranscriptionLogger", filename="QuoteMiner.log")


def transcribeAudio(audio_path: Path, save_path: Optional[Path] = None) -> list:
    device, compute_type, whisper_model_size = get_pytorch_settings()
    logger.info(
        f"Transcribing audio: {audio_path} using model: {whisper_model_size} on {device}"
    )

    logger.debug(f"Loading Whisper model: {whisper_model_size}")
    model = WhisperModel(whisper_model_size, device=device)
    logger.debug("Whisper model loaded successfully")

    try:
        logger.info("Starting transcription process")
        segments, info = model.transcribe(
            audio=audio_path,
            beam_size=5,
            language="en",
            max_new_tokens=128,
            condition_on_previous_text=False,
        )

        logger.info(
            f"Transcription completed for {audio_path}. Duration: {info.duration:.2f} seconds"
        )
        logger.debug(
            f"Transcription info - Language: {info.language}, Language probability: {info.language_probability:.2f}"
        )

    except FileNotFoundError as e:
        logger.error(f"Audio file not found: {audio_path} - {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Transcription error for {audio_path}: {str(e)}")
        return []

    segments = list(segments)
    logger.info(f"Processing {len(segments)} transcript segments")

    extracted_texts = []
    for i, segment in enumerate(segments):
        extracted_texts.append([segment.text, segment.start, segment.end])
        logger.debug(
            f"Segment {i + 1}: {segment.start:.2f}-{segment.end:.2f}s: {segment.text[:50]}..."
        )

    logger.info(f"Successfully extracted {len(extracted_texts)} text segments")

    if save_path:
        logger.info(f"Saving transcription to {save_path}")
        with open(save_path, "w") as f:
            json.dump({"transcription": extracted_texts}, f, indent=4)
        logger.info("Transcription saved successfully")

    return extracted_texts
