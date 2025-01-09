import json
import whisperx
import re
from pathlib import Path
from typing import Dict, Any

def transcribe(video_path: Path, device: str, compute_type: str, whisper_model_size: str, transcriptions_dir: Path) -> None:
    """Transcribe video and save results with word-level timestamps."""
    model = whisperx.load_model(whisper_model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(video_path.absolute()))
    result = model.transcribe(audio, language="en", batch_size=16)
    
    complete_transcript = ""
    for segment in result["segments"]:
        complete_transcript += segment["text"]
    
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    refined_result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    
    word_timestamps = extract_word_timestamps(refined_result)
    save_transcript(video_path, complete_transcript, word_timestamps, transcriptions_dir)

def extract_word_timestamps(refined_result: Dict[str, Any]) -> list:
    """Extract word-level timestamps from refined transcription result."""
    word_timestamps = []
    for segment in refined_result["segments"]:
        for word_info in segment.get("words", []):
            if "start" in word_info and "end" in word_info:
                word_timestamps.append({
                    "word": word_info["word"],
                    "start": word_info["start"],
                    "end": word_info["end"],
                    "score": word_info.get("score", 0)
                })
    return word_timestamps

def save_transcript(video_path: Path, transcript: str, word_timestamps: list, transcriptions_dir: Path) -> None:
    """Save transcript and word timestamps to JSON file."""
    transcript_path = transcriptions_dir / f"{video_path.stem}.json"
    with open(transcript_path, "w") as file:
        json.dump({
            "text": transcript,
            "word_timestamps": word_timestamps
        }, file)


def remove_trailing_commas(json_string):
    """
    Remove trailing commas from a JSON string.

    Args:
        json_string (str): The JSON string to clean.

    Returns:
        str: The JSON string without trailing commas.
    """
    # Regex to detect trailing commas before a closing brace or bracket
    trailing_comma_pattern = re.compile(r",\s*([}\]])")

    # Replace all trailing commas with the closing brace/bracket
    cleaned_json = trailing_comma_pattern.sub(r"\1", json_string)

    return cleaned_json
