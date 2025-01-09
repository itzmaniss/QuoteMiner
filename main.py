import os
import json
from pathlib import Path

from config.settings import (
    INPUT_DIR, OUTPUT_DIR, TRANSCRIPTIONS_DIR,
    get_pytorch_settings
)
from src.transcription import transcribe
from src.quote_extraction import extract_quotes
from src.timestamp_utils import precompute_word_indices, find_best_fit_optimized
from src.video_processing import generate_clips

def main() -> None:
    """Main execution function."""
    # Create necessary directories
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPTIONS_DIR, exist_ok=True)

    # Get PyTorch settings
    device, compute_type, whisper_model_size = get_pytorch_settings()

    # Process each video
    for video_file in INPUT_DIR.iterdir():
        if video_file.suffix != ".mp4":
            print(f"Skipping non-MP4 file: {video_file.name}")
            continue

        print(f"Processing video: {video_file.name}")

        try:
            # Step 1: Transcription
            print("Starting transcription...")
            transcribe(video_file, device, compute_type, whisper_model_size, TRANSCRIPTIONS_DIR)
            print("Transcription completed.")

            # Load transcript for quote extraction
            transcript_path = TRANSCRIPTIONS_DIR / f"{video_file.stem}.json"
            with open(transcript_path, "r") as file:
                transcript_data = json.load(file)

            # Precompute word indices
            print("Precomputing word indices...")
            word_to_indices, transcript_words = precompute_word_indices(transcript_data["text"])

            # Step 2: Extract quotes
            print("Extracting quotes...")
            quotes = extract_quotes(TRANSCRIPTIONS_DIR)
            print(quotes)

            # Step 3: Find timestamps for quotes
            print("Finding timestamps for quotes...")
            timestamped_quotes = []
            for quote in quotes:
                try:
                    start, end = find_best_fit_optimized(
                        quote,
                        transcript_data["word_timestamps"],
                        transcript_words,
                        word_to_indices
                    )
                    timestamped_quotes.append({
                        "text": quote,
                        "start": start,
                        "end": end
                    })
                except ValueError as e:
                    print(f"Could not find timestamps for quote: {quote}. Error: {e}")

            # Step 4: Generate video clips
            print("Generating video clips...")
            generate_clips(video_file, timestamped_quotes, OUTPUT_DIR)
            print(f"Video clips generated for: {video_file.name}")

        except Exception as e:
            print(f"An error occurred while processing {video_file.name}: {e}")

    print("Processing completed for all videos.")

if __name__ == "__main__":
    main()
