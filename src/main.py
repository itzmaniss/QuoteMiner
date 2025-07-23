import json
from pathlib import Path
from typing import Optional
from multiprocessing import cpu_count, Pool
import argparse

from config import INPUT_DIR, OUTPUT_DIR, TRANSCRIPTIONS_DIR, MAX_WORKERS
from utils import (
    Logger,
    QuoteExtractor,
    QuoteExtractorConfig,
    transcribeAudio as transcribe,
    download,
    VideoCropper,
)

logger = Logger(name="QuoteMiner", filename="QuoteMiner.log")


def ensure_directories_exist():
    """Ensure all required directories exist for the application to run properly."""
    logger.info("Ensuring required directories exist...")

    # Core directories from settings
    directories_to_create = [
        INPUT_DIR,  # ../data/videos
        OUTPUT_DIR,  # ../data/
        TRANSCRIPTIONS_DIR,  # ./transcriptions
        Path(OUTPUT_DIR) / "quotes",  # ../data/quotes
        Path(OUTPUT_DIR) / "tiktok_clips",  # ../data/tiktok_clips
    ]

    for directory in directories_to_create:
        directory = Path(directory)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"✅ Directory ensured: {directory}")
        except Exception as e:
            logger.error(f"❌ Failed to create directory {directory}: {e}")
            raise

    # Check for critical files
    critical_files = [
        "../.env",  # Environment variables
    ]

    for file_path in critical_files:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"⚠️  Critical file missing: {file_path}")
            logger.warning(
                "Please create .env file with ANTHROPIC_API_KEY and HUGGINGFACE_TOKEN"
            )
        else:
            logger.debug(f"✅ Critical file found: {file_path}")

    logger.info("Directory structure verification complete")


def process_video(
    video_path: Path, background_video_path: Optional[Path] = None
) -> None:
    """Process a single video file for transcription and quote extraction."""

    # transcribe the video
    logger.info(f"Transcribing {video_path.name}")
    transcript_path = TRANSCRIPTIONS_DIR / f"{video_path.stem}.json"
    transcription = transcribe(audio_path=video_path, save_path=transcript_path)
    if not transcription:
        logger.error(f"No transcription available for {video_path.name}")
        return
    logger.info(f"Transcription completed for {video_path.name}")

    # Extract quotes from the transcription
    logger.info(f"Extracting quotes from {video_path.name}")
    extractor_config = QuoteExtractorConfig()
    extractor = QuoteExtractor(config=extractor_config)
    quotes = extractor.extract_quotes(transcription)
    if not quotes:
        logger.error(f"No quotes extracted from {video_path.name}")
        return
    logger.info(f"Extracted {len(quotes)} quotes from {video_path.name}")
    with open(OUTPUT_DIR / f"quotes/{video_path.stem}_quotes.json", "w") as f:
        json.dump([quote.to_dict() for quote in quotes], f, indent=4)

    # Generate video clips for each quote
    video_cropper = VideoCropper()
    logger.info(f"Generating video clips for quotes in {video_path.name}")
    output_files = video_cropper.process_quotes_to_tiktok_videos(
        video_path=video_path,
        quotes=quotes,
        output_dir=OUTPUT_DIR / "tiktok_clips",
        background_video_path=background_video_path,
    )  # Optional background video path


def main() -> None:
    """Main execution function."""
    # Ensure all required directories exist before processing
    ensure_directories_exist()

    parser = argparse.ArgumentParser(description="Create TikTok clips from videos")
    parser.add_argument(
        "--video-link", type=str, help="YouTube video link to download and process"
    )
    parser.add_argument(
        "--input-dir", type=str, help="Directory containing input video files"
    )
    parser.add_argument(
        "--video-path", type=str, help="Path to the video file to process"
    )
    parser.add_argument(
        "--background-video", type=str, help="Path to background video (optional)"
    )

    args = parser.parse_args()
    print(args)

    if args.video_path:
        video_files = [
            Path(args.video_path),
        ]
        if not video_files:
            logger.error("No MP4 video files found in the input directory.")
            return

    elif args.video_link:
        video_files = download(link=args.video_link, output_dir=INPUT_DIR)
        if not video_files[0]:
            logger.error(f"Failed to download video from {args.video_link}")
            return

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error(f"Input directory {input_dir} does not exist.")
            return

        video_files = list(input_dir.glob("*.mp4"))
        if not video_files:
            logger.error(
                f"No MP4 video files found in the input directory {input_dir}."
            )
            return

    background_video_path = args.background_video
    logger.info(
        f"Processing {len(video_files)} video files with background video: {background_video_path}"
    )

    num_workers = min(MAX_WORKERS, cpu_count(), len(video_files))
    logger.info(f"Using {num_workers} workers for parallel processing")

    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            task_args = [(video, background_video_path) for video in video_files]
            pool.map(process_video, task_args)
    else:
        # Single-threaded fallback
        for video in video_files:
            process_video(video_path=video, background_video_path=None)


if __name__ == "__main__":
    main()
