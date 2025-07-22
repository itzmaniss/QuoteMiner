import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count
import cv2
from moviepy import VideoFileClip, CompositeAudioClip
from pyannote.audio import Pipeline
import torch
from dotenv import load_dotenv

from utils import Logger
from models import Quote, SpeakerSegment, CropRegion


def _process_quote_worker(
    args: Tuple[str, Quote, int, str, CropRegion, Optional[str], Optional[str]],
) -> Optional[str]:
    """Worker function for multiprocessing - must be at module level."""
    (
        video_path,
        quote,
        quote_index,
        output_dir,
        crop_region,
        hf_token,
        background_video_path,
    ) = args

    # Create logger for this worker
    logger = Logger(name=f"Worker-{quote_index}", filename="QuoteMiner.log")

    try:
        # Create a temporary audio file for this specific quote
        temp_audio_path = os.path.join(output_dir, f"temp_audio_{quote_index}.wav")

        # Extract just the audio segment for this quote
        with VideoFileClip(video_path) as video:
            audio_clip = video.subclipped(quote.start, quote.end).audio
            if audio_clip is not None:
                audio_clip.write_audiofile(temp_audio_path)
                audio_clip.close()

        # Perform speaker diarization on just this small audio segment
        active_speaker = "unknown"
        if os.path.exists(temp_audio_path) and hf_token:
            try:
                # Initialize pipeline for this worker
                load_dotenv()
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
                )

                if not torch.cuda.is_available():
                    pipeline.to(torch.device("cpu"))

                # Run diarization
                diarization = pipeline(temp_audio_path)
                speaker_time = {}

                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    duration = turn.end - turn.start
                    speaker_time[speaker] = speaker_time.get(speaker, 0) + duration

                active_speaker = (
                    max(speaker_time, key=speaker_time.get)
                    if speaker_time
                    else "unknown"
                )

            except Exception as e:
                logger.warning(
                    f"Speaker diarization failed for quote {quote_index}: {e}"
                )

        # Generate output filename
        safe_content = "".join(
            c for c in quote.content if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_content = safe_content[:50]
        output_filename = f"quote_{quote_index + 1}_{active_speaker}_{safe_content}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # Create TikTok clip
        with VideoFileClip(video_path) as video:
            # Extract the clip for the quote duration
            clip = video.subclipped(quote.start, quote.end)

            # Crop the video to focus on the speaker
            cropped = clip.cropped(
                x1=crop_region.x,
                y1=crop_region.y,
                x2=crop_region.x + crop_region.width,
                y2=crop_region.y + crop_region.height,
            )

            # Resize to TikTok dimensions (1080x1920)
            resized = cropped.resized((1080, 1920))

            # Add background video if provided
            if background_video_path and os.path.exists(background_video_path):
                with VideoFileClip(background_video_path) as bg_video:
                    # Get background clip duration matching quote duration
                    quote_duration = quote.end - quote.start

                    # Loop background video if it's shorter than quote duration
                    if bg_video.duration < quote_duration:
                        # Calculate how many loops needed
                        loops_needed = int(quote_duration / bg_video.duration) + 1
                        bg_clip = bg_video.concatenate_videoclips(
                            [bg_video] * loops_needed
                        )
                    else:
                        bg_clip = bg_video

                    # Crop background to quote duration
                    bg_clip = bg_clip.subclipped(0, quote_duration)

                    # Resize background to TikTok dimensions
                    bg_resized = bg_clip.resized((1080, 1920))

                    # Reduce background audio to 15% volume
                    if bg_resized.audio:
                        bg_resized = bg_resized.with_audio(
                            bg_resized.audio.with_volume_scaled(0.15)
                        )

                    # Composite: background video behind main video
                    # Main video takes priority, background fills the space
                    final_clip = bg_resized.with_mask().composite([resized])

                    # Ensure main audio is preserved at full volume
                    if resized.audio and bg_resized.audio:
                        # Mix main audio (full volume) with background audio (15% volume)
                        # Use CompositeAudioClip for proper audio mixing
                        mixed_audio = CompositeAudioClip(
                            [
                                resized.audio.with_volume_scaled(1.0),
                                bg_resized.audio.with_volume_scaled(0.15),
                            ]
                        )
                        final_clip = final_clip.with_audio(mixed_audio)
                    elif resized.audio:
                        # Only main audio
                        final_clip = final_clip.with_audio(resized.audio)

                    # Write the output
                    final_clip.write_videofile(
                        output_path, codec="libx264", audio_codec="aac"
                    )
            else:
                # No background video, use original workflow
                resized.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Clean up temporary audio
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        logger.info(f"Created TikTok clip: {output_filename}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to process quote {quote_index + 1}: {e}")
        # Clean up on error
        temp_audio_path = os.path.join(output_dir, f"temp_audio_{quote_index}.wav")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return None


class VideoCropper:
    def __init__(self, hf_token: Optional[str] = None, log_level=None):
        self.logger = Logger(name="VideoCropper", filename="QuoteMiner.log")
        # Load environment variables
        load_dotenv()

        # Use provided token or load from environment
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            self.logger.warning(
                "No HuggingFace token provided. Speaker diarization may fail."
            )

        self.pipeline = None
        self.tiktok_aspect_ratio = 9 / 16  # 9:16 for TikTok
        self.target_height = 1920
        self.target_width = 1080

    def _initialize_diarization_pipeline(self):
        """Initialize the speaker diarization pipeline."""
        if self.pipeline is None:
            try:
                self.logger.info("Initializing speaker diarization pipeline...")
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", use_auth_token=self.hf_token
                )

                # Use CPU if CUDA is not available to save memory
                if not torch.cuda.is_available():
                    self.logger.info("CUDA not available, using CPU for diarization")
                    self.pipeline.to(torch.device("cpu"))
                else:
                    self.logger.info("Using GPU for diarization")

            except Exception as e:
                self.logger.error(f"Failed to initialize diarization pipeline: {e}")
                raise

    def perform_speaker_diarization(self, audio_path: str) -> List[SpeakerSegment]:
        """Perform speaker diarization on audio file."""
        self._initialize_diarization_pipeline()

        try:
            self.logger.info(f"Performing speaker diarization on: {audio_path}")
            diarization = self.pipeline(audio_path)

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    SpeakerSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker=speaker,
                        confidence=1.0,  # pyannote doesn't provide confidence scores directly
                    )
                )

            self.logger.info(f"Found {len(segments)} speaker segments")
            return segments

        except Exception as e:
            self.logger.error(f"Speaker diarization failed: {e}")
            raise

    def _find_active_speaker(
        self, quote: Quote, speaker_segments: List[SpeakerSegment]
    ) -> Optional[str]:
        """Find the most active speaker during a quote timespan."""
        quote_start, quote_end = quote.start, quote.end
        speaker_overlap = {}

        for segment in speaker_segments:
            # Calculate overlap with quote timespan
            overlap_start = max(quote_start, segment.start)
            overlap_end = min(quote_end, segment.end)

            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                if segment.speaker not in speaker_overlap:
                    speaker_overlap[segment.speaker] = 0
                speaker_overlap[segment.speaker] += overlap_duration

        if not speaker_overlap:
            return None

        # Return speaker with most overlap
        active_speaker = max(speaker_overlap, key=speaker_overlap.get)
        self.logger.debug(
            f"Active speaker for quote '{quote.content[:50]}...': {active_speaker}"
        )
        return active_speaker

    def _detect_face_regions(
        self, video_path: str, timestamps: List[float]
    ) -> Dict[float, CropRegion]:
        """Detect face regions at specific timestamps using OpenCV."""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        face_regions = {}

        try:
            for timestamp in timestamps:
                frame_number = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) > 0:
                    # Use the largest face
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = largest_face

                    # Expand the crop region around the face
                    frame_height, frame_width = frame.shape[:2]

                    # Calculate expanded region for TikTok format
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Ensure we maintain 9:16 aspect ratio
                    crop_width = min(frame_width, int(frame_height * (9 / 16)))
                    crop_height = min(frame_height, int(crop_width * (16 / 9)))

                    # Center the crop on the face
                    crop_x = max(0, center_x - crop_width // 2)
                    crop_y = max(0, center_y - crop_height // 2)

                    # Adjust if crop goes beyond frame boundaries
                    if crop_x + crop_width > frame_width:
                        crop_x = frame_width - crop_width
                    if crop_y + crop_height > frame_height:
                        crop_y = frame_height - crop_height

                    face_regions[timestamp] = CropRegion(
                        x=crop_x, y=crop_y, width=crop_width, height=crop_height
                    )

        finally:
            cap.release()

        return face_regions

    def _create_tiktok_clip(
        self, video_path: str, quote: Quote, crop_region: CropRegion, output_path: str
    ):
        """Create a TikTok-formatted clip from the video."""
        try:
            with VideoFileClip(video_path) as video:
                # Extract the clip for the quote duration
                clip = video.subclipped(quote.start, quote.end)

                # Crop the video to focus on the speaker
                cropped = clip.cropped(
                    x1=crop_region.x,
                    y1=crop_region.y,
                    x2=crop_region.x + crop_region.width,
                    y2=crop_region.y + crop_region.height,
                )

                # Resize to TikTok dimensions
                resized = cropped.resized((self.target_width, self.target_height))

                # Write the output
                resized.write_videofile(output_path, codec="libx264", audio_codec="aac")

        except Exception as e:
            self.logger.error(f"Failed to create TikTok clip: {e}")
            raise

    def process_quotes_to_tiktok_videos(
        self,
        video_path: str,
        quotes: List[Quote],
        output_dir: str,
        max_workers: int = 4,
        background_video_path: Optional[str] = None,
    ) -> List[str]:
        """Process quotes and create TikTok-formatted videos with multiprocessing."""
        self.logger.info(f"Processing {len(quotes)} quotes from video: {video_path}")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Get face detection for all quotes upfront (this is fast)
        timestamps = [quote.start + quote.duration / 2 for quote in quotes]
        face_regions = self._detect_face_regions(video_path, timestamps)

        # Prepare crop regions for all quotes
        quote_data = []
        for i, quote in enumerate(quotes):
            mid_timestamp = quote.start + quote.duration / 2
            crop_region = face_regions.get(mid_timestamp)

            if not crop_region:
                # Default crop region if no face detected
                with VideoFileClip(video_path) as temp_video:
                    width, height = temp_video.w, temp_video.h
                    crop_width = min(width, int(height * (9 / 16)))
                    crop_height = min(height, int(crop_width * (16 / 9)))
                    crop_region = CropRegion(
                        x=(width - crop_width) // 2,
                        y=(height - crop_height) // 2,
                        width=crop_width,
                        height=crop_height,
                    )

            quote_data.append(
                (
                    video_path,
                    quote,
                    i,
                    output_dir,
                    crop_region,
                    self.hf_token,
                    background_video_path,
                )
            )

        # Process quotes with multiprocessing (max 4 workers)
        num_workers = min(max_workers, cpu_count(), len(quotes))
        self.logger.info(f"Using {num_workers} workers for parallel processing")

        output_files = []
        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                results = pool.map(_process_quote_worker, quote_data)
                output_files = [f for f in results if f is not None]
        else:
            # Single-threaded fallback
            for data in quote_data:
                result = _process_quote_worker(data)
                if result:
                    output_files.append(result)

        self.logger.info(f"Successfully created {len(output_files)} TikTok clips")
        return output_files

    def process_from_json(
        self,
        video_path: str,
        quotes_json_path: str,
        output_dir: str,
        background_video_path: Optional[str] = None,
    ) -> List[str]:
        """Process quotes from JSON file and create TikTok videos."""
        try:
            with open(quotes_json_path, "r") as f:
                quotes_data = json.load(f)

            quotes = []
            for quote_data in quotes_data:
                if isinstance(quote_data, dict):
                    quote = Quote(
                        start=float(quote_data["start"]),
                        content=quote_data["content"],
                        end=float(quote_data["end"]),
                    )
                    quotes.append(quote)

            return self.process_quotes_to_tiktok_videos(
                video_path,
                quotes,
                output_dir,
                background_video_path=background_video_path,
            )

        except Exception as e:
            self.logger.error(f"Failed to process quotes from JSON: {e}")
            raise
