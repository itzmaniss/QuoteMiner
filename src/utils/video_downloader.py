from pathlib import Path
import subprocess
from utils import Logger

logger = Logger(name="VideoDownloader", filename="QuoteMiner.log")


def download(output_dir=Path("../data/video"), file=None, link=None):
    output_template = f"{output_dir}/%(title)s.%(ext)s"

    if file is None:
        logger.info(f"Downloading single video: {link}")
        result = subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bv*[ext=mp4]+ba",
                "--merge-output-format",
                "mp4",
                "-o",
                output_template,
                "--print",
                "after_move:filepath",
                link,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"yt-dlp failed: {result.stderr}")
            return []

        video_path = Path(result.stdout.strip())
        logger.info(f"Download completed: {video_path}")
        return [video_path]

    else:
        logger.info(f"Downloading videos from file: {file}")
        result = subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bv*[ext=mp4]+ba",
                "--merge-output-format",
                "mp4",
                "-o",
                output_template,
                "--print",
                "after_move:filepath",
                "-a",
                file,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"yt-dlp batch failed: {result.stderr}")
            return []

        # Each line in stdout is a downloaded file path
        video_paths = [
            Path(line.strip()) for line in result.stdout.strip().splitlines()
        ]
        logger.info(f"Batch download completed: {len(video_paths)} videos")
        return video_paths
