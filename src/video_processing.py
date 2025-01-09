import subprocess
from pathlib import Path
from typing import List, Dict

def generate_clips(video_path: Path, quotes: List[Dict], output_dir: Path) -> None:
    """Generate video clips for each quote using ffmpeg."""
    for idx, quote in enumerate(quotes):
        start = quote.get("start")
        end = quote.get("end")

        if start is None or end is None:
            print(f"Skipping quote {idx + 1} due to missing timestamps.")
            continue

        output_file = output_dir / f"{video_path.stem}_{idx + 1}.mp4"
        cmd = [
            "ffmpeg", "-i", str(video_path), 
            "-ss", str(start), 
            "-to", str(end),
            "-c:v", "libx264", 
            "-c:a", "aac", 
            str(output_file)
        ]
        subprocess.run(cmd)
