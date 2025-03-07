from pathlib import Path
import subprocess



def download(output_folder = Path("./videos"), file = None, link = None):
    if file == None:
        subprocess.run([
            "yt-dlp",
            "-f",
            "bv*[ext=mp4]+ba",
            "--merge-output-format",
            "mp4",
            "-o",
            f"{output_folder}/%(title)s.%(ext)s",
            link,
            "--progress"
        ])

    else:
        subprocess.run([
            "yt-dlp",
            "-f",
            "bv*[ext=mp4]+ba",
            "--merge-output-format",
            "mp4",
            "-o",
            f"{output_folder}/%(title)s.%(ext)s",
            "-a",
            file,
            "--progress"
        ])