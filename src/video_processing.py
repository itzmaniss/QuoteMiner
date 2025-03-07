from pathlib import Path
from typing import List, Dict
import subprocess
from tqdm import tqdm

def generate_clips(video_path: Path, quotes: List[Dict], output_dir: Path, max_concurrent: int = 4) -> None:
    """Generate video clips concurrently using subprocess.Popen."""
    
    active_processes = []
    
    with tqdm(total=len(quotes)) as pbar:
        for idx, quote in enumerate(quotes):
            start = quote.get("start")
            end = quote.get("end")
            
            if start is None or end is None:
                print(f"Skipping quote {idx + 1} due to missing timestamps.")
                continue
                
            output_file = output_dir / f"{video_path.stem}_{idx + 1}.mp4"
            
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-ss", str(start),
                "-to", str(end),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-loglevel", "error",
                str(output_file)
            ]
            
            # Start new process
            process = subprocess.Popen(cmd)
            active_processes.append(process)
            
            # If we've hit max concurrent processes, wait for one to finish
            while len(active_processes) >= max_concurrent:
                for i, proc in enumerate(active_processes):
                    if proc.poll() is not None:  # Process finished
                        active_processes.pop(i)
                        pbar.update(1)
                        break
                        
        # Wait for remaining processes to finish
        for proc in active_processes:
            proc.wait()
            pbar.update(1)