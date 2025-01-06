import os
import whisperx
import subprocess
import pathlib
import json
import dotenv
import unicodedata
import string
import torch
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Tuple
from collections import Counter
from difflib import SequenceMatcher

#for future developments
# dotenv.load_dotenv()
# hf_token = os.getenv("hugging_face_token")

# Defining constants
input_dir = pathlib.Path("./videos")
output_dir = pathlib.Path("./clips")
transcriptions_dir = pathlib.Path("./transcriptions")

# set if you have local whisperx model
model_path = None

#Set as true if you have lesser than 16gb of ram
constraint = False

# Function to set PyTorch variables and WhisperX model size
def set_pytorch_vars():
    global device, compute_type, whisper_model_size

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = "cuda"  # Use CUDA for GPU acceleration
        compute_type = "float16"  # Mixed precision for better performance
        whisper_model_size = "large-v2"  # High-accuracy model for GPUs
        print("CUDA available: Using GPU with large-v2 model and float16 precision.")
    else:
        device = "cpu"  # Fallback to CPU
        compute_type = "int8"  # Lower precision for faster CPU inference
        whisper_model_size = "small"  # Lightweight model for CPUs
        print("CUDA not available: Using CPU with tiny model and int8 precision.")
    
    if constraint:
        whisper_model_size = "tiny"

    return device, compute_type, whisper_model_size



def main():
    # Ensure necessary directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(transcriptions_dir, exist_ok=True)

    device, compute_type, whisper_model_size = set_pytorch_vars()

    # Process each video in the input directory
    for video_file in input_dir.iterdir():
        if video_file.suffix != ".mp4":
            print(f"Skipping non-MP4 file: {video_file.name}")
            continue

        print(f"Processing video: {video_file.name}")

        try:
            # Step 1: Transcription
            print("Starting transcription...")
            transcribe(video_file)
            print("Transcription completed.")

            # Load the transcript for quote extraction
            transcript_path = transcriptions_dir / f"{video_file.stem}.json"
            with open(transcript_path, "r") as file:
                transcript_data = json.load(file)

            transcript_text = transcript_data["text"]
            word_timestamps = transcript_data["word_timestamps"]

            # Precompute word indices for efficient matching
            print("Precomputing word indices...")
            word_to_indices, transcript_words = precompute_word_indices(transcript_text)

            # Step 2: Quote Extraction
            print("Extracting quotes...")
            quotes = extract_quotes(transcriptions_dir)
            print(quotes)

            # Step 3: Assign timestamps to quotes
            print("Finding timestamps for quotes...")
            timestamped_quotes = []
            for quote in quotes:
                try:
                    start, end = find_best_fit_optimized(
                        quote, word_timestamps, transcript_words, word_to_indices
                    )
                    timestamped_quotes.append(
                        {"text": quote, "start": start, "end": end}
                    )
                except ValueError as e:
                    print(f"Could not find timestamps for quote: {quote}. Error: {e}")

            # Step 4: Generate video clips
            print("Generating video clips...")
            generate_clips(video_file, timestamped_quotes)
            print(f"Video clips generated for: {video_file.name}")

        except Exception as e:
            print(f"An error occurred while processing {video_file.name}: {e}")

    print("Processing completed for all videos.")


# Step 1: Transcription
def transcribe(video_path):
    model = whisperx.load_model(whisper_model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(str(video_path.absolute()))
    result = model.transcribe(audio, language="en", batch_size=16)
    complete_trancsript = ""
    for segment in result["segments"]:
        complete_trancsript += segment["text"]
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
    word_timestamps = []
    # Iterate through each segment to extract word timestamps and confidence scores
    for segment in refined_result["segments"]:
        for word_info in segment.get("words", []):
            if "start" in word_info and "end" in word_info:
                word_timestamps.append(
                    {
                        "word": word_info["word"],
                        "start": word_info[
                            "start"
                        ],  # Assuming start and end are present
                        "end": word_info["end"],  # Assuming start and end are present
                        "score": word_info.get(
                            "score", 0
                        ),  # Default score if not provided
                    }
                )
    transcript_path = transcriptions_dir / (video_path.stem + ".json")
    with open(transcript_path, "w") as file:
        json.dump(
            {"text": complete_trancsript, "word_timestamps": word_timestamps}, file
        )


def extract_quotes(transcriptions_dir) -> list:
    Settings.llm = Ollama(model="gemma2")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    documents = SimpleDirectoryReader(input_dir=transcriptions_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)

    detailed_prompt = """
    Your task is to identify motivational or inspirational excerpts from the provided transcript. These excerpts must:

    1. Be **at least 2-5 sentences long** to provide enough context for a meaningful video clip.
    2. **Preserve the exact wording** from the transcript without any modifications or omissions.
    3. Focus on **overcoming challenges, achieving goals, or self-improvement**.
    4. Evoke **positive emotions** such as hope, determination, or resilience.
    5. Provide **unique insights or actionable advice** that can benefit the audience.

    When analyzing the text:
    - Avoid generic statements, conversational filler, or overly brief responses.
    - Select excerpts that stand out as powerful and memorable.

    Output the excerpts in **verbatim** as a list of strings, with no additional commentary or explanation. Each string must correspond exactly to a section of the transcript.

    """

    query_engine = index.as_query_engine()
    response = query_engine.query(detailed_prompt)

    quotes = json.loads(response.response.strip().replace("```", "").strip())

    with open("motivational_lines.json", "w") as output_file:
        json.dump(quotes, output_file)

    return quotes


def normalize_word(word):
    """Normalize a word by removing excess punctuation while preserving contractions."""
    word = unicodedata.normalize("NFKD", word)  # Normalize unicode characters
    return word.strip(string.punctuation).lower()


def precompute_word_indices(transcript: str) -> dict:
    """Precompute a mapping of words to their positions in the transcript."""
    word_to_indices = {}
    transcript_words = [normalize_word(w) for w in transcript.split()]
    for idx, word in enumerate(transcript_words):
        if word not in word_to_indices:
            word_to_indices[word] = []
        word_to_indices[word].append(idx)
    return word_to_indices, transcript_words


def compute_similarity_score(quote_words, window_words):
    """Compute a similarity score based on word overlap and sequence similarity."""
    # Word overlap (Jaccard index)
    quote_counter = Counter(quote_words)
    window_counter = Counter(window_words)
    overlap = sum((quote_counter & window_counter).values())
    union = sum((quote_counter | window_counter).values())
    word_overlap_score = overlap / union if union > 0 else 0

    # Sequence similarity
    sequence_similarity = SequenceMatcher(None, quote_words, window_words).ratio()

    # Weighted score
    return 0.6 * word_overlap_score + 0.4 * sequence_similarity


def find_best_fit_optimized(
    quote: str, word_timestamps: list, transcript_words, word_indices
) -> tuple:
    """Find the best fit for a quote in the transcript using optimized search."""
    # Normalize and tokenize
    quote_words = [normalize_word(w) for w in quote.split()]

    # Find candidate windows
    candidate_windows = set()
    for word in quote_words:
        if word in word_indices:
            candidate_windows.update(word_indices[word])

    # Expand candidate windows to full quote length
    candidate_windows = sorted(candidate_windows)
    windows = [
        (start, start + len(quote_words) - 1)
        for start in candidate_windows
        if start + len(quote_words) - 1 < len(transcript_words)
    ]

    # Score each window
    best_score = 0
    best_window = (0, 0)
    for start, end in windows:
        window_words = transcript_words[start : end + 1]
        score = compute_similarity_score(quote_words, window_words)
        if score > best_score:
            best_score = score
            best_window = (start, end)

    # Retrieve timestamps
    start_idx, end_idx = best_window
    start_time = word_timestamps[start_idx]["start"]
    end_time = word_timestamps[end_idx]["end"] + 2.0

    return start_time, end_time


def generate_clips(video_path, quotes: List[dict]) -> None:
    for idx, quote in enumerate(quotes):
        start = quote["start"]
        end = quote["end"]

        if start is None or end is None:
            print(f"Skipping quote {idx + 1} due to missing timestamps.")
            continue

        output_file = output_dir / f"{video_path.stem}_{idx + 1}.mp4"
        cmd = [
            "ffmpeg", "-i", str(video_path), "-ss", str(start), "-to", str(end),
            "-c:v", "libx264", "-c:a", "aac", str(output_file)
        ]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
