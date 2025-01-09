import unicodedata
import string
from typing import Dict, List, Tuple
from collections import Counter
from difflib import SequenceMatcher

def normalize_word(word: str) -> str:
    """Normalize a word by removing excess punctuation while preserving contractions."""
    word = unicodedata.normalize("NFKD", word)
    return word.strip(string.punctuation).lower()

def precompute_word_indices(transcript: str) -> Tuple[Dict[str, List[int]], List[str]]:
    """Precompute a mapping of words to their positions in the transcript."""
    word_to_indices = {}
    transcript_words = [normalize_word(w) for w in transcript.split()]
    
    for idx, word in enumerate(transcript_words):
        if word not in word_to_indices:
            word_to_indices[word] = []
        word_to_indices[word].append(idx)
    
    return word_to_indices, transcript_words

def compute_similarity_score(quote_words: List[str], window_words: List[str]) -> float:
    """Compute a similarity score based on word overlap and sequence similarity."""
    quote_counter = Counter(quote_words)
    window_counter = Counter(window_words)
    overlap = sum((quote_counter & window_counter).values())
    union = sum((quote_counter | window_counter).values())
    word_overlap_score = overlap / union if union > 0 else 0

    sequence_similarity = SequenceMatcher(None, quote_words, window_words).ratio()
    return 0.6 * word_overlap_score + 0.4 * sequence_similarity

def find_best_fit_optimized(quote: str, word_timestamps: List[Dict], transcript_words: List[str], 
                          word_indices: Dict[str, List[int]]) -> Tuple[float, float]:
    """Find the best fit for a quote in the transcript using optimized search."""
    quote_words = [normalize_word(w) for w in quote.split()]

    # Find candidate windows
    candidate_windows = set()
    for word in quote_words:
        if word in word_indices:
            candidate_windows.update(word_indices[word])

    # Expand candidate windows
    windows = [(start, start + len(quote_words) - 1) 
              for start in sorted(candidate_windows)
              if start + len(quote_words) - 1 < len(transcript_words)]

    # Find best matching window
    best_score = 0
    best_window = (0, 0)
    for start, end in windows:
        window_words = transcript_words[start : end + 1]
        score = compute_similarity_score(quote_words, window_words)
        if score > best_score:
            best_score = score
            best_window = (start, end)

    # Get timestamps
    start_idx, end_idx = best_window
    start_time = word_timestamps[start_idx]["start"]
    end_time = word_timestamps[end_idx]["end"] + 2.0

    return start_time, end_time
