"""
DTW Text Matching for Hebrew Transcription Alignment

Aligns segments from a pre-fix transcription file to regions in an LLM-corrected file,
tracking position and detecting potential hallucinations.

Supports two modes:
- Global DTW: Best when both files represent the same content (default)
- Subsequence DTW: Best when searching for segments in a longer, potentially unrelated reference
"""

from dataclasses import dataclass
import re
import sys
import argparse
import logging
from datetime import datetime
import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Fix Windows console encoding for Hebrew
sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
def setup_logging(log_file: str = None):
    """Setup logging to both console and file."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Create logger
    logger = logging.getLogger('dtw_match')
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # File handler
    if log_file is None:
        log_file = "dtw.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    logger.info(f"Logging to: {log_file}")
    return logger


@dataclass
class Segment:
    id: int
    timestamp: str
    text: str
    words: list[str]


@dataclass
class AlignmentResult:
    segment_id: int
    original_text: str
    matched_text: str
    start_pos: int
    end_pos: int
    match_score: float
    insertions: list[str]  # Words in corrected not in original (potential hallucinations)
    deletions: list[str]   # Words in original not in corrected (missing)


def tokenize(text: str) -> list[str]:
    """Tokenize Hebrew text into words."""
    text = re.sub(r'[^\w\s\u0590-\u05FF]', ' ', text)
    return [w for w in text.split() if w.strip()]


def parse_prefix_file(path: str) -> list[Segment]:
    """Parse the pre-fix file into segments."""
    segments = []
    pattern = r'\[(\d+)\]\s*([\d:.]+ - [\d:.]+):\s*(.+)'

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(pattern, line)
            if match:
                seg_id = int(match.group(1))
                timestamp = match.group(2)
                text = match.group(3)
                words = tokenize(text)
                segments.append(Segment(
                    id=seg_id,
                    timestamp=timestamp,
                    text=text,
                    words=words
                ))
    return segments


def parse_corrected_file(path: str) -> list[str]:
    """Parse the corrected file into a word list."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.replace('\n', ' ')
    return tokenize(text)


def parse_prefix_file_as_words(path: str) -> list[str]:
    """Parse the pre-fix file and extract only the text words (no segment IDs or timestamps)."""
    words = []
    pattern = r'\[(\d+)\]\s*([\d:.]+ - [\d:.]+):\s*(.+)'

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(pattern, line)
            if match:
                text = match.group(3)
                words.extend(tokenize(text))
    return words


def parse_file_as_chunks(path: str, chunk_size: int = 10) -> list[Segment]:
    """Parse a file into fixed-size word chunks as segments."""
    words = parse_corrected_file(path)
    segments = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        segments.append(Segment(
            id=i // chunk_size,
            timestamp=f"words {i}-{i + len(chunk_words) - 1}",
            text=' '.join(chunk_words),
            words=chunk_words
        ))
    return segments


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev_row[j + 1] + 1
            dels = curr_row[j] + 1
            subs = prev_row[j] + (c1 != c2)
            curr_row.append(min(ins, dels, subs))
        prev_row = curr_row
    return prev_row[-1]


def word_distance(w1: str, w2: str) -> float:
    """Normalized edit distance between words. 0 = identical, 1 = completely different."""
    if w1 == w2:
        return 0.0
    edit_dist = levenshtein_distance(w1, w2)
    max_len = max(len(w1), len(w2))
    return edit_dist / max_len if max_len > 0 else 0.0


# =============================================================================
# GLOBAL DTW ALIGNMENT
# =============================================================================

def global_dtw_alignment(all_prefix_words: list[str], corrected_words: list[str], band_width: int = None):
    """
    Perform global DTW alignment between all pre-fix words and corrected words.

    Args:
        band_width: If set, use Sakoe-Chiba band constraint (local matching within band)

    Returns (alignment_path, dtw_object, dist_matrix) for visualization.
    """
    n, m = len(all_prefix_words), len(corrected_words)
    print(f"  Computing distance matrix ({n} x {m})...")

    dist_matrix = np.zeros((n, m), dtype=np.float64)
    for i, w1 in enumerate(all_prefix_words):
        if i % 500 == 0:
            print(f"    Processing row {i}/{n}...")
        for j, w2 in enumerate(corrected_words):
            dist_matrix[i, j] = word_distance(w1, w2)

    print("  Running DTW...")
    if band_width:
        print(f"  Using slanted band constraint (width={band_width})")
        print(f"  (Divergence beyond {band_width} words = potential hallucination)")
        alignment = dtw(
            dist_matrix,
            step_pattern='asymmetric',
            keep_internals=True,
            window_type='slantedband',
            window_args={'window_size': band_width}
        )
    else:
        alignment = dtw(dist_matrix, step_pattern='asymmetric', keep_internals=True)

    return list(zip(alignment.index1, alignment.index2)), alignment, dist_matrix


def map_segments_from_global(
    segments: list[Segment],
    all_prefix_words: list[str],
    corrected_words: list[str],
    alignment_path: list[tuple[int, int]]
) -> list[AlignmentResult]:
    """Map segment boundaries using global alignment path."""
    results = []
    match_threshold = 0.5

    # Build prefix-to-corrected mapping from alignment
    prefix_to_corrected = {}
    for p_idx, c_idx in alignment_path:
        if p_idx not in prefix_to_corrected:
            prefix_to_corrected[p_idx] = []
        prefix_to_corrected[p_idx].append(c_idx)

    # Process each segment
    word_idx = 0
    for seg in segments:
        seg_start_idx = word_idx
        seg_end_idx = word_idx + len(seg.words) - 1

        # Find corrected positions for this segment
        corrected_positions = []
        for i in range(seg_start_idx, seg_end_idx + 1):
            if i in prefix_to_corrected:
                corrected_positions.extend(prefix_to_corrected[i])

        if corrected_positions:
            matched_start = min(corrected_positions)
            matched_end = max(corrected_positions) + 1
        else:
            matched_start = matched_end = 0

        # Calculate match score
        matches = 0
        insertions = []
        deletions = []
        matched_corrected = set(corrected_positions)

        for i, word in enumerate(seg.words):
            prefix_idx = seg_start_idx + i
            if prefix_idx in prefix_to_corrected:
                good_match = False
                for c_idx in prefix_to_corrected[prefix_idx]:
                    if c_idx < len(corrected_words):
                        dist = word_distance(word, corrected_words[c_idx])
                        if dist <= match_threshold:
                            good_match = True
                            break
                if good_match:
                    matches += 1
                else:
                    deletions.append(word)
            else:
                deletions.append(word)

        # Find insertions
        if corrected_positions:
            for c_idx in range(matched_start, matched_end):
                if c_idx not in matched_corrected:
                    insertions.append(corrected_words[c_idx])

        match_score = matches / len(seg.words) if seg.words else 0.0
        matched_text = ' '.join(corrected_words[matched_start:matched_end]) if corrected_positions else ""

        # Debug: log word-level alignment for low scores
        logger = logging.getLogger('dtw_match')
        if match_score < 0.5:
            logger.debug(f"  Low score segment [{seg.id}]:")
            for i, word in enumerate(seg.words):
                prefix_idx = seg_start_idx + i
                if prefix_idx in prefix_to_corrected:
                    c_indices = prefix_to_corrected[prefix_idx]
                    c_words = [corrected_words[c] if c < len(corrected_words) else "?" for c in c_indices]
                    dists = [word_distance(word, corrected_words[c]) if c < len(corrected_words) else 99 for c in c_indices]
                    logger.debug(f"    [{prefix_idx}] '{word}' -> {c_indices} {c_words} dists={dists}")
                else:
                    logger.debug(f"    [{prefix_idx}] '{word}' -> NO MAPPING")

        results.append(AlignmentResult(
            segment_id=seg.id,
            original_text=seg.text,
            matched_text=matched_text,
            start_pos=matched_start,
            end_pos=matched_end,
            match_score=match_score,
            insertions=insertions,
            deletions=deletions
        ))

        word_idx += len(seg.words)

    return results


# =============================================================================
# SUBSEQUENCE DTW ALIGNMENT
# =============================================================================

def align_segment_subsequence(
    segment: Segment,
    corrected_words: list[str],
    search_start: int,
    search_window: int = 10,
    return_alignment: bool = False
):
    """
    Find where segment best matches using subsequence DTW (open_begin + open_end).
    Best when searching for segments in potentially unrelated reference.
    """
    seg_words = segment.words
    if not seg_words:
        return AlignmentResult(
            segment_id=segment.id,
            original_text=segment.text,
            matched_text="",
            start_pos=search_start,
            end_pos=search_start,
            match_score=0.0,
            insertions=[],
            deletions=[]
        )

    window_start = max(0, search_start - 10)
    window_end = min(len(corrected_words), search_start + len(seg_words) + search_window)

    if window_start >= len(corrected_words):
        return AlignmentResult(
            segment_id=segment.id,
            original_text=segment.text,
            matched_text="",
            start_pos=len(corrected_words),
            end_pos=len(corrected_words),
            match_score=0.0,
            insertions=[],
            deletions=list(seg_words)
        )

    window_words = corrected_words[window_start:window_end]
    if not window_words:
        return AlignmentResult(
            segment_id=segment.id,
            original_text=segment.text,
            matched_text="",
            start_pos=window_start,
            end_pos=window_start,
            match_score=0.0,
            insertions=[],
            deletions=list(seg_words)
        )

    n, m = len(seg_words), len(window_words)
    dist_matrix = np.zeros((n, m), dtype=np.float64)
    for i, w1 in enumerate(seg_words):
        for j, w2 in enumerate(window_words):
            dist_matrix[i, j] = word_distance(w1, w2)

    # Subsequence DTW: find best matching window anywhere in reference
    alignment = dtw(
        dist_matrix,
        step_pattern='asymmetric',
        open_begin=True,
        open_end=True,
        keep_internals=True
    )

    path_query = alignment.index1
    path_ref = alignment.index2

    matched_start = window_start + min(path_ref)
    matched_end = window_start + max(path_ref) + 1

    match_threshold = 0.5
    matches = 0
    deletions = []
    insertions = []

    query_to_ref = {}
    for qi, ri in zip(path_query, path_ref):
        if qi not in query_to_ref:
            query_to_ref[qi] = []
        query_to_ref[qi].append(ri)

    matched_ref_indices = set(path_ref)

    for i, word in enumerate(seg_words):
        if i in query_to_ref:
            best_dist = float('inf')
            for ri in query_to_ref[i]:
                dist = dist_matrix[i, ri]
                if dist < best_dist:
                    best_dist = dist
            if best_dist <= match_threshold:
                matches += 1
            else:
                deletions.append(word)
        else:
            deletions.append(word)

    for ri in range(min(path_ref), max(path_ref) + 1):
        if ri not in matched_ref_indices:
            insertions.append(window_words[ri])

    match_score = matches / len(seg_words) if seg_words else 0.0
    matched_text = ' '.join(corrected_words[matched_start:matched_end])

    result = AlignmentResult(
        segment_id=segment.id,
        original_text=segment.text,
        matched_text=matched_text,
        start_pos=matched_start,
        end_pos=matched_end,
        match_score=match_score,
        insertions=insertions,
        deletions=deletions
    )

    if return_alignment:
        return result, alignment, dist_matrix, window_start
    return result


def run_subsequence_alignment(segments: list[Segment], corrected_words: list[str], logger: logging.Logger = None):
    """Run subsequence DTW for each segment."""
    if logger is None:
        logger = logging.getLogger('dtw_match')

    results = []
    current_pos = 0
    best_alignment = None  # Keep the largest segment's alignment for cumulative cost plot

    logger.info(f"Starting subsequence alignment: {len(segments)} segments, {len(corrected_words)} corrected words")

    for i, segment in enumerate(segments):
        if i % 100 == 0:
            print(f"  Processing segment {i}/{len(segments)}...")
            logger.info(f"Progress: segment {i}/{len(segments)}")

        # For segments with 5+ words, capture alignment for potential plotting
        if len(segment.words) >= 5:
            ret = align_segment_subsequence(segment, corrected_words, current_pos, return_alignment=True)
            if isinstance(ret, tuple):
                result, alignment, dist_matrix, window_start = ret
                # Keep the largest segment's alignment for cumulative cost plot
                if best_alignment is None or len(segment.words) > best_alignment[0]:
                    best_alignment = (len(segment.words), segment.id, alignment, dist_matrix)
            else:
                result = ret
        else:
            result = align_segment_subsequence(segment, corrected_words, current_pos)

        results.append(result)

        # Log each segment result
        logger.debug(f"Segment [{segment.id}]: {len(segment.words)} words, "
                     f"matched at {result.start_pos}-{result.end_pos}, "
                     f"score={result.match_score:.1%}, "
                     f"insertions={len(result.insertions)}, deletions={len(result.deletions)}")

        if result.end_pos > current_pos:
            current_pos = result.end_pos

    # Log summary
    avg_score = sum(r.match_score for r in results) / len(results) if results else 0
    total_insertions = sum(len(r.insertions) for r in results)
    total_deletions = sum(len(r.deletions) for r in results)
    logger.info(f"Subsequence alignment complete: avg_score={avg_score:.1%}, "
                f"total_insertions={total_insertions}, total_deletions={total_deletions}")

    return results, best_alignment


# =============================================================================
# CHUNKED MATCHING (10 words at a time, search within next 10 words)
# =============================================================================

@dataclass
class ChunkResult:
    chunk_id: int
    chunk_start: int  # Start position in pre-fix words
    chunk_end: int    # End position in pre-fix words
    chunk_words: list[str]
    matched_start: int  # Start position in corrected words
    matched_end: int    # End position in corrected words
    matched_words: list[str]
    match_score: float
    insertions: list[str]
    deletions: list[str]
    is_anomaly: bool  # True if match is poor


def run_chunked_matching(
    all_prefix_words: list[str],
    corrected_words: list[str],
    chunk_size: int = 10,
    search_window: int = 10,
    logger: logging.Logger = None
) -> tuple[list[ChunkResult], list]:
    """
    Match chunks of words from pre-fix to corrected file.

    Args:
        chunk_size: Number of words per chunk (query size)
        search_window: How many words ahead to search in corrected file

    Returns:
        List of ChunkResult objects
    """
    if logger is None:
        logger = logging.getLogger('dtw_match')

    results = []
    current_corrected_pos = 0
    match_threshold = 0.5

    num_chunks = (len(all_prefix_words) + chunk_size - 1) // chunk_size
    logger.info(f"Processing {num_chunks} chunks of {chunk_size} words each")
    logger.info(f"Search window: {search_window} words")

    for chunk_id in range(num_chunks):
        # Get chunk from pre-fix
        chunk_start = chunk_id * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(all_prefix_words))
        chunk_words = all_prefix_words[chunk_start:chunk_end]

        if not chunk_words:
            continue

        # Define search window in corrected file
        window_start = max(0, current_corrected_pos)
        window_end = min(len(corrected_words), current_corrected_pos + len(chunk_words) + search_window)
        window_words = corrected_words[window_start:window_end]

        logger.debug(f"Chunk [{chunk_id}]: prefix[{chunk_start}:{chunk_end}] -> search corrected[{window_start}:{window_end}]")

        if not window_words:
            # No more words in corrected file
            result = ChunkResult(
                chunk_id=chunk_id,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                chunk_words=chunk_words,
                matched_start=window_start,
                matched_end=window_start,
                matched_words=[],
                match_score=0.0,
                insertions=[],
                deletions=list(chunk_words),
                is_anomaly=True
            )
            results.append(result)
            logger.warning(f"Chunk [{chunk_id}]: No corrected words left to match!")
            continue

        # Build distance matrix for this chunk vs window
        n, m = len(chunk_words), len(window_words)
        dist_matrix = np.zeros((n, m), dtype=np.float64)
        for i, w1 in enumerate(chunk_words):
            for j, w2 in enumerate(window_words):
                dist_matrix[i, j] = word_distance(w1, w2)

        # Run DTW on this small chunk
        alignment = dtw(dist_matrix, step_pattern='symmetric2')

        path_query = alignment.index1
        path_ref = alignment.index2

        matched_start = window_start + min(path_ref)
        matched_end = window_start + max(path_ref) + 1
        matched_words = corrected_words[matched_start:matched_end]

        # Calculate match score
        matches = 0
        deletions = []
        insertions = []

        query_to_ref = {}
        for qi, ri in zip(path_query, path_ref):
            if qi not in query_to_ref:
                query_to_ref[qi] = []
            query_to_ref[qi].append(ri)

        matched_ref_indices = set(path_ref)

        for i, word in enumerate(chunk_words):
            if i in query_to_ref:
                best_dist = float('inf')
                for ri in query_to_ref[i]:
                    dist = dist_matrix[i, ri]
                    if dist < best_dist:
                        best_dist = dist
                if best_dist <= match_threshold:
                    matches += 1
                else:
                    deletions.append(word)
            else:
                deletions.append(word)

        for ri in range(min(path_ref), max(path_ref) + 1):
            if ri not in matched_ref_indices:
                insertions.append(window_words[ri])

        match_score = matches / len(chunk_words) if chunk_words else 0.0
        is_anomaly = match_score < 0.5 or len(insertions) > 3

        result = ChunkResult(
            chunk_id=chunk_id,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            chunk_words=chunk_words,
            matched_start=matched_start,
            matched_end=matched_end,
            matched_words=matched_words,
            match_score=match_score,
            insertions=insertions,
            deletions=deletions,
            is_anomaly=is_anomaly
        )
        results.append(result)

        # Log result
        if is_anomaly:
            logger.warning(f"Chunk [{chunk_id}]: ANOMALY - {match_score:.1%} match, {len(insertions)} insertions, {len(deletions)} deletions")
            logger.warning(f"  Query: {' '.join(chunk_words[:5])}...")
            logger.warning(f"  Matched: {' '.join(matched_words[:5])}...")
            if insertions:
                logger.warning(f"  Insertions: {insertions[:5]}")
            if deletions:
                logger.warning(f"  Deletions: {deletions[:5]}")
        else:
            logger.debug(f"Chunk [{chunk_id}]: OK - {match_score:.1%} match")

        # Advance position in corrected file
        current_corrected_pos = matched_end

    return results


def print_chunk_results(results: list[ChunkResult], logger: logging.Logger = None):
    """Print chunk matching results."""
    if logger is None:
        logger = logging.getLogger('dtw_match')

    print("\n" + "=" * 80)
    print("CHUNK MATCHING RESULTS")
    print("=" * 80)

    anomalies = [r for r in results if r.is_anomaly]
    total_insertions = sum(len(r.insertions) for r in results)
    total_deletions = sum(len(r.deletions) for r in results)
    avg_score = sum(r.match_score for r in results) / len(results) if results else 0

    for result in results:
        status = "ANOMALY" if result.is_anomaly else "OK"
        print(f"\nChunk [{result.chunk_id}] ({status}) - pos {result.matched_start}-{result.matched_end}")
        print(f"  Query ({len(result.chunk_words)} words): {' '.join(result.chunk_words[:8])}{'...' if len(result.chunk_words) > 8 else ''}")
        print(f"  Match score: {result.match_score:.1%}")
        if result.insertions:
            print(f"  Insertions (hallucinations): {result.insertions[:5]}{'...' if len(result.insertions) > 5 else ''}")
        if result.deletions:
            print(f"  Deletions (missing): {result.deletions[:5]}{'...' if len(result.deletions) > 5 else ''}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal chunks: {len(results)}")
    print(f"Average match score: {avg_score:.1%}")
    print(f"Anomalies detected: {len(anomalies)} ({len(anomalies)/len(results)*100:.1f}%)")
    print(f"Total insertions (potential hallucinations): {total_insertions}")
    print(f"Total deletions (missing from corrected): {total_deletions}")

    if anomalies:
        print(f"\nAnomalous chunks (match < 50% or >3 insertions):")
        for r in anomalies[:20]:
            print(f"  Chunk [{r.chunk_id}] - {r.match_score:.1%} match, {len(r.insertions)} ins, {len(r.deletions)} del")

    # Log summary
    logger.info(f"Total chunks: {len(results)}, Anomalies: {len(anomalies)}, Avg score: {avg_score:.1%}")
    logger.info(f"Total insertions: {total_insertions}, Total deletions: {total_deletions}")


def plot_chunk_results(results: list[ChunkResult], prefix_len: int, corrected_len: int, title="Chunked DTW Alignment", save_path=None):
    """
    Plot chunk matching results showing alignment and anomalies.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    chunk_ids = [r.chunk_id for r in results]
    match_scores = [r.match_score for r in results]
    is_anomaly = [r.is_anomaly for r in results]

    # Plot 1: Alignment map (chunk position vs corrected position)
    ax1 = axes[0, 0]
    prefix_positions = [(r.chunk_start + r.chunk_end) / 2 for r in results]
    corrected_positions = [(r.matched_start + r.matched_end) / 2 for r in results]
    colors = ['red' if a else 'green' for a in is_anomaly]

    ax1.scatter(corrected_positions, prefix_positions, c=colors, s=20, alpha=0.6)
    # Draw ideal diagonal
    max_pos = max(prefix_len, corrected_len)
    ax1.plot([0, max_pos], [0, max_pos], 'b--', alpha=0.3, label='Ideal diagonal')
    ax1.set_xlabel('Corrected file (word position)')
    ax1.set_ylabel('Pre-fix file (word position)')
    ax1.set_title('Chunk Alignment Map (green=OK, red=anomaly)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Match score by chunk
    ax2 = axes[0, 1]
    colors_score = ['red' if a else 'green' for a in is_anomaly]
    ax2.bar(chunk_ids, match_scores, color=colors_score, alpha=0.7)
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50% threshold')
    ax2.set_xlabel('Chunk ID')
    ax2.set_ylabel('Match Score')
    ax2.set_title('Match Score by Chunk')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative position drift
    ax3 = axes[1, 0]
    expected_positions = [r.chunk_start * (corrected_len / prefix_len) for r in results]
    actual_positions = [r.matched_start for r in results]
    drift = [actual - expected for actual, expected in zip(actual_positions, expected_positions)]

    ax3.plot(chunk_ids, drift, 'b-', linewidth=1, alpha=0.7)
    ax3.scatter(chunk_ids, drift, c=colors, s=20, alpha=0.6)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Chunk ID')
    ax3.set_ylabel('Position Drift (words)')
    ax3.set_title('Alignment Drift from Expected Position')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Insertions and Deletions
    ax4 = axes[1, 1]
    insertions = [len(r.insertions) for r in results]
    deletions = [len(r.deletions) for r in results]

    ax4.bar(chunk_ids, deletions, color='orange', alpha=0.7, label='Deletions')
    ax4.bar(chunk_ids, insertions, bottom=deletions, color='red', alpha=0.7, label='Insertions')
    ax4.set_xlabel('Chunk ID')
    ax4.set_ylabel('Count')
    ax4.set_title('Insertions (hallucinations) and Deletions per Chunk')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")
    plt.show()


def detect_drop(results: list[AlignmentResult], threshold: float = 0.25,
                word_count_threshold: int = 15, ma_window: int = 10,
                logger: logging.Logger = None):
    """
    Detect where alignment quality drops, indicating potential hallucinations.

    Args:
        threshold: Score threshold (default 0.25)
        word_count_threshold: Number of words below threshold to trigger detection (default 15)
        ma_window: Window size for moving average (default 10 segments)

    Returns:
        dict with:
        - moving_avg: list of moving average scores
        - first_drop_idx: index where first drop occurs (or None)
        - first_drop_type: 'words' or 'moving_avg' or None
        - words_below_regions: list of (start_idx, end_idx, word_count) for regions below threshold
    """
    if logger is None:
        logger = logging.getLogger('dtw_match')

    match_scores = [r.match_score for r in results]
    # Get word count for each segment by tokenizing original_text
    word_counts = [len(tokenize(r.original_text)) for r in results]
    n = len(match_scores)

    # Log each segment's data
    for r, wc in zip(results, word_counts):
        logger.debug(f"  Segment [{r.segment_id}]: {r.match_score:.0%} ({wc} words) pos {r.start_pos}-{r.end_pos}")
        logger.debug(f"    Original: {r.original_text[:80]}")
        logger.debug(f"    Matched:  {r.matched_text[:80] if r.matched_text else '(empty)'}")

    # Calculate moving average (window of ma_window segments)
    moving_avg = []
    for i in range(n):
        start = max(0, i - ma_window + 1)
        window = match_scores[start:i + 1]
        moving_avg.append(sum(window) / len(window))

    # Find first place where moving average drops below threshold
    # Report the first segment of that window (start of the problematic region)
    first_ma_drop = None
    ma_crossed_at = None
    for i, ma in enumerate(moving_avg):
        if ma < threshold:
            ma_crossed_at = i
            # Report the start of the window (first segment that contributed to this MA)
            first_ma_drop = max(0, i - ma_window + 1)
            break

    # Find first place with word_count_threshold WORDS below threshold
    # Count cumulative words from segments with score < threshold
    first_words_drop = None
    words_below_regions = []
    cumulative_words = 0
    start_idx = None

    for i, (score, wc) in enumerate(zip(match_scores, word_counts)):
        if score < threshold:
            if start_idx is None:
                start_idx = i
                cumulative_words = 0
            cumulative_words += wc
            if cumulative_words >= word_count_threshold and first_words_drop is None:
                first_words_drop = start_idx
                words_below_regions.append((start_idx, i, cumulative_words))
        else:
            if start_idx is not None and cumulative_words >= word_count_threshold:
                words_below_regions.append((start_idx, i - 1, cumulative_words))
            start_idx = None
            cumulative_words = 0

    # Handle case where sequence ends while in a low stretch
    if start_idx is not None and cumulative_words >= word_count_threshold:
        words_below_regions.append((start_idx, n - 1, cumulative_words))

    # Determine which drop occurs first
    first_drop_idx = None
    first_drop_type = None

    if first_words_drop is not None and first_ma_drop is not None:
        if first_words_drop <= first_ma_drop:
            first_drop_idx = first_words_drop
            first_drop_type = 'words'
        else:
            first_drop_idx = first_ma_drop
            first_drop_type = 'moving_avg'
    elif first_words_drop is not None:
        first_drop_idx = first_words_drop
        first_drop_type = 'words'
    elif first_ma_drop is not None:
        first_drop_idx = first_ma_drop
        first_drop_type = 'moving_avg'

    # Log findings
    logger.info(f"Drop detection: threshold={threshold}, word_count_threshold={word_count_threshold}, ma_window={ma_window}")
    if first_words_drop is not None:
        logger.info(f"First {word_count_threshold}+ words below threshold at segment index {first_words_drop}")
    if first_ma_drop is not None:
        r = results[first_ma_drop]
        logger.info(f"MA dropped below {threshold} at index {ma_crossed_at}, window start: index {first_ma_drop} (Segment [{r.segment_id}], score={r.match_score:.1%})")
    if first_drop_idx is not None:
        logger.info(f"==> First drop at segment index {first_drop_idx} (type: {first_drop_type})")
        if first_drop_idx < len(results):
            r = results[first_drop_idx]
            logger.info(f"    Segment [{r.segment_id}]: score={r.match_score:.1%}, pos={r.start_pos}-{r.end_pos}")
    else:
        logger.info("No significant drop detected")

    return {
        'moving_avg': moving_avg,
        'first_drop_idx': first_drop_idx,
        'first_drop_type': first_drop_type,
        'words_below_regions': words_below_regions,
        'first_ma_drop': first_ma_drop,
        'ma_crossed_at': ma_crossed_at,
        'first_words_drop': first_words_drop,
        'word_counts': word_counts,
    }


def detect_vertical_jumps(alignment, min_jump_size: int = 5, logger: logging.Logger = None):
    """
    Detect vertical jumps in the DTW alignment path.

    Vertical jumps occur when the query (prefix) index advances while the reference
    (corrected) index stays the same - indicating content in the prefix that was
    removed/condensed in the corrected file.

    Args:
        alignment: DTW alignment object with index1/index2 attributes
        min_jump_size: Minimum jump size to report (default: 5 words)
        logger: Logger instance

    Returns:
        List of dicts with jump info: corrected_idx, prefix_start, prefix_end, jump_size
    """
    if logger is None:
        logger = logging.getLogger('dtw_match')

    idx1 = np.array(alignment.index1)  # prefix indices
    idx2 = np.array(alignment.index2)  # corrected indices

    # Detect vertical segments: where corrected stays same but prefix advances
    diff_idx1 = np.diff(idx1)
    diff_idx2 = np.diff(idx2)

    vertical_mask = (diff_idx2 == 0) & (diff_idx1 > 0)
    vertical_positions = np.where(vertical_mask)[0]

    if len(vertical_positions) == 0:
        logger.info("No vertical jumps detected in alignment path")
        return []

    # Group consecutive vertical positions into segments
    jumps_raw = []
    start = vertical_positions[0]
    for i in range(1, len(vertical_positions)):
        if vertical_positions[i] != vertical_positions[i-1] + 1:
            jumps_raw.append((start, vertical_positions[i-1]))
            start = vertical_positions[i]
    jumps_raw.append((start, vertical_positions[-1]))

    # Convert to structured results
    jumps = []
    for s, e in jumps_raw:
        prefix_start = idx1[s]
        prefix_end = idx1[e+1] if e+1 < len(idx1) else idx1[e]
        jump_size = prefix_end - prefix_start
        if jump_size >= min_jump_size:
            jumps.append({
                'corrected_idx': int(idx2[s]),
                'prefix_start': int(prefix_start),
                'prefix_end': int(prefix_end),
                'jump_size': int(jump_size)
            })

    # Sort by jump size descending
    jumps.sort(key=lambda x: x['jump_size'], reverse=True)

    # Log results
    if jumps:
        logger.info(f"Detected {len(jumps)} vertical jumps (>= {min_jump_size} words)")
        for i, j in enumerate(jumps[:10]):
            logger.info(f"  {i+1}. Corrected idx {j['corrected_idx']}: "
                       f"prefix {j['prefix_start']}-{j['prefix_end']} ({j['jump_size']} words)")

    return jumps


def plot_subsequence_alignments(results: list[AlignmentResult], corrected_len: int,
                                 title="Subsequence DTW Alignment Map", save_path=None,
                                 drop_info: dict = None):
    """
    Plot a coverage map showing where each segment matched in the corrected file.
    Includes moving average and drop detection visualization.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    segment_ids = [r.segment_id for r in results]
    indices = list(range(len(results)))
    start_positions = [r.start_pos for r in results]
    end_positions = [r.end_pos for r in results]
    match_scores = [r.match_score for r in results]

    # Plot 1: Segment positions and match scores
    ax1 = axes[0]
    colors = plt.cm.RdYlGn(match_scores)

    for i, (seg_id, start, end, score) in enumerate(zip(segment_ids, start_positions, end_positions, match_scores)):
        ax1.barh(seg_id, end - start, left=start, height=0.8, color=colors[i], alpha=0.7)

    ax1.set_xlabel('Position in corrected file (word index)')
    ax1.set_ylabel('Segment ID')
    ax1.set_title('Segment Alignments (color = match score: red=low, green=high)')
    ax1.set_xlim(0, corrected_len)

    # Mark first drop position
    if drop_info and drop_info['first_drop_idx'] is not None:
        drop_idx = drop_info['first_drop_idx']
        if drop_idx < len(results):
            drop_pos = results[drop_idx].start_pos
            ax1.axvline(x=drop_pos, color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'First drop at pos {drop_pos}')
            ax1.legend()

    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label='Match Score')

    # Plot 2: Match score progression with moving average
    ax2 = axes[1]
    ax2.plot(indices, match_scores, 'b-', linewidth=1, alpha=0.5, label='Raw score')
    ax2.scatter(indices, match_scores, c=match_scores, cmap='RdYlGn', s=20, alpha=0.6)

    # Plot moving average
    if drop_info and 'moving_avg' in drop_info:
        moving_avg = drop_info['moving_avg']
        ax2.plot(indices, moving_avg, 'purple', linewidth=2, alpha=0.9, label='Moving avg (10 segments)')

    # Threshold line
    ax2.axhline(y=0.25, color='red', linestyle='--', linewidth=2, alpha=0.8, label='0.25 threshold')

    # Mark first drop
    if drop_info and drop_info['first_drop_idx'] is not None:
        drop_idx = drop_info['first_drop_idx']
        ax2.axvline(x=drop_idx, color='red', linestyle='-', linewidth=2, alpha=0.8)
        ax2.annotate(f'First drop\n(segment {drop_idx})\nType: {drop_info["first_drop_type"]}',
                     xy=(drop_idx, 0.25), xytext=(drop_idx + 5, 0.5),
                     fontsize=10, color='red',
                     arrowprops=dict(arrowstyle='->', color='red'))

    ax2.set_xlabel('Segment Index')
    ax2.set_ylabel('Match Score')
    ax2.set_title('Match Score with Moving Average (window=10)')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def plot_cumulative_cost_landscape(alignment, dist_matrix, title="DTW Cumulative Cost Landscape", save_path=None, band_width=None):
    """
    Plot the cumulative cost matrix as a topographic landscape with the optimal path.

    Args:
        band_width: If set, draw slanted band boundaries on the plot.
    """
    # Get cumulative cost matrix
    cost_matrix = alignment.costMatrix

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Local cost matrix (distance matrix)
    ax1 = axes[0]
    im1 = ax1.imshow(dist_matrix, aspect='auto', origin='lower', cmap='terrain')
    ax1.plot(alignment.index2, alignment.index1, 'r-', linewidth=1.5, alpha=0.8, label='Optimal path')
    ax1.set_xlabel('Corrected file (word index)')
    ax1.set_ylabel('Pre-fix file (word index)')
    ax1.set_title('Local Cost Matrix (Word Distances)')
    plt.colorbar(im1, ax=ax1, label='Distance')
    ax1.legend(loc='upper left')

    # Plot 2: Cumulative cost matrix with contours (topographic)
    ax2 = axes[1]
    im2 = ax2.imshow(cost_matrix, aspect='auto', origin='lower', cmap='terrain')

    # Add contour lines for topographic effect
    n, m = cost_matrix.shape
    x = np.arange(m)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)

    # Determine contour levels based on data range
    vmin, vmax = cost_matrix.min(), cost_matrix.max()
    levels = np.linspace(vmin, vmax, 20)
    ax2.contour(X, Y, cost_matrix, levels=levels, colors='black', alpha=0.3, linewidths=0.5)

    # Overlay optimal warping path
    ax2.plot(alignment.index2, alignment.index1, 'r-', linewidth=2, alpha=0.9, label='Optimal path')

    ax2.set_xlabel('Corrected file (word index)')
    ax2.set_ylabel('Pre-fix file (word index)')
    ax2.set_title('Cumulative Cost Landscape (Topographic)')
    plt.colorbar(im2, ax=ax2, label='Cumulative cost')
    ax2.legend(loc='upper left')

    # Draw band boundaries if band_width is set
    if band_width:
        n, m = dist_matrix.shape
        i_vals = np.array([0, n - 1])
        slope = m / n

        # Slanted band boundaries
        upper = slope * i_vals + band_width
        lower = slope * i_vals - band_width

        for ax in [ax1, ax2]:
            ax.plot(upper, i_vals, 'y--', linewidth=2, alpha=0.9, label='Band boundary')
            ax.plot(lower, i_vals, 'y--', linewidth=2, alpha=0.9)
            ax.set_xlim(0, m - 1)
            ax.set_ylim(0, n - 1)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper left')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")
    plt.show()


def print_results(results: list[AlignmentResult]):
    """Print alignment results and summary."""
    print("\n" + "=" * 80)
    print("ALIGNMENT RESULTS")
    print("=" * 80)

    for result in results:
        print(f"\nSegment [{result.segment_id}] (pos {result.start_pos}-{result.end_pos})")
        print(f"  Original: {result.original_text[:60]}{'...' if len(result.original_text) > 60 else ''}")
        print(f"  Match score: {result.match_score:.1%}")

        if result.insertions:
            print(f"  Insertions (potential hallucinations): {result.insertions[:5]}{'...' if len(result.insertions) > 5 else ''}")
        if result.deletions:
            print(f"  Deletions (missing from corrected): {result.deletions[:5]}{'...' if len(result.deletions) > 5 else ''}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_segments = len(results)
    if total_segments == 0:
        print("\nNo segments to analyze.")
        return

    avg_score = sum(r.match_score for r in results) / total_segments
    perfect_matches = [r for r in results if r.match_score == 1.0]
    high_matches = [r for r in results if r.match_score >= 0.8]
    low_match_segments = [r for r in results if r.match_score < 0.5]
    high_insertion_segments = [r for r in results if len(r.insertions) > 3]

    total_insertions = sum(len(r.insertions) for r in results)
    total_deletions = sum(len(r.deletions) for r in results)

    print(f"\nTotal segments: {total_segments}")
    print(f"Average match score: {avg_score:.1%}")
    print(f"Perfect matches (100%): {len(perfect_matches)} ({len(perfect_matches)/total_segments*100:.1f}%)")
    print(f"High matches (>=80%): {len(high_matches)} ({len(high_matches)/total_segments*100:.1f}%)")
    print(f"Low matches (<50%): {len(low_match_segments)} ({len(low_match_segments)/total_segments*100:.1f}%)")
    print(f"\nTotal insertions (potential hallucinations): {total_insertions}")
    print(f"Total deletions (missing from corrected): {total_deletions}")
    print(f"Segments with high insertions (>3): {len(high_insertion_segments)}")

    if high_insertion_segments:
        print("\nSegments with potential hallucinations (>3 insertions):")
        for r in high_insertion_segments[:10]:
            print(f"  [{r.segment_id}] - {len(r.insertions)} insertions: {r.insertions[:3]}...")

    if low_match_segments:
        print("\nSegments with low match scores (<50%):")
        for r in low_match_segments[:15]:
            print(f"  [{r.segment_id}] - {r.match_score:.1%} match - deletions: {r.deletions[:3]}{'...' if len(r.deletions) > 3 else ''}")


def main():
    parser = argparse.ArgumentParser(description='DTW Text Matching for Hebrew Transcription Alignment')
    parser.add_argument('--mode', choices=['global', 'subsequence', 'banded', 'chunked'], default='global',
                        help='Alignment mode: global, subsequence, banded, or chunked')
    parser.add_argument('--band-width', type=int, default=10,
                        help='Band width for banded mode (default: 10 words)')
    parser.add_argument('--chunk-size', type=int, default=10,
                        help='Chunk size for chunked mode (default: 10 words)')
    parser.add_argument('--search-window', type=int, default=10,
                        help='Search window for chunked mode (default: 10 words)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file path (default: dtw_match_TIMESTAMP.log)')
    parser.add_argument('--prefix', default='154556.pre-fix.txt', help='Pre-fix file path')
    parser.add_argument('--corrected', default='154556.txt', help='Corrected file path')
    parser.add_argument('--swap', action='store_true', help='Swap prefix and corrected files')
    parser.add_argument('--save-plot', metavar='PATH', help='Save plot to file instead of showing')
    parser.add_argument('--no-plot', action='store_true', help='Skip the plot')
    args = parser.parse_args()

    # Swap files if requested
    if args.swap:
        args.prefix, args.corrected = args.corrected, args.prefix

    # Setup logging
    logger = setup_logging(args.log_file)

    print("=" * 80)
    print(f"DTW Text Matching - Mode: {args.mode.upper()}")
    print("=" * 80)
    logger.info(f"Mode: {args.mode}")

    # Parse files
    print(f"\nParsing {args.prefix}...")
    if args.swap:
        segments = parse_file_as_chunks(args.prefix, chunk_size=args.chunk_size)
        print(f"  Found {len(segments)} chunks ({args.chunk_size} words each)")
        logger.info(f"Parsed {args.prefix} as chunks: {len(segments)} chunks ({args.chunk_size} words each)")
    else:
        segments = parse_prefix_file(args.prefix)
        print(f"  Found {len(segments)} segments")
        logger.info(f"Parsed {args.prefix}: {len(segments)} segments")

    print(f"\nParsing {args.corrected}...")
    if args.swap:
        # When swapped, corrected is the pre-fix file - parse only text, not markers/timestamps
        corrected_words = parse_prefix_file_as_words(args.corrected)
        print(f"  Found {len(corrected_words)} words (extracted from segments)")
        logger.info(f"Parsed {args.corrected} as segment text: {len(corrected_words)} words")
    else:
        corrected_words = parse_corrected_file(args.corrected)
        print(f"  Found {len(corrected_words)} words")
        logger.info(f"Parsed {args.corrected}: {len(corrected_words)} words")

    alignment_obj = None
    dist_matrix = None
    chunk_results = None

    if args.mode == 'chunked':
        # Chunked matching - 10 words at a time, search within next 10 words
        all_prefix_words = []
        for seg in segments:
            all_prefix_words.extend(seg.words)
        print(f"  Total pre-fix words: {len(all_prefix_words)}")
        logger.info(f"Total pre-fix words: {len(all_prefix_words)}")

        print(f"\nRunning chunked matching (chunk={args.chunk_size}, window={args.search_window})...")
        logger.info(f"Chunk size: {args.chunk_size}, Search window: {args.search_window}")
        chunk_results = run_chunked_matching(
            all_prefix_words,
            corrected_words,
            chunk_size=args.chunk_size,
            search_window=args.search_window,
            logger=logger
        )
        print_chunk_results(chunk_results, logger)

        # Plot chunk results
        if not args.no_plot:
            print("\nGenerating chunk alignment plot...")
            plot_chunk_results(
                chunk_results,
                len(all_prefix_words),
                len(corrected_words),
                title=f"Chunked DTW (chunk={args.chunk_size}, window={args.search_window})",
                save_path=args.save_plot
            )

    elif args.mode in ('global', 'banded'):
        # Global/Banded DTW - best for same content with corrections
        all_prefix_words = []
        for seg in segments:
            all_prefix_words.extend(seg.words)
        print(f"  Total pre-fix words: {len(all_prefix_words)}")

        band = args.band_width if args.mode == 'banded' else None
        print(f"\nRunning {'banded' if band else 'global'} DTW alignment...")
        alignment_path, alignment_obj, dist_matrix = global_dtw_alignment(all_prefix_words, corrected_words, band_width=band)
        print(f"  Alignment path length: {len(alignment_path)}")

        # Detect vertical jumps in the alignment path
        print("\nDetecting vertical jumps in alignment path...")
        vertical_jumps = detect_vertical_jumps(alignment_obj, min_jump_size=5, logger=logger)
        if vertical_jumps:
            print(f"\n*** VERTICAL JUMPS DETECTED ({len(vertical_jumps)} total) ***")
            print("(Content in prefix removed/condensed in corrected file)")
            for i, j in enumerate(vertical_jumps[:10]):
                print(f"  {i+1}. Corrected index {j['corrected_idx']}: "
                      f"prefix words {j['prefix_start']}-{j['prefix_end']} ({j['jump_size']} words skipped)")
            if len(vertical_jumps) > 10:
                print(f"  ... and {len(vertical_jumps) - 10} more (see log for details)")
        else:
            print("  No significant vertical jumps detected")

        print("\nMapping segments from alignment...")
        results = map_segments_from_global(segments, all_prefix_words, corrected_words, alignment_path)
        print_results(results)

    elif args.mode == 'subsequence':
        # Subsequence DTW - best for searching in longer/different reference
        all_prefix_words = []
        for seg in segments:
            all_prefix_words.extend(seg.words)
        print(f"  Total pre-fix words: {len(all_prefix_words)}")

        print("\nRunning subsequence DTW alignment...")
        logger.info(f"Total pre-fix words: {len(all_prefix_words)}")
        results, best_alignment = run_subsequence_alignment(segments, corrected_words, logger)
        print_results(results)

    # Show plot (not for chunked mode - it has its own summary)
    if not args.no_plot and args.mode != 'chunked':
        if args.mode in ('global', 'banded') and alignment_obj is not None and dist_matrix is not None:
            mode_name = f"Banded (width={args.band_width})" if args.mode == 'banded' else "Global"

            # Show match score map (same as subsequence mode)
            print("\nAnalyzing alignment quality drop...")
            drop_info = detect_drop(results, threshold=0.25, word_count_threshold=15, ma_window=10, logger=logger)

            if drop_info['first_drop_idx'] is not None:
                print(f"\n*** FIRST DROP DETECTED ***")
                print(f"  At segment index: {drop_info['first_drop_idx']}")
                print(f"  Detection type: {drop_info['first_drop_type']}")
                r = results[drop_info['first_drop_idx']]
                print(f"  Segment [{r.segment_id}]: score={r.match_score:.1%}")
                print(f"  Position in corrected file: {r.start_pos}")
            else:
                print("\nNo significant drop detected (threshold=0.25)")

            print("\nGenerating match score map...")
            save_path_map = args.save_plot.replace('.png', '_scores.png') if args.save_plot else None
            plot_subsequence_alignments(
                results,
                len(corrected_words),
                title=f"DTW Match Scores ({mode_name}): {args.prefix} → {args.corrected}",
                save_path=save_path_map,
                drop_info=drop_info
            )

            print("\nGenerating cumulative cost landscape plot...")
            plot_cumulative_cost_landscape(
                alignment_obj,
                dist_matrix,
                title=f"DTW Alignment ({mode_name}): {args.prefix} → {args.corrected}",
                save_path=args.save_plot,
                band_width=args.band_width if args.mode == 'banded' else None
            )
        elif args.mode == 'subsequence':
            # Detect drop in alignment quality
            print("\nAnalyzing alignment quality drop...")
            drop_info = detect_drop(results, threshold=0.25, word_count_threshold=15, ma_window=10, logger=logger)

            if drop_info['first_drop_idx'] is not None:
                print(f"\n*** FIRST DROP DETECTED ***")
                print(f"  At segment index: {drop_info['first_drop_idx']}")
                print(f"  Detection type: {drop_info['first_drop_type']}")
                r = results[drop_info['first_drop_idx']]
                print(f"  Segment [{r.segment_id}]: score={r.match_score:.1%}")
                print(f"  Position in corrected file: {r.start_pos}")
                if drop_info['first_drop_type'] == 'words':
                    print(f"  (15+ words accumulated below threshold starting here)")
                else:
                    print(f"  (Moving average of 10 segments dropped below 0.25)")
            else:
                print("\nNo significant drop detected (threshold=0.25)")

            # Show the alignment map with drop info
            print("\nGenerating subsequence alignment map...")
            plot_subsequence_alignments(
                results,
                len(corrected_words),
                title=f"DTW Alignment (Subsequence): {args.prefix} → {args.corrected}",
                save_path=args.save_plot,
                drop_info=drop_info
            )
            # Then show cumulative cost for the largest segment
            if best_alignment is not None:
                num_words, seg_id, alignment_obj, dist_matrix = best_alignment
                print(f"\nGenerating cumulative cost landscape for segment [{seg_id}] ({num_words} words)...")
                save_path2 = args.save_plot.replace('.png', '_cost.png') if args.save_plot else None
                plot_cumulative_cost_landscape(
                    alignment_obj,
                    dist_matrix,
                    title=f"Cumulative Cost Landscape - Segment [{seg_id}]",
                    save_path=save_path2
                )



if __name__ == "__main__":
    main()
