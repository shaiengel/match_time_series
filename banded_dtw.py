"""
Banded DTW Text Matching for Hebrew Transcription Alignment

Aligns segments from a pre-fix transcription file to regions in an LLM-corrected file
using banded (Sakoe-Chiba) DTW to detect hallucinations and content removal.
"""

from dataclasses import dataclass
import re
import sys
import argparse
import logging
import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')


def setup_logging(log_file: str = None):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger('dtw_match')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

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
    insertions: list[str]
    deletions: list[str]


def tokenize(text: str) -> list[str]:
    text = re.sub(r'[^\w\s\u0590-\u05FF]', ' ', text)
    return [w for w in text.split() if w.strip()]


def parse_prefix_file(path: str) -> list[Segment]:
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
                segments.append(Segment(id=seg_id, timestamp=timestamp, text=text, words=words))
    return segments


def parse_corrected_file(path: str) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.replace('\n', ' ')
    return tokenize(text)


def parse_prefix_file_as_words(path: str) -> list[str]:
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
    if w1 == w2:
        return 0.0
    edit_dist = levenshtein_distance(w1, w2)
    max_len = max(len(w1), len(w2))
    return edit_dist / max_len if max_len > 0 else 0.0


# =============================================================================
# BANDED DTW ALIGNMENT
# =============================================================================

def banded_dtw_alignment(all_prefix_words: list[str], corrected_words: list[str], band_width: int = 200, step_pattern: str = 'asymmetric'):
    n, m = len(all_prefix_words), len(corrected_words)
    print(f"  Computing distance matrix ({n} x {m})...")

    dist_matrix = np.zeros((n, m), dtype=np.float64)
    for i, w1 in enumerate(all_prefix_words):
        if i % 500 == 0:
            print(f"    Processing row {i}/{n}...")
        for j, w2 in enumerate(corrected_words):
            dist_matrix[i, j] = word_distance(w1, w2)

    print(f"  Running DTW with slanted band constraint (width={band_width})...")
    print(f"  (Divergence beyond {band_width} words = potential hallucination)")
    alignment = dtw(
        dist_matrix,
        step_pattern=step_pattern,
        keep_internals=True,
        window_type='slantedband',
        window_args={'window_size': band_width}
    )

    return list(zip(alignment.index1, alignment.index2)), alignment, dist_matrix


def map_segments_from_global(
    segments: list[Segment],
    all_prefix_words: list[str],
    corrected_words: list[str],
    alignment_path: list[tuple[int, int]],
    match_threshold: float = 0.5
) -> list[AlignmentResult]:
    results = []
    logger = logging.getLogger('dtw_match')

    prefix_to_corrected = {}
    for p_idx, c_idx in alignment_path:
        if p_idx not in prefix_to_corrected:
            prefix_to_corrected[p_idx] = []
        prefix_to_corrected[p_idx].append(c_idx)

    word_idx = 0
    for seg in segments:
        seg_start_idx = word_idx
        seg_end_idx = word_idx + len(seg.words) - 1

        corrected_positions = []
        for i in range(seg_start_idx, seg_end_idx + 1):
            if i in prefix_to_corrected:
                corrected_positions.extend(prefix_to_corrected[i])

        if corrected_positions:
            matched_start = min(corrected_positions)
            matched_end = max(corrected_positions) + 1
        else:
            matched_start = matched_end = 0

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

        if corrected_positions:
            for c_idx in range(matched_start, matched_end):
                if c_idx not in matched_corrected:
                    insertions.append(corrected_words[c_idx])

        match_score = matches / len(seg.words) if seg.words else 0.0
        matched_text = ' '.join(corrected_words[matched_start:matched_end]) if corrected_positions else ""

        if match_score < match_threshold:
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
# VERTICAL JUMP DETECTION
# =============================================================================

def detect_vertical_jumps(alignment, min_jump_size: int = 5, logger: logging.Logger = None):
    if logger is None:
        logger = logging.getLogger('dtw_match')

    idx1 = np.array(alignment.index1)
    idx2 = np.array(alignment.index2)

    diff_idx1 = np.diff(idx1)
    diff_idx2 = np.diff(idx2)

    vertical_mask = (diff_idx2 == 0) & (diff_idx1 > 0)
    vertical_positions = np.where(vertical_mask)[0]

    if len(vertical_positions) == 0:
        logger.info("No vertical jumps detected in alignment path")
        return []

    jumps_raw = []
    start = vertical_positions[0]
    for i in range(1, len(vertical_positions)):
        if vertical_positions[i] != vertical_positions[i-1] + 1:
            jumps_raw.append((start, vertical_positions[i-1]))
            start = vertical_positions[i]
    jumps_raw.append((start, vertical_positions[-1]))

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

    jumps.sort(key=lambda x: x['jump_size'], reverse=True)

    if jumps:
        logger.info(f"Detected {len(jumps)} vertical jumps (>= {min_jump_size} words)")
        for i, j in enumerate(jumps[:10]):
            logger.info(f"  {i+1}. Corrected idx {j['corrected_idx']}: "
                       f"prefix {j['prefix_start']}-{j['prefix_end']} ({j['jump_size']} words)")

    return jumps


# =============================================================================
# DROP DETECTION
# =============================================================================

def detect_drop(results: list[AlignmentResult], threshold: float = 0.25,
                word_count_threshold: int = 15, ma_window: int = 10,
                logger: logging.Logger = None):
    if logger is None:
        logger = logging.getLogger('dtw_match')

    match_scores = [r.match_score for r in results]
    word_counts = [len(tokenize(r.original_text)) for r in results]
    n = len(match_scores)

    for r, wc in zip(results, word_counts):
        logger.debug(f"  Segment [{r.segment_id}]: {r.match_score:.0%} ({wc} words) pos {r.start_pos}-{r.end_pos}")
        logger.debug(f"    Original: {r.original_text[:80]}")
        logger.debug(f"    Matched:  {r.matched_text[:80] if r.matched_text else '(empty)'}")

    moving_avg = []
    for i in range(n):
        start = max(0, i - ma_window + 1)
        window = match_scores[start:i + 1]
        moving_avg.append(sum(window) / len(window))

    first_ma_drop = None
    ma_crossed_at = None
    for i, ma in enumerate(moving_avg):
        if ma < threshold:
            ma_crossed_at = i
            first_ma_drop = max(0, i - ma_window + 1)
            break

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

    if start_idx is not None and cumulative_words >= word_count_threshold:
        words_below_regions.append((start_idx, n - 1, cumulative_words))

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


# =============================================================================
# TRUNCATED FILE CREATION
# =============================================================================

def find_cutoff_index(vertical_jumps: list[dict], drop_info: dict,
                      results: list, jump_threshold: int = 40,
                      logger: logging.Logger = None):
    if logger is None:
        logger = logging.getLogger('dtw_match')

    # Find first vertical jump > jump_threshold words (earliest by corrected_idx)
    large_jumps = [j for j in vertical_jumps if j['jump_size'] > jump_threshold]
    jump_cutoff = None
    if large_jumps:
        large_jumps.sort(key=lambda x: x['corrected_idx'])
        jump_cutoff = large_jumps[0]['corrected_idx']
        logger.info(f"First vertical jump > {jump_threshold} words at corrected index {jump_cutoff} "
                    f"({large_jumps[0]['jump_size']} words)")

    # Find first drop position in corrected file
    drop_cutoff = None
    if drop_info and drop_info['first_drop_idx'] is not None:
        drop_idx = drop_info['first_drop_idx']
        if drop_idx < len(results):
            drop_cutoff = results[drop_idx].start_pos
            logger.info(f"First drop at corrected index {drop_cutoff} "
                       f"(segment index {drop_idx})")

    candidates = [c for c in [jump_cutoff, drop_cutoff] if c is not None]
    if not candidates:
        logger.info("No cutoff point found (no large vertical jumps or drops detected)")
        return None

    cutoff = min(candidates)
    source = 'vertical_jump' if (jump_cutoff is not None and cutoff == jump_cutoff) else 'drop'
    logger.info(f"Cutoff index: {cutoff} (source: {source})")
    return cutoff


def build_word_char_spans(text: str) -> list[tuple[int, int]]:
    cleaned = re.sub(r'[^\w\s\u0590-\u05FF]', ' ', text)
    spans = []
    i = 0
    while i < len(cleaned):
        while i < len(cleaned) and cleaned[i].isspace():
            i += 1
        if i >= len(cleaned):
            break
        start = i
        while i < len(cleaned) and not cleaned[i].isspace():
            i += 1
        spans.append((start, i))
    return spans


def identify_replacements(
    results: list[AlignmentResult],
    segments: list[Segment],
    all_prefix_words: list[str],
    corrected_words: list[str],
    alignment_path: list[tuple[int, int]],
    cutoff_index: int,
    low_score_threshold: float = 0.5,
    high_dist_threshold: float = 0.7,
    match_threshold: float = 0.5,
    logger: logging.Logger = None
) -> list[dict]:
    if logger is None:
        logger = logging.getLogger('dtw_match')

    prefix_to_corrected = {}
    corrected_to_prefix = {}
    for p_idx, c_idx in alignment_path:
        if p_idx not in prefix_to_corrected:
            prefix_to_corrected[p_idx] = []
        prefix_to_corrected[p_idx].append(c_idx)
        if c_idx not in corrected_to_prefix:
            corrected_to_prefix[c_idx] = []
        corrected_to_prefix[c_idx].append(p_idx)

    def is_anchored_externally(c_idx, seg_start, seg_end):
        """Check if corrected word is well-matched by a prefix word outside the current segment."""
        for p_idx in corrected_to_prefix.get(c_idx, []):
            if p_idx < seg_start or p_idx > seg_end:
                dist = word_distance(all_prefix_words[p_idx], corrected_words[c_idx])
                if dist <= match_threshold:
                    return True, p_idx
        return False, None

    replacements = []
    word_idx = 0
    for seg, result in zip(segments, results):
        seg_start_idx = word_idx
        seg_end_idx = word_idx + len(seg.words) - 1
        word_idx += len(seg.words)

        if result.match_score >= low_score_threshold:
            continue
        if result.end_pos > cutoff_index:
            continue

        # Build per-word info: (prefix_idx, prefix_word, corrected_idx, distance)
        word_info = []
        for i, word in enumerate(seg.words):
            prefix_idx = seg_start_idx + i
            if prefix_idx in prefix_to_corrected:
                c_idx = int(prefix_to_corrected[prefix_idx][0])
                dist = word_distance(word, corrected_words[c_idx]) if c_idx < len(corrected_words) else 1.0
                word_info.append((prefix_idx, word, c_idx, dist))
            else:
                word_info.append((prefix_idx, word, None, 1.0))

        # Case A: all words map to the same corrected index
        corrected_indices = set(w[2] for w in word_info if w[2] is not None)
        if len(corrected_indices) == 1:
            c_idx = corrected_indices.pop()
            anchored, anchor_p_idx = is_anchored_externally(c_idx, seg_start_idx, seg_end_idx)
            if anchored:
                # Corrected word is well-matched elsewhere — insert instead of replace
                mode = 'insert_before' if anchor_p_idx > seg_end_idx else 'insert_after'
                replacements.append({
                    'corrected_indices': [c_idx],
                    'prefix_words': [w[1] for w in word_info],
                    'mode': mode,
                })
                logger.debug(f"  Segment [{result.segment_id}] Case A: all map to corrected[{c_idx}] "
                            f"-> {mode} {len(word_info)} prefix words (anchored by prefix[{anchor_p_idx}])")
            else:
                replacements.append({
                    'corrected_indices': [c_idx],
                    'prefix_words': [w[1] for w in word_info],
                    'mode': 'replace',
                })
                logger.debug(f"  Segment [{result.segment_id}] Case A: all map to corrected[{c_idx}] "
                            f"-> replacing with {len(word_info)} prefix words")
            continue

        # Case B: find consecutive groups with dist > threshold mapping to same corrected index
        # A good-distance prefix word (dist <= threshold) or a different corrected index breaks the group
        groups = []
        current_group = []
        current_c_idx = None
        for info in word_info:
            p_idx, p_word, c_idx, dist = info
            if dist > high_dist_threshold and c_idx is not None and (current_c_idx is None or c_idx == current_c_idx):
                current_group.append(info)
                current_c_idx = c_idx
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = []
                current_c_idx = None
        if len(current_group) >= 2:
            groups.append(current_group)

        for group in groups:
            c_idx = group[0][2]
            # Include all prefix words mapping to the same corrected index, in original prefix order
            all_for_c_idx = [w[1] for w in word_info if w[2] == c_idx]
            anchored, anchor_p_idx = is_anchored_externally(c_idx, seg_start_idx, seg_end_idx)
            if anchored:
                mode = 'insert_before' if anchor_p_idx > seg_end_idx else 'insert_after'
                replacements.append({
                    'corrected_indices': [c_idx],
                    'prefix_words': all_for_c_idx,
                    'mode': mode,
                })
                logger.debug(f"  Segment [{result.segment_id}] Case B: corrected[{c_idx}] "
                            f"-> {mode} {len(all_for_c_idx)} prefix words (anchored by prefix[{anchor_p_idx}])")
            else:
                replacements.append({
                    'corrected_indices': [c_idx],
                    'prefix_words': all_for_c_idx,
                    'mode': 'replace',
                })
                logger.debug(f"  Segment [{result.segment_id}] Case B: corrected[{c_idx}] "
                            f"-> {len(all_for_c_idx)} prefix words (group of {len(group)} triggered)")

    # Sort by corrected index and merge overlaps (only merge same mode)
    replacements.sort(key=lambda r: min(r['corrected_indices']))
    merged = []
    for rep in replacements:
        if (merged
            and min(rep['corrected_indices']) <= max(merged[-1]['corrected_indices'])
            and rep.get('mode', 'replace') == merged[-1].get('mode', 'replace')):
            prev = merged[-1]
            prev['corrected_indices'] = sorted(set(prev['corrected_indices'] + rep['corrected_indices']))
            prev['prefix_words'].extend(rep['prefix_words'])
        else:
            merged.append(rep)

    if merged:
        logger.info(f"Identified {len(merged)} replacements for low-score segments before cutoff")
        for i, r in enumerate(merged):
            mode = r.get('mode', 'replace')
            logger.info(f"  {i+1}. [{mode}] Corrected indices {r['corrected_indices']} -> "
                       f"{len(r['prefix_words'])} prefix words: {' '.join(r['prefix_words'][:5])}{'...' if len(r['prefix_words']) > 5 else ''}")

    return merged


def create_truncated_file(corrected_path: str, corrected_words: list[str],
                          cutoff_index: int, replacements: list[dict] = None,
                          logger: logging.Logger = None):
    if logger is None:
        logger = logging.getLogger('dtw_match')

    with open(corrected_path, 'r', encoding='utf-8') as f:
        original_text = f.read()

    word_spans = build_word_char_spans(original_text)

    # Apply replacements in reverse order to preserve character positions
    if replacements:
        for rep in reversed(replacements):
            first_idx = min(rep['corrected_indices'])
            last_idx = max(rep['corrected_indices'])
            if first_idx < len(word_spans) and last_idx < len(word_spans):
                char_start = word_spans[first_idx][0]
                char_end = word_spans[last_idx][1]
                mode = rep.get('mode', 'replace')
                insert_text = ' '.join(rep['prefix_words'])
                if mode == 'insert_before':
                    original_text = original_text[:char_start] + insert_text + ' ' + original_text[char_start:]
                elif mode == 'insert_after':
                    original_text = original_text[:char_end] + ' ' + insert_text + original_text[char_end:]
                else:  # replace
                    original_text = original_text[:char_start] + insert_text + original_text[char_end:]

        # Rebuild spans after replacements
        word_spans = build_word_char_spans(original_text)

    # Compute adjusted cutoff accounting for word count changes
    adjusted_cutoff = cutoff_index
    if replacements:
        for rep in replacements:
            if min(rep['corrected_indices']) < cutoff_index:
                mode = rep.get('mode', 'replace')
                if mode in ('insert_before', 'insert_after'):
                    # Corrected word preserved, only adding prefix words
                    adjusted_cutoff += len(rep['prefix_words'])
                else:
                    # Replacing corrected words with prefix words
                    adjusted_cutoff += len(rep['prefix_words']) - len(rep['corrected_indices'])

    # Find character position for the adjusted cutoff
    if adjusted_cutoff < len(word_spans):
        cutoff_pos = word_spans[adjusted_cutoff][0]
    else:
        cutoff_pos = len(original_text)

    truncated_text = original_text[:cutoff_pos].rstrip()

    base, ext = corrected_path.rsplit('.', 1) if '.' in corrected_path else (corrected_path, 'txt')
    output_path = f"{base}.truncated.{ext}"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(truncated_text)

    logger.info(f"Created truncated file: {output_path}")
    logger.info(f"  Kept {cutoff_index} words (adjusted to {adjusted_cutoff} after replacements, char position {cutoff_pos})")
    if replacements:
        logger.info(f"  Applied {len(replacements)} replacements")
    return output_path


# =============================================================================
# PLOTTING
# =============================================================================

def plot_match_scores(results: list[AlignmentResult], corrected_len: int,
                      title="Banded DTW Match Scores", save_path=None,
                      drop_info: dict = None, drop_threshold: float = 0.25,
                      ma_window: int = 10):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    segment_ids = [r.segment_id for r in results]
    indices = list(range(len(results)))
    start_positions = [r.start_pos for r in results]
    end_positions = [r.end_pos for r in results]
    match_scores = [r.match_score for r in results]

    ax1 = axes[0]
    colors = plt.cm.RdYlGn(match_scores)

    for i, (seg_id, start, end, score) in enumerate(zip(segment_ids, start_positions, end_positions, match_scores)):
        ax1.barh(seg_id, end - start, left=start, height=0.8, color=colors[i], alpha=0.7)

    ax1.set_xlabel('Position in corrected file (word index)')
    ax1.set_ylabel('Segment ID')
    ax1.set_title('Segment Alignments (color = match score: red=low, green=high)')
    ax1.set_xlim(0, corrected_len)

    if drop_info and drop_info['first_drop_idx'] is not None:
        drop_idx = drop_info['first_drop_idx']
        if drop_idx < len(results):
            drop_pos = results[drop_idx].start_pos
            ax1.axvline(x=drop_pos, color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'First drop at pos {drop_pos}')
            ax1.legend()

    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label='Match Score')

    ax2 = axes[1]
    ax2.plot(indices, match_scores, 'b-', linewidth=1, alpha=0.5, label='Raw score')
    ax2.scatter(indices, match_scores, c=match_scores, cmap='RdYlGn', s=20, alpha=0.6)

    if drop_info and 'moving_avg' in drop_info:
        moving_avg = drop_info['moving_avg']
        ax2.plot(indices, moving_avg, 'purple', linewidth=2, alpha=0.9, label=f'Moving avg ({ma_window} segments)')

    ax2.axhline(y=drop_threshold, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'{drop_threshold} threshold')

    if drop_info and drop_info['first_drop_idx'] is not None:
        drop_idx = drop_info['first_drop_idx']
        ax2.axvline(x=drop_idx, color='red', linestyle='-', linewidth=2, alpha=0.8)
        ax2.annotate(f'First drop\n(segment {drop_idx})\nType: {drop_info["first_drop_type"]}',
                     xy=(drop_idx, drop_threshold), xytext=(drop_idx + 5, 0.5),
                     fontsize=10, color='red',
                     arrowprops=dict(arrowstyle='->', color='red'))

    ax2.set_xlabel('Segment Index')
    ax2.set_ylabel('Match Score')
    ax2.set_title(f'Match Score with Moving Average (window={ma_window})')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")
    plt.show()


def plot_cumulative_cost_landscape(alignment, dist_matrix, band_width,
                                    title="DTW Cumulative Cost Landscape", save_path=None):
    cost_matrix = alignment.costMatrix

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax1 = axes[0]
    im1 = ax1.imshow(dist_matrix, aspect='auto', origin='lower', cmap='terrain')
    ax1.plot(alignment.index2, alignment.index1, 'r-', linewidth=1.5, alpha=0.8, label='Optimal path')
    ax1.set_xlabel('Corrected file (word index)')
    ax1.set_ylabel('Pre-fix file (word index)')
    ax1.set_title('Local Cost Matrix (Word Distances)')
    plt.colorbar(im1, ax=ax1, label='Distance')
    ax1.legend(loc='upper left')

    ax2 = axes[1]
    im2 = ax2.imshow(cost_matrix, aspect='auto', origin='lower', cmap='terrain')

    n, m = cost_matrix.shape
    x = np.arange(m)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)

    vmin, vmax = cost_matrix.min(), cost_matrix.max()
    levels = np.linspace(vmin, vmax, 20)
    ax2.contour(X, Y, cost_matrix, levels=levels, colors='black', alpha=0.3, linewidths=0.5)

    ax2.plot(alignment.index2, alignment.index1, 'r-', linewidth=2, alpha=0.9, label='Optimal path')
    ax2.set_xlabel('Corrected file (word index)')
    ax2.set_ylabel('Pre-fix file (word index)')
    ax2.set_title('Cumulative Cost Landscape (Topographic)')
    plt.colorbar(im2, ax=ax2, label='Cumulative cost')
    ax2.legend(loc='upper left')

    # Draw band boundaries
    n_dist, m_dist = dist_matrix.shape
    i_vals = np.array([0, n_dist - 1])
    slope = m_dist / n_dist
    upper = slope * i_vals + band_width
    lower = slope * i_vals - band_width

    for ax in [ax1, ax2]:
        ax.plot(upper, i_vals, 'y--', linewidth=2, alpha=0.9, label='Band boundary')
        ax.plot(lower, i_vals, 'y--', linewidth=2, alpha=0.9)
        ax.set_xlim(0, m_dist - 1)
        ax.set_ylim(0, n_dist - 1)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")
    plt.show()


# =============================================================================
# OUTPUT
# =============================================================================

def print_results(results: list[AlignmentResult], logger: logging.Logger = None):
    if logger is None:
        logger = logging.getLogger('dtw_match')

    def output(msg=""):
        print(msg)
        logger.info(msg)

    output("\n" + "=" * 80)
    output("ALIGNMENT RESULTS")
    output("=" * 80)

    for result in results:
        output(f"\nSegment [{result.segment_id}] (pos {result.start_pos}-{result.end_pos})")
        output(f"  Original: {result.original_text[:60]}{'...' if len(result.original_text) > 60 else ''}")
        output(f"  Match score: {result.match_score:.1%}")

        if result.insertions:
            output(f"  Insertions (potential hallucinations): {result.insertions[:5]}{'...' if len(result.insertions) > 5 else ''}")
        if result.deletions:
            output(f"  Deletions (missing from corrected): {result.deletions[:5]}{'...' if len(result.deletions) > 5 else ''}")

    output("\n" + "=" * 80)
    output("SUMMARY")
    output("=" * 80)

    total_segments = len(results)
    if total_segments == 0:
        output("\nNo segments to analyze.")
        return

    avg_score = sum(r.match_score for r in results) / total_segments
    perfect_matches = [r for r in results if r.match_score == 1.0]
    high_matches = [r for r in results if r.match_score >= 0.8]
    low_match_segments = [r for r in results if r.match_score < 0.5]
    high_insertion_segments = [r for r in results if len(r.insertions) > 3]

    total_insertions = sum(len(r.insertions) for r in results)
    total_deletions = sum(len(r.deletions) for r in results)

    output(f"\nTotal segments: {total_segments}")
    output(f"Average match score: {avg_score:.1%}")
    output(f"Perfect matches (100%): {len(perfect_matches)} ({len(perfect_matches)/total_segments*100:.1f}%)")
    output(f"High matches (>=80%): {len(high_matches)} ({len(high_matches)/total_segments*100:.1f}%)")
    output(f"Low matches (<50%): {len(low_match_segments)} ({len(low_match_segments)/total_segments*100:.1f}%)")
    output(f"\nTotal insertions (potential hallucinations): {total_insertions}")
    output(f"Total deletions (missing from corrected): {total_deletions}")
    output(f"Segments with high insertions (>3): {len(high_insertion_segments)}")

    if high_insertion_segments:
        output("\nSegments with potential hallucinations (>3 insertions):")
        for r in high_insertion_segments[:10]:
            output(f"  [{r.segment_id}] - {len(r.insertions)} insertions: {r.insertions[:3]}...")

    if low_match_segments:
        output("\nSegments with low match scores (<50%):")
        for r in low_match_segments[:15]:
            output(f"  [{r.segment_id}] - {r.match_score:.1%} match - deletions: {r.deletions[:3]}{'...' if len(r.deletions) > 3 else ''}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Banded DTW Text Matching for Hebrew Transcription Alignment')
    parser.add_argument('--band-width', type=int, default=200,
                        help='Band width for Sakoe-Chiba constraint (default: 200)')
    parser.add_argument('--chunk-size', type=int, default=10,
                        help='Chunk size when using --swap (default: 10 words)')
    parser.add_argument('--prefix', default='154556.pre-fix.txt', help='Pre-fix file path')
    parser.add_argument('--corrected', default='154556.txt', help='Corrected file path')
    parser.add_argument('--swap', action='store_true', help='Swap prefix/corrected; parse prefix as chunks')
    parser.add_argument('--save-plot', metavar='PATH', help='Save plot to file instead of showing')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--log-file', type=str, default=None, help='Log file path (default: dtw.log)')
    step_pattern_choices = [
        'asymmetric', 'asymmetricP0', 'asymmetricP05', 'asymmetricP1', 'asymmetricP2',
        'symmetric1', 'symmetric2', 'symmetricP0', 'symmetricP05', 'symmetricP1', 'symmetricP2',
        'rigid', 'mori2006',
        'typeIa', 'typeIas', 'typeIb', 'typeIbs', 'typeIc', 'typeIcs', 'typeId', 'typeIds',
        'typeIIa', 'typeIIb', 'typeIIc', 'typeIId', 'typeIIIc', 'typeIVc',
    ]
    parser.add_argument('--step-pattern', type=str, default='asymmetric',
                        choices=step_pattern_choices,
                        help='DTW step pattern (default: asymmetric)')
    parser.add_argument('--match-threshold', type=float, default=0.5,
                        help='Word-level distance threshold for a good match (default: 0.5)')
    parser.add_argument('--high-dist-threshold', type=float, default=0.7,
                        help='Distance threshold for grouping bad-match words in replacements (default: 0.7)')
    parser.add_argument('--low-score-threshold', type=float, default=0.5,
                        help='Segment score threshold for triggering replacements (default: 0.5)')
    parser.add_argument('--jump-threshold', type=int, default=40,
                        help='Minimum vertical jump size for cutoff detection (default: 40)')
    parser.add_argument('--drop-threshold', type=float, default=0.25,
                        help='Moving average threshold for quality drop detection (default: 0.25)')
    parser.add_argument('--ma-window', type=int, default=10,
                        help='Moving average window size for drop detection (default: 10)')
    args = parser.parse_args()

    if args.swap:
        args.prefix, args.corrected = args.corrected, args.prefix

    logger = setup_logging(args.log_file)

    print("=" * 80)
    print(f"Banded DTW Text Matching (band_width={args.band_width})")
    print("=" * 80)
    logger.info(f"Mode: banded, band_width={args.band_width}")

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
        corrected_words = parse_prefix_file_as_words(args.corrected)
        print(f"  Found {len(corrected_words)} words (extracted from segments)")
        logger.info(f"Parsed {args.corrected} as segment text: {len(corrected_words)} words")
    else:
        corrected_words = parse_corrected_file(args.corrected)
        print(f"  Found {len(corrected_words)} words")
        logger.info(f"Parsed {args.corrected}: {len(corrected_words)} words")

    all_prefix_words = []
    for seg in segments:
        all_prefix_words.extend(seg.words)
    print(f"  Total pre-fix words: {len(all_prefix_words)}")

    # Run banded DTW
    print(f"\nRunning banded DTW alignment...")
    alignment_path, alignment_obj, dist_matrix = banded_dtw_alignment(
        all_prefix_words, corrected_words, band_width=args.band_width,
        step_pattern=args.step_pattern
    )
    print(f"  Alignment path length: {len(alignment_path)}")

    # Detect vertical jumps
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

    # Map segments
    print("\nMapping segments from alignment...")
    results = map_segments_from_global(segments, all_prefix_words, corrected_words, alignment_path,
                                       match_threshold=args.match_threshold)
    print_results(results, logger=logger)

    # Drop detection (always run)
    print("\nAnalyzing alignment quality drop...")
    drop_info = detect_drop(results, threshold=args.drop_threshold, word_count_threshold=15, ma_window=args.ma_window, logger=logger)

    if drop_info['first_drop_idx'] is not None:
        print(f"\n*** FIRST DROP DETECTED ***")
        print(f"  At segment index: {drop_info['first_drop_idx']}")
        print(f"  Detection type: {drop_info['first_drop_type']}")
        r = results[drop_info['first_drop_idx']]
        print(f"  Segment [{r.segment_id}]: score={r.match_score:.1%}")
        print(f"  Position in corrected file: {r.start_pos}")
    else:
        print(f"\nNo significant drop detected (threshold={args.drop_threshold})")

    # Truncated file creation
    cutoff_index = find_cutoff_index(vertical_jumps, drop_info, results,
                                     jump_threshold=args.jump_threshold, logger=logger)
    if cutoff_index is not None:
        replacements = identify_replacements(
            results, segments, all_prefix_words, corrected_words,
            alignment_path, cutoff_index,
            low_score_threshold=args.low_score_threshold,
            high_dist_threshold=args.high_dist_threshold,
            match_threshold=args.match_threshold,
            logger=logger
        )
        output_path = create_truncated_file(
            args.corrected, corrected_words, cutoff_index,
            replacements=replacements, logger=logger
        )
        print(f"\n*** TRUNCATED FILE CREATED ***")
        print(f"  Output: {output_path}")
        print(f"  Kept {cutoff_index} of {len(corrected_words)} words")
    else:
        print("\nNo truncation needed (no large jumps or drops detected)")

    # Plots
    if not args.no_plot:
        print("\nGenerating match score map...")
        save_path_scores = args.save_plot.replace('.png', '_scores.png') if args.save_plot else None
        plot_match_scores(
            results,
            len(corrected_words),
            title=f"Banded DTW Match Scores (width={args.band_width}): {args.prefix} → {args.corrected}",
            save_path=save_path_scores,
            drop_info=drop_info,
            drop_threshold=args.drop_threshold,
            ma_window=args.ma_window
        )

        print("\nGenerating cumulative cost landscape plot...")
        plot_cumulative_cost_landscape(
            alignment_obj,
            dist_matrix,
            band_width=args.band_width,
            title=f"Banded DTW Alignment (width={args.band_width}): {args.prefix} → {args.corrected}",
            save_path=args.save_plot
        )


if __name__ == "__main__":
    main()
