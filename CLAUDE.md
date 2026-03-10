# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**match-time-series** — DTW (Dynamic Time Warping) text alignment tool for comparing Hebrew transcription files. Aligns segments from a pre-fix transcription file to regions in an LLM-corrected file to detect hallucinations. Uses Python 3.12+ and is managed with [uv](https://docs.astral.sh/uv/).

## Commands

```bash
# Install dependencies
uv sync

# Run the application
uv run python main.py --mode <mode>

# Add a dependency
uv add <package>
```

## DTW Modes

| Mode | Description | Use case |
|------|-------------|----------|
| `global` | Aligns all words at once using full DTW | Best for same content with minor corrections |
| `banded` | Global DTW with Sakoe-Chiba band constraint | Limits drift, good for hallucination detection |
| `subsequence` | Per-segment DTW with open_begin/open_end | Searching segments in potentially different reference |
| `chunked` | Fixed-size chunks (10 words) matched within search window (10 words) | Strict local matching for hallucination detection |

## File Structure

- `154556.pre-fix.txt` — Original transcription with segments: `[N] timestamp: text` (query)
- `154556.txt` — LLM-corrected version (reference)
- `dtw.log` — Log file (overwritten each run)

## Key Arguments

```bash
--mode {global,banded,subsequence,chunked}
--prefix <file>          # Pre-fix file (default: 154556.pre-fix.txt)
--corrected <file>       # Corrected file (default: 154556.txt)
--swap                   # Swap prefix/corrected files; parses prefix as 10-word chunks
--band-width N           # Band width for banded mode (recommended: 200)
--chunk-size N           # Words per chunk for chunked mode (default: 10)
--search-window N        # Search window for chunked/subsequence (default: 10)
--save-plot <file.png>   # Save plot to file
--no-plot                # Disable plotting
--log-file <file>        # Log file (default: dtw.log)
--step-pattern <pattern> # DTW step pattern (default: asymmetric)
--match-threshold F      # Word-level distance threshold for a good match (default: 0.5)
--high-dist-threshold F  # Distance threshold for grouping bad-match words in replacements (default: 0.7)
--low-score-threshold F  # Segment score threshold for triggering replacements (default: 0.5)
--jump-threshold N       # Minimum vertical jump size for cutoff detection (default: 40)
--drop-threshold F       # Moving average threshold for quality drop detection (default: 0.25)
--ma-window N            # Moving average window size for drop detection (default: 10)
```

## Banded Mode Details

- Uses `asymmetric` step pattern by default (configurable via `--step-pattern`)
- Recommended `--band-width 200` for transcription files with ~500 word differences
- Generates two plots:
  - `*_scores.png` — Match score map (segment positions + score progression)
  - `*.png` — Cumulative cost landscape with band boundaries (yellow dashed lines)

### Vertical Jump Detection

Banded mode automatically detects **vertical jumps** in the alignment path — regions where the prefix file advances while the corrected file stays at the same position. This indicates content removed or condensed by the LLM.

Output example:
```
*** VERTICAL JUMPS DETECTED (46 total) ***
  1. Corrected index 1661: prefix words 1721-1824 (103 words skipped)
  2. Corrected index 1660: prefix words 1663-1720 (57 words skipped)
```

- Top 10 jumps printed to console
- All jumps (≥5 words) logged to `dtw.log`
- Consecutive jumps at adjacent indices often represent a single large deletion

### Automatic Truncation & Fixing

Banded mode automatically creates a truncated reference file (`*.truncated.txt`) by:

1. **Finding the cutoff point** — takes the minimum of:
   - First vertical jump > `--jump-threshold` words (default: 40, by `corrected_idx`)
   - First moving-average drop below `--drop-threshold` (default: 0.25, segment's `start_pos`, window size: `--ma-window`)

2. **Fixing low-score segments** before the cutoff — for segments with match score < `--low-score-threshold` (default: 0.5):
   - **Case A:** All prefix words map to the same corrected word → replace that word with all prefix words (reference has missing content)
   - **Case B:** Consecutive groups (≥2) of prefix words with distance > `--high-dist-threshold` (default: 0.7) mapping to the same corrected word → replace the corresponding corrected words with the prefix words. A prefix word with good distance (≤ threshold) breaks the group. All prefix words mapping to the targeted corrected word are included in the replacement (in original prefix order), not just the group members.
   - **External anchoring:** Before replacing, checks if the corrected word is well-matched (dist ≤ 0.5) by a prefix word in an adjacent segment. If anchored, inserts the prefix words before/after the corrected word (based on speech order) instead of replacing it.

   See `replacements.md` for detailed logic documentation.

3. **Writing the file** — truncates the original reference file at the cutoff position, preserving original punctuation and formatting. Replacements are applied at character positions in the original text.

Key functions in `banded_dtw.py`:
- `find_cutoff_index()` — determines where to truncate
- `identify_replacements()` — finds low-score segments to fix
- `build_word_char_spans()` — maps word indices to character positions in original text
- `create_truncated_file()` — applies replacements and truncates

## Architecture

Entry point is `main.py`. Dependencies: `dtw-python`, `numpy`, `matplotlib`, `pandas`, `openpyxl`.

## Debug Logging

The log file (`dtw.log`) shows per-segment data:
```
Segment [17]: 85% (7 words) pos 120-126
  Original: <text>
  Matched:  <text>
```

For low-scoring segments (<50%), word-level alignment details are logged.

## Stable Whisper Probability Analysis

`probability_analysis.py` — Analyzes JSON output from stable_whisper. Extracts word-level data and detects transcription quality degradation.

```bash
uv run python probability_analysis.py
```

### JSON Input Structure (stable_whisper output)
```json
{
  "segments": [
    {
      "words": [
        {"word": "...", "start": 2.85, "end": 3.2, "probability": 0.85, "tokens": [...]}
      ]
    }
  ]
}
```

### Output Excel Structure

#### Sheet 1: "Words"
| Column | Description |
|--------|-------------|
| word | Transcribed word (Hebrew) |
| start_time | Start time in seconds |
| end_time | End time in seconds |
| probability | Raw probability from Whisper |
| moving_avg | 100-point rolling average of probability |
| cusum | Cumulative sum of deviations from baseline |
| gradient | Rate of change of moving average |

#### Sheet 2: "Detection Results"
| Column | Description |
|--------|-------------|
| Method | Detection algorithm name |
| Index | Detected degradation index (-1 = not detected) |
| Description | What the method detects |

### Degradation Detection Methods

1. **Gradient** (steepest drop) — Finds single point with most negative gradient. May detect temporary dips.
2. **Sustained** (recommended) — Finds where moving_avg stays below 0.25 for 200 consecutive points. Most reliable for confirming permanent degradation.
3. **Rolling Average** (50% baseline) — Detects when moving_avg drops below 50% of baseline (indices 100-300).
4. **CUSUM** (-50) — Cumulative sum of deviations from baseline. Triggers when cumsum < -50.

**Note:** Methods return -1 when no degradation is detected.
