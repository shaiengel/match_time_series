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
```

## Banded Mode Details

- Uses `asymmetric` step pattern with slanted band constraint
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

## Architecture

Entry point is `main.py`. Dependencies: `dtw-python`, `numpy`, `matplotlib`.

## Debug Logging

The log file (`dtw.log`) shows per-segment data:
```
Segment [17]: 85% (7 words) pos 120-126
  Original: <text>
  Matched:  <text>
```

For low-scoring segments (<50%), word-level alignment details are logged.
