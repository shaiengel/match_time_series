# DTW Window Types

## The `--window-type` Argument

Controls the shape of the band constraint applied during DTW alignment. Available choices:

| Type | Uses `--band-width`? | Description |
|------|---------------------|-------------|
| `slantedband` (default) | Yes | Band around the `(0,0) → (n,m)` diagonal. Adjusts for length difference between sequences. |
| `sakoechiba` | Yes | Band around the `i = j` diagonal. Does **not** adjust for length difference. |
| `itakura` | No (ignored) | Itakura parallelogram constraint. |
| `none` | No (ignored) | No window constraint at all. |

## How They Differ

### `slantedband`

The band follows the diagonal from corner `(0,0)` to corner `(n,m)` of the cost matrix. This means it accounts for the overall length ratio between prefix and corrected files.

**Assumption:** The extra words in the longer sequence are **uniformly distributed** across the text.

**Problem:** If extra words cluster in one region (e.g., the LLM condensed mostly the second half), the band center drifts away from the true alignment early on. The correct match position can fall outside the band even with a generous width.

**Example (file 153145, band-width=200):**
- Prefix: 2752 words, Corrected: 2124 words (628 extra in prefix)
- At prefix index 938, the slanted diagonal center is at corrected index **724**
- Band extends to 724 + 200 = **924**
- Actual correct match is at corrected **924** — exactly at the edge
- By prefix index 960, correct match (~946) is **outside the band** — alignment breaks

### `sakoechiba`

The band follows the simple `i = j` diagonal. The allowed region is `|i - j| <= band_width`.

**Advantage:** Does not assume uniform distribution of extra words. Works better when the two files are closely aligned for the first portion and diverge later.

**Tradeoff:** The effective band is wider when sequences differ in length, so it is less restrictive overall. This may slightly delay detection of true hallucinations at the tail end.

**Same example with sakoechiba (band-width=200):**
- At prefix index 938, band allows corrected indices **738 to 1138**
- Correct match at ~924 is well within range
- Alignment stays correct through the transition

## When to Use Which

| Scenario | Recommended |
|----------|-------------|
| Files are similar length, edits are spread evenly | `slantedband` |
| Prefix is much longer, extra words cluster in certain sections | `sakoechiba` |
| Unsure / alignment breaks mid-file with `slantedband` | Try `sakoechiba` |
| Debugging / want unconstrained alignment | `none` |

## Usage

```bash
# Default (slantedband)
python banded_dtw.py --band-width 200 --prefix input/file.pre-fix.txt --corrected input/file.txt

# Switch to sakoechiba
python banded_dtw.py --window-type sakoechiba --band-width 200 --prefix input/file.pre-fix.txt --corrected input/file.txt

# No constraint
python banded_dtw.py --window-type none --prefix input/file.pre-fix.txt --corrected input/file.txt
```
