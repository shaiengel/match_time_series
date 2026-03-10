# DTW Step Patterns

This document describes the available step patterns for the `--step-pattern` argument in `banded_dtw.py`.

## Chosen Pattern: `asymmetric`

We use `asymmetric` as the default because:
- The prefix (query) and reference (corrected) files have **different lengths** (~500 word difference)
- Asymmetric allows the reference to advance without consuming query words, handling length mismatches naturally
- Combined with slanted band constraint (`--band-width`), it limits drift while allowing flexible alignment
- Skipped reference words (words the LLM added/hallucinated) don't distort the alignment — they're simply not matched
- Vertical jump detection works cleanly: jumps represent real content gaps, not forced many-to-one artifacts

Trade-off: reference words that no prefix word maps to are invisible in the alignment. This is handled separately by the replacement logic (see `replacements.md`).

## Asymmetric Patterns

| Pattern | Slope constraint | Description |
|---|---|---|
| `asymmetric` | None | Basic asymmetric. Reference can advance freely without consuming query words. No penalty for skipping. **Default choice.** |
| `asymmetricP0` | P=0 | Rabiner-style step structure with no slope penalty. |
| `asymmetricP05` | P=0.5 | Moderate slope penalty — discourages large deviations from diagonal. |
| `asymmetricP1` | P=1 | Stronger slope penalty — path encouraged to stay near diagonal. |
| `asymmetricP2` | P=2 | Strongest slope penalty — heavily penalizes deviation from diagonal, similar to banding but via cost. |

Higher P values force the alignment closer to the diagonal (1:1 mapping). With slanted band already constraining drift, the P variants add redundant control. Use if you want finer cost-based diagonal enforcement on top of banding.

## Symmetric Patterns

Symmetric patterns weight both directions equally. Skipping in either direction incurs cost. No words are "free to skip."

| Pattern | Steps allowed | Description |
|---|---|---|
| `symmetric1` | (1,1) only | Strictly diagonal — one-to-one mapping. No warping. Sequences must be similar length. |
| `symmetric2` | (1,1), (1,0), (0,1) | Classic DTW. Allows horizontal/vertical steps with cost. Every word participates. Most commonly used symmetric pattern. |
| `symmetricP0` | P=0 | Multi-step transitions with no slope penalty. |
| `symmetricP05` | P=0.5 | Moderate slope penalty. |
| `symmetricP1` | P=1 | Stronger slope penalty. |
| `symmetricP2` | P=2 | Heaviest slope penalty — path stays close to diagonal. |

For transcription alignment with different-length files, symmetric patterns force many-to-one mappings to "use up" extra words. This creates noise in vertical jump detection — jumps would include forced mappings, not just real content gaps.

## Special Patterns

| Pattern | Description |
|---|---|
| `rigid` | Only (1,1) diagonal steps. Equivalent to `symmetric1`. No warping — pure element-wise comparison. Sequences must be the same length. |
| `mori2006` | From Mori et al. 2006. Specialized asymmetric pattern designed for speech recognition with varying tempo. |

## Rabiner & Juang Classification (Type I-IV)

These patterns come from the Rabiner & Juang speech recognition classification. They define allowed slope ranges for the alignment path.

| Family | Slope range | Description |
|---|---|---|
| Type I | 0 to infinity | Most flexible — allows horizontal and vertical steps freely. |
| Type II | 0.5 to 2 | Moderate — path slope constrained between 0.5 and 2. |
| Type III | 0.67 to 1.5 | Tighter — path stays closer to diagonal. |
| Type IV | 1 to 1 | Strictest — nearly rigid diagonal movement. |

### Sub-variants

- **Letter suffix** (`a`, `b`, `c`, `d`) — different step sizes and weight distributions within each family.
- **`s` suffix** (e.g., `typeIas`, `typeIbs`) — "smoothed" variant with interpolation for better path continuity.

### Full list

| Pattern | Notes |
|---|---|
| `typeIa`, `typeIas` | Type I, variant a (smoothed) |
| `typeIb`, `typeIbs` | Type I, variant b (smoothed) |
| `typeIc`, `typeIcs` | Type I, variant c (smoothed) |
| `typeId`, `typeIds` | Type I, variant d (smoothed) |
| `typeIIa` | Type II, variant a |
| `typeIIb` | Type II, variant b |
| `typeIIc` | Type II, variant c |
| `typeIId` | Type II, variant d |
| `typeIIIc` | Type III, variant c |
| `typeIVc` | Type IV, variant c |

These are mostly academic/legacy patterns from speech recognition literature. For Hebrew transcription alignment they are unlikely to offer advantages over `asymmetric` with banding.
