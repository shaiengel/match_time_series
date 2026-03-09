# Replacement Logic

This document describes how `identify_replacements()` in `banded_dtw.py` decides which corrected words to replace or augment with prefix words.

## Overview

After banded DTW alignment, each prefix word maps to one or more corrected words. For low-score segments (< 50% match), the code identifies corrected words that should be replaced with the original prefix words to restore content the LLM incorrectly changed.

## Grouping Rules

A **group** is a set of consecutive prefix words within a segment that:
1. All have **high distance** (> 0.7) to their mapped corrected word
2. All map to the **same corrected index**
3. Contains at least **2 words**

A prefix word **breaks the group** if:
- Its distance is <= 0.7 (good match), OR
- It maps to a different corrected index, OR
- It has no mapping at all

## Case A: All Words Map to Same Corrected Index

When every prefix word in a low-score segment maps to the same single corrected word.

Example:
```
[1729] 'abc' -> [1849] 'xyz' dist=1.0
[1730] 'def' -> [1849] 'xyz' dist=1.0
[1731] 'xyz' -> [1849] 'xyz' dist=0.0
```

All three map to corrected[1849]. The replacement includes **all** prefix words in their original order: `abc def xyz`.

## Case B: Consecutive High-Distance Groups

When only some prefix words in the segment form a group.

Example:
```
[1729] 'abc' -> [1849] 'xyz' dist=1.0   <-- group start
[1730] 'def' -> [1849] 'xyz' dist=1.0   <-- group end (dist=0.0 breaks it)
[1731] 'xyz' -> [1849] 'xyz' dist=0.0   <-- not in group (good match)
[1732] 'ghi' -> [1850] 'ghi' dist=0.0   <-- different corrected index
```

The group `[abc, def]` triggers a replacement of corrected[1849]. The replacement includes **all prefix words mapping to the same corrected index** (not just the group members), in original prefix order: `abc def xyz`.

This ensures that a good-match prefix word (like `xyz` with dist=0.0) is not lost when the corrected word it maps to gets replaced by the group.

## External Anchoring Check

Before replacing a corrected word, the code checks if that word is **anchored externally** — meaning a prefix word in a **different segment** maps to it with good distance (<= 0.5).

If anchored, the corrected word is **preserved** and the prefix words are **inserted** alongside it instead of replacing it.

### Insert Position (Speech Order)

The insert position depends on where the anchoring prefix word is relative to the current segment, since prefix order = speech order:

- Anchoring word is in a **later** segment (higher prefix index) -> `insert_before`: group words are inserted before the corrected word
- Anchoring word is in an **earlier** segment (lower prefix index) -> `insert_after`: group words are inserted after the corrected word

Example — anchored by later segment:
```
Segment [183]: [כך, צריך, לגרוס] all map to corrected 'קתני' with dist=1.0
Segment [184]: 'קתני' maps to corrected 'קתני' with dist=0.0
```
Result: `כך צריך לגרוס קתני` (insert_before, group words come first in speech order)

Example — anchored by earlier segment:
```
Segment [182]: 'קתני' maps to corrected 'קתני' with dist=0.0
Segment [183]: [כך, צריך, לגרוס] all map to corrected 'קתני' with dist=1.0
```
Result: `קתני כך צריך לגרוס` (insert_after, corrected word stays first)

### Not Anchored

If no external prefix word matches the corrected word well, the corrected word is **replaced** entirely with the prefix words.

## Replacement Modes Summary

| Mode | Corrected word | Prefix words | When |
|------|---------------|-------------|------|
| `replace` | Removed | Substituted in its place | No external anchor |
| `insert_before` | Preserved | Inserted before it | Anchored by later segment |
| `insert_after` | Preserved | Inserted after it | Anchored by earlier segment |

## Merging

Replacements targeting overlapping corrected indices are merged, but only if they share the same mode. This prevents a `replace` and an `insert_before` on the same word from being combined incorrectly.

## Cutoff Adjustment

After replacements are applied, the truncation cutoff is adjusted:
- `replace`: cutoff shifts by `+len(prefix_words) - len(corrected_indices)` (net word count change)
- `insert_before` / `insert_after`: cutoff shifts by `+len(prefix_words)` (corrected word preserved, only adding)
