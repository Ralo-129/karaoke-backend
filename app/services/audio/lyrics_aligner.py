from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# Aligns user-provided reference lyrics (the correct words) to a transcription's
# word timestamps, producing an LRC where the WORDS come from the reference and
# the TIMES come from the transcription. This gives perfect words with reasonable
# timing, even when the speech-to-text mishears the song.


def _normalize(word: str) -> str:
    """Lowercase + strip accents + keep alphanumerics, for matching only."""
    decomposed = unicodedata.normalize("NFKD", word)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", stripped.lower())


def _fmt(t: float) -> str:
    if t < 0:
        t = 0.0
    m = int(t // 60)
    s = int(t % 60)
    c = int((t % 1) * 100)
    return f"{m:02d}:{s:02d}.{c:02d}"


def align_to_lrc(reference: str, words: list[dict]) -> str:
    """Builds an LRC string from reference lyrics + transcription word timings.

    - ``reference``: plain-text lyrics (lines separated by newlines).
    - ``words``: ordered ``[{"start": float, "word": str}, ...]`` from the
      transcription.

    Returns an LRC string (``[mm:ss.cc] <mm:ss.cc> palabra ...``) or ``""`` if it
    can't produce anything (caller should fall back to plain transcription).
    """
    # 1) Reference tokens, keeping their line so we can rebuild line breaks.
    ref_tokens: list[dict] = []
    for line_idx, line in enumerate(reference.splitlines()):
        for raw in line.split():
            norm = _normalize(raw)
            if norm:
                ref_tokens.append({"line": line_idx, "word": raw, "norm": norm, "time": None})

    # 2) Transcription tokens (drop ones that normalize to nothing).
    hyp_norm: list[str] = []
    hyp_time: list[float] = []
    for w in words:
        norm = _normalize(str(w.get("word", "")))
        if norm:
            hyp_norm.append(norm)
            hyp_time.append(float(w.get("start") or 0.0))

    if not ref_tokens or not hyp_time:
        return ""

    ref_norm = [t["norm"] for t in ref_tokens]

    # 3) Use only confident matches ("equal") as time anchors. Words in mismatched
    # regions (replace/delete) are left without a time and spread evenly between
    # the surrounding anchors below — this avoids piling many words on the same
    # timestamp when the transcription missed or misheard a section.
    matcher = SequenceMatcher(None, ref_norm, hyp_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ref_tokens[i1 + k]["time"] = hyp_time[j1 + k]

    _fill_missing_times(ref_tokens)

    # 4) Rebuild LRC grouped by reference line.
    lines: list[str] = []
    current_line = None
    current_words: list[dict] = []

    def flush():
        if not current_words:
            return
        start = current_words[0]["time"]
        body = "".join(f"<{_fmt(t['time'])}> {t['word']} " for t in current_words).strip()
        lines.append(f"[{_fmt(start)}] {body}")

    for tok in ref_tokens:
        if tok["line"] != current_line:
            flush()
            current_line = tok["line"]
            current_words = []
        current_words.append(tok)
    flush()

    return "\n".join(lines)


# Minimum spacing between consecutive words so no two ever share a timestamp
# (which would make whole lines pile up on the same time tag).
_MIN_GAP = 0.08


def _fill_missing_times(ref_tokens: list[dict]) -> None:
    """Fill ``None`` times by spreading words evenly between anchors, then enforce
    strictly increasing timestamps so lines never pile on the same time."""
    n = len(ref_tokens)
    known = [i for i, t in enumerate(ref_tokens) if t["time"] is not None]

    if not known:
        # No anchors at all: spread evenly (last resort).
        for i, t in enumerate(ref_tokens):
            t["time"] = float(i) * 0.4
        return

    # Leading None -> spread from 0 up to the first anchor (keep the anchor put).
    first = known[0]
    if first > 0:
        t1 = ref_tokens[first]["time"]
        for i in range(first):
            ref_tokens[i]["time"] = t1 * (i + 1) / (first + 1)

    # Gaps between anchors -> spread words evenly across the time span.
    for a, b in zip(known, known[1:]):
        if b - a > 1:
            t0 = ref_tokens[a]["time"]
            t1 = ref_tokens[b]["time"]
            step = (t1 - t0) / (b - a)
            for k in range(a + 1, b):
                ref_tokens[k]["time"] = t0 + step * (k - a)

    # Trailing None -> keep increasing slightly after the last anchor.
    last = known[-1]
    for i in range(last + 1, n):
        ref_tokens[i]["time"] = ref_tokens[i - 1]["time"] + 0.4

    # Enforce STRICTLY increasing times. This is what prevents several lines from
    # sharing one timestamp when the transcription lost sync in a section.
    for i in range(1, n):
        if ref_tokens[i]["time"] <= ref_tokens[i - 1]["time"]:
            ref_tokens[i]["time"] = ref_tokens[i - 1]["time"] + _MIN_GAP
