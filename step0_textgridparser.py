"""
textgrid_parser.py
====================
Minimal Praat TextGrid parser — extracts (start, end, speaker, text)
utterances from a .TextGrid file. No external dependencies.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Utterance:
    speaker: str
    start: float
    end: float
    text: str


def clean_text(text: str) -> str:
    """Strip Praat annotation tags like <UNSURE>...</UNSURE> and <UNIN/>."""
    text = re.sub(r"<UNSURE>(.*?)</UNSURE>", r"\1", text)
    text = re.sub(r"<UNIN\s*/?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_textgrid(path: str) -> List[Utterance]:
    """
    Parses a Praat TextGrid file and returns a flat list of Utterance
    objects across all tiers (speakers), sorted by start time.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    utterances = []

    # Split into tier blocks
    tier_blocks = re.split(r'item \[\d+\]:', content)[1:]

    for block in tier_blocks:
        name_match = re.search(r'name\s*=\s*"([^"]*)"', block)
        speaker = name_match.group(1) if name_match else "UNKNOWN"

        # Find all intervals within this tier block
        interval_pattern = re.compile(
            r'xmin\s*=\s*([\d.]+)\s*'
            r'xmax\s*=\s*([\d.]+)\s*'
            r'text\s*=\s*"([^"]*)"',
            re.MULTILINE
        )

        for match in interval_pattern.finditer(block):
            xmin, xmax, text = match.groups()
            text = clean_text(text)
            if text:  # skip empty intervals (silence gaps)
                utterances.append(Utterance(
                    speaker=speaker,
                    start=float(xmin),
                    end=float(xmax),
                    text=text,
                ))

    utterances.sort(key=lambda u: u.start)
    return utterances


if __name__ == "__main__":
    import sys
    utts = parse_textgrid(sys.argv[1])
    print(f"Parsed {len(utts)} utterances")
    for u in utts[:5]:
        print(f"[{u.start:.2f}-{u.end:.2f}] {u.speaker}: {u.text[:80]}")
