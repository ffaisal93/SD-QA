#!/usr/bin/env python3
"""
Simple script to extract (example_id, question_text, answers) triples
from an SD-QA / TyDi-style JSONL file.

Usage:
  python3 extract_qa_from_jsonl.py /path/to/file.jsonl > output.tsv

Output format (tab-separated):
  example_id<TAB>question_text<TAB>answer_1 ||| answer_2 ||| ...
"""

import json
import sys
from pathlib import Path
from typing import List, Set


def utf8_slice(text: str, start: int, end: int) -> str:
    """Slice text using UTF-8 byte offsets [start, end)."""
    if start < 0 or end < 0:
        return ""
    b = text.encode("utf-8")
    # Guard against out-of-range indexes
    start = max(0, min(start, len(b)))
    end = max(0, min(end, len(b)))
    if end <= start:
        return ""
    return b[start:end].decode("utf-8", errors="replace").strip()


def sanitize_field(s: str) -> str:
    """Make a string safe for TSV: remove tabs/newlines and collapse spaces."""
    if not s:
        return ""
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    # Collapse multiple spaces
    return " ".join(s.split())


def extract_answers(example: dict) -> List[str]:
    """
    Extract a set of gold answer strings for a single example.

    Priority:
      1) Use minimal_answer byte offsets if present.
      2) Otherwise, use passage_answer.candidate_index to look up
         passage_answer_candidates and slice by their byte offsets.
    """
    doc = example.get("document_plaintext", "")
    annotations = example.get("annotations", [])
    candidates = example.get("passage_answer_candidates", [])

    answers: Set[str] = set()

    # Helper: get candidate span by index
    def candidate_span(idx: int) -> str:
        if idx is None or idx < 0 or idx >= len(candidates):
            return ""
        cand = candidates[idx]
        start = cand.get("plaintext_start_byte", -1)
        end = cand.get("plaintext_end_byte", -1)
        return utf8_slice(doc, start, end)

    for ann in annotations:
        # 1) Minimal answer span if available
        minimal = ann.get("minimal_answer", {}) or {}
        ms = minimal.get("plaintext_start_byte", -1)
        me = minimal.get("plaintext_end_byte", -1)
        if ms is not None and me is not None and ms >= 0 and me >= 0:
            ans = utf8_slice(doc, ms, me)
            if ans:
                answers.add(ans)

        # 2) Passage-level candidate index
        passage = ann.get("passage_answer", {}) or {}
        idx = passage.get("candidate_index", -1)
        if idx is not None and idx >= 0:
            ans = candidate_span(idx)
            if ans:
                answers.add(ans)

    return sorted(a for a in answers if a)


def process_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            example_id = ex.get("example_id")
            question_text = ex.get("question_text", "")
            if example_id is None:
                continue

            answers = [sanitize_field(a) for a in extract_answers(ex)]
            answers_str = " ||| ".join(a for a in answers if a)
            question_text_sanitized = sanitize_field(question_text)
            # Output: example_id<TAB>question<TAB>answers_joined
            print(f"{example_id}\t{question_text_sanitized}\t{answers_str}")


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(
            "Usage: python3 extract_qa_from_jsonl.py /path/to/file.jsonl",
            file=sys.stderr,
        )
        return 1

    jsonl_path = Path(argv[1])
    if not jsonl_path.exists():
        print(f"Error: file not found: {jsonl_path}", file=sys.stderr)
        return 1

    process_file(jsonl_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))


