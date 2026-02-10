#!/usr/bin/env python3
"""
Build (audio, question, answer) triples from SD-QA / TyDi-style JSONL files.

Two usage modes:

1) Single file:
   python3 sdqa_triples_from_jsonl.py /path/to/file.jsonl
   -> writes /path/to/file.triples.jsonl next to the input.

2) Repository root (recommended):
   cd /Users/faisal/Projects/SD-QA
   python3 sdqa_triples_from_jsonl.py .

   This will:
     - Walk all subfolders of ./test and ./dev
     - For every *.jsonl or *.jsonl.gz file, create a sibling
       *.triples.jsonl file, e.g.:
         english.test.gbr-en-GB.triples.jsonl
         english.test.gbr-en-US.triples.jsonl
"""

import json
import sys
import gzip
from pathlib import Path
from typing import List, Set, Optional


def utf8_slice(text: str, start: int, end: int) -> str:
    """Slice text using UTF-8 byte offsets [start, end)."""
    if start is None or end is None or start < 0 or end < 0:
        return ""
    b = text.encode("utf-8")
    start = max(0, min(start, len(b)))
    end = max(0, min(end, len(b)))
    if end <= start:
        return ""
    return b[start:end].decode("utf-8", errors="replace").strip()


def extract_answers(example: dict) -> List[str]:
    """
    Extract a set of gold answer strings for a single example.

    Priority:
      1) Use minimal_answer byte offsets if present.
      2) Also include passage_answer.candidate_index spans.
    """
    doc = example.get("document_plaintext", "")
    annotations = example.get("annotations", [])
    candidates = example.get("passage_answer_candidates", [])

    answers: Set[str] = set()

    def candidate_span(idx: int) -> str:
        if idx is None or idx < 0 or idx >= len(candidates):
            return ""
        cand = candidates[idx]
        start = cand.get("plaintext_start_byte", -1)
        end = cand.get("plaintext_end_byte", -1)
        return utf8_slice(doc, start, end)

    for ann in annotations:
        # Minimal answer span if available
        minimal = ann.get("minimal_answer", {}) or {}
        ms = minimal.get("plaintext_start_byte", -1)
        me = minimal.get("plaintext_end_byte", -1)
        if ms is not None and me is not None and ms >= 0 and me >= 0:
            ans = utf8_slice(doc, ms, me)
            if ans:
                answers.add(ans)

        # Passage-level candidate index
        passage = ann.get("passage_answer", {}) or {}
        idx = passage.get("candidate_index", -1)
        if idx is not None and idx >= 0:
            ans = candidate_span(idx)
            if ans:
                answers.add(ans)

    # Return sorted list for determinism
    return sorted(a for a in answers if a)


def find_audio_dir(jsonl_path: Path) -> Optional[Path]:
    """Find the corresponding wav_* directory next to the JSONL file."""
    dialect_dir = jsonl_path.parent
    # Prefer any directory whose name starts with 'wav_'
    for child in dialect_dir.iterdir():
        if child.is_dir() and child.name.startswith("wav_"):
            return child
    return None


def triples_output_path(jsonl_path: Path) -> Path:
    """
    Derive the triples file path from the JSONL path.
    Examples:
      english.test.gbr-en-GB.jsonl     -> english.test.gbr-en-GB.triples.jsonl
      english.test.aus-en-AU.jsonl.gz  -> english.test.aus-en-AU.triples.jsonl
    """
    name = jsonl_path.name
    if name.endswith(".jsonl.gz"):
        stem = name[:-len(".jsonl.gz")]
    elif name.endswith(".jsonl"):
        stem = name[:-len(".jsonl")]
    else:
        stem = jsonl_path.stem
    return jsonl_path.with_name(f"{stem}.triples.jsonl")


def process_single_jsonl(jsonl_path: Path) -> None:
    """Process one JSONL/JSONL.GZ file and write a .triples.jsonl next to it."""
    audio_dir = find_audio_dir(jsonl_path)
    out_path = triples_output_path(jsonl_path)

    # Repository root (assumed to be the directory containing this script)
    repo_root = Path(__file__).resolve().parent

    is_gz = jsonl_path.name.endswith(".jsonl.gz")
    open_func = gzip.open if is_gz else open

    with open_func(jsonl_path, "rt", encoding="utf-8") as f_in, \
            out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
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

            answers = extract_answers(ex)
            has_answer = len(answers) > 0
            single_answer = answers[0] if has_answer else "-1"

            audio_path: Optional[str] = None
            audio_exists = False
            if audio_dir is not None:
                candidate = audio_dir / f"{example_id}.wav"
                audio_exists = candidate.exists()
                try:
                    rel = candidate.resolve().relative_to(repo_root)
                    # Path starting with 'SD-QA/...'
                    audio_path = str(Path(repo_root.name) / rel)
                except ValueError:
                    # Fallback to just repo-relative or absolute path
                    try:
                        rel = candidate.relative_to(repo_root)
                        audio_path = str(Path(repo_root.name) / rel)
                    except ValueError:
                        audio_path = str(candidate)

            out_obj = {
                "example_id": example_id,
                "question": question_text,
                "answer": single_answer,
                "answers": answers,
                "has_answer": has_answer,
                "audio_path": audio_path,
                "audio_exists": audio_exists,
            }
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    print(f"Wrote triples: {out_path}", file=sys.stderr)


def collect_jsonl_files(root: Path) -> List[Path]:
    """
    Collect all relevant JSONL / JSONL.GZ files under root/test and root/dev.
    Prefer uncompressed .jsonl when both .jsonl and .jsonl.gz exist.
    """
    files: List[Path] = []
    for split in ("test", "dev"):
        split_dir = root / split
        if not split_dir.exists():
            continue

        # First collect plain .jsonl files, excluding already-generated triples.
        plain = [
            p for p in split_dir.rglob("*.jsonl")
            if not p.name.endswith(".triples.jsonl")
        ]
        files.extend(plain)
        plain_stems = {p.name[:-len(".jsonl")] for p in plain}

        # Then gzipped .jsonl.gz files whose base name is not already present,
        # also skipping any '*.triples.jsonl.gz' just in case.
        for gz in split_dir.rglob("*.jsonl.gz"):
            if gz.name.endswith(".triples.jsonl.gz"):
                continue
            base = gz.name[:-len(".jsonl.gz")]
            if base not in plain_stems:
                files.append(gz)

    return sorted(files)


def main(argv: List[str]) -> int:
    if len(argv) == 2:
        target = Path(argv[1])
    else:
        # Default: repository root (directory of this script)
        target = Path(__file__).resolve().parent

    if target.is_file():
        if not target.exists():
            print(f"Error: file not found: {target}", file=sys.stderr)
            return 1
        process_single_jsonl(target)
        return 0

    if not target.is_dir():
        print(f"Error: path is neither file nor directory: {target}", file=sys.stderr)
        return 1

    # Directory mode: walk test/ and dev/ and process each JSONL file.
    jsonl_files = collect_jsonl_files(target)
    if not jsonl_files:
        print(f"No JSONL files found under {target}/test or {target}/dev", file=sys.stderr)
        return 1

    print(f"Found {len(jsonl_files)} JSONL files to process.", file=sys.stderr)
    for path in jsonl_files:
        process_single_jsonl(path)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))


