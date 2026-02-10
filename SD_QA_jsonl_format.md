### SD-QA JSONL Annotation Format (TyDi QA-Compatible)

This document explains how to interpret the SD-QA `*.jsonl` files (e.g. `english.test.gbr-en-GB.jsonl`) and how they relate to the TyDi QA format.  
The structure is essentially **identical to TyDi QA**, and you can refer to the official TyDi evaluation script for further details:  
[`tydi_eval.py`](https://github.com/google-research-datasets/tydiqa/blob/master/tydi_eval.py).

The only substantive difference is that in SD-QA **the `question_text` is replaced** with the SD-QA question, while the rest of the annotation schema (document, candidates, annotations, byte offsets) is the same as TyDi QA.

---

### 1. One JSON Object per Example

Each line in a `*.jsonl` file is a single example with fields like:

```json
{
  "annotations": [...],
  "document_plaintext": "... full Wikipedia article text ...",
  "document_title": "Rock (geology)",
  "document_url": "https://en.wikipedia.org/wiki/Rock%20%28geology%29",
  "example_id": -1886191442981300792,
  "language": "english",
  "passage_answer_candidates": [
    {"plaintext_start_byte": 5, "plaintext_end_byte": 265},
    {"plaintext_start_byte": 266, "plaintext_end_byte": 396},
    ...
  ],
  "question_text": "what is the most common type of rock on Earth ?"
}
```

- **`example_id`**: The unique key that ties together:
  - JSONL line (question + answers + document)
  - `*.txt` line (`example_id<TAB>question_text`)
  - WAV file (`{example_id}.wav` in `wav_*` directories)
- **`question_text`**: The SD-QA question for this example (replaces the original TyDi QA question, but plays the same role).
- **`document_plaintext`**: The full article text from which answer spans are taken.
- **`passage_answer_candidates`**: Pre-segmented candidate spans within `document_plaintext`, defined in **UTF‑8 byte offsets**.

---

### 2. Byte Offsets: `plaintext_start_byte` / `plaintext_end_byte`

All spans are stored as **byte offsets into `document_plaintext` encoded as UTF‑8**:

- For a candidate span:

  ```json
  {"plaintext_start_byte": 266, "plaintext_end_byte": 396}
  ```

  This means:

  ```python
  text = example["document_plaintext"]
  b = text.encode("utf-8")
  span_bytes = b[266:396]
  answer_text = span_bytes.decode("utf-8")
  ```

  The `answer_text` is the candidate passage span.

- The same convention is used for minimal answers:

  ```json
  "minimal_answer": {
    "plaintext_start_byte": 3133,
    "plaintext_end_byte": 3140
  }
  ```

  If either value is `-1`, it means **no span (null answer)**.

#### Why byte offsets and not character indices?

TyDi QA (and thus SD-QA) uses byte offsets to match the original Wikipedia dumps precisely and to avoid ambiguity with multi-byte UTF‑8 characters. The official evaluation script [`tydi_eval.py`](https://github.com/google-research-datasets/tydiqa/blob/master/tydi_eval.py) expects byte-level spans.

---

### 3. `passage_answer_candidates`

`passage_answer_candidates` is a **pool of all plausible answer spans** for the passage:

```json
"passage_answer_candidates": [
  {"plaintext_start_byte": 5, "plaintext_end_byte": 265},
  {"plaintext_start_byte": 266, "plaintext_end_byte": 396},
  ...
]
```

- Each element is one contiguous span in `document_plaintext`.
- There can be **many more candidates than actual answers**.
- Annotators (and systems) **do not** annotate spans directly in free-form; they select one of these candidate spans by index.

---

### 4. `annotations`: What the Annotators Chose

`annotations` is a list of N annotator labels for this example:

```json
"annotations": [
  {
    "annotation_id": 12852520138824648163,
    "minimal_answer": {
      "plaintext_start_byte": 3133,
      "plaintext_end_byte": 3140
    },
    "passage_answer": {
      "candidate_index": 11
    },
    "yes_no_answer": "NONE"
  },
  {
    "annotation_id": 2009769336257936557,
    "minimal_answer": {
      "plaintext_start_byte": 3133,
      "plaintext_end_byte": 3146
    },
    "passage_answer": {
      "candidate_index": 11
    },
    "yes_no_answer": "NONE"
  },
  ...
]
```

For each annotation:

- **`annotation_id`**: Identifies the annotator; not needed for evaluation or extraction.
- **`passage_answer.candidate_index`**:
  - If `>= 0`, points into `passage_answer_candidates[idx]`.
  - If `-1`, the annotator did **not** select any passage-level span.
- **`minimal_answer`**:
  - If both `plaintext_start_byte` and `plaintext_end_byte` are `>= 0`, this defines a **minimal answer span** in `document_plaintext`.
  - If either is `-1`, there is **no minimal span** from this annotator.
- **`yes_no_answer`**:
  - One of `"YES"`, `"NO"`, or `"NONE"` (TyDi QA convention).
  - For most SD-QA cases this will be `"NONE"` (span-based answers), but the field is present for full TyDi compatibility.

#### Important: What if all indices are `-1`?

If every annotation has:

```json
"minimal_answer": {"plaintext_start_byte": -1, "plaintext_end_byte": -1},
"passage_answer": {"candidate_index": -1}
```

then **this example is treated as having no gold span answer** (unanswerable / null).  
It is normal that `passage_answer_candidates` still contains many candidates; they are simply unused for this example.

This is exactly how TyDi QA defines null-answer examples; see the logic in `gold_has_passage_answer` and `gold_has_minimal_answer` in TyDi’s `eval_utils.py`, called from [`tydi_eval.py`](https://github.com/google-research-datasets/tydiqa/blob/master/tydi_eval.py).

---

### 5. How to Extract Gold Answers (TyDi-Style)

To get gold reference answers for a given `example_id` (question):

1. Load the JSONL line for that `example_id`.
2. Read:
   - `question_text`
   - `document_plaintext`
   - `annotations`
   - `passage_answer_candidates`
3. For each entry in `annotations`:
   - **Minimal answer first (if present)**:
     - If `minimal_answer.plaintext_start_byte >= 0` and `plaintext_end_byte >= 0`, slice:

       ```python
       start = minimal["plaintext_start_byte"]
       end = minimal["plaintext_end_byte"]
       answer = utf8_slice(document_plaintext, start, end)
       ```

   - **Otherwise, fall back to passage candidate**:
     - If `passage_answer.candidate_index >= 0`, say `k`:
       - Look up `passage_answer_candidates[k]` and slice with its byte offsets.

This is functionally the same as what TyDi QA does internally when evaluating minimal and passage answers, as implemented in [`tydi_eval.py`](https://github.com/google-research-datasets/tydiqa/blob/master/tydi_eval.py) and `eval_utils.py`.

> In SD-QA we typically:
> - Collect **all distinct non-empty spans** from all annotations, using the rules above.
> - Treat that set as the gold answer set for that `example_id`.

---

### 6. How This Relates to the Official TyDi Evaluation

The TyDi evaluation expects prediction JSON lines of the form (see [`tydi_eval.py`](https://github.com/google-research-datasets/tydiqa/blob/master/tydi_eval.py)):

```json
{
  "example_id": -2226525965842375672,
  "passage_answer_index": 2,
  "passage_answer_score": 13.5,
  "minimal_answer": {
    "start_byte_offset": 64206,
    "end_byte_offset": 64280
  },
  "minimal_answer_score": 26.4,
  "yes_no_answer": "NONE"
}
```

The **gold format** (our JSONL) mirrors this:

- `example_id`: same identifier.
- `passage_answer_index` ↔ `passage_answer.candidate_index` in `annotations`.
- `minimal_answer.start_byte_offset` / `end_byte_offset` ↔ `minimal_answer.plaintext_start_byte` / `plaintext_end_byte`.
- `yes_no_answer`: same semantics.

Because SD-QA’s JSONL is a **direct structural replica of the TyDi QA gold format** (with only `question_text` changed), you can:

- Use the same logic as TyDi to extract gold spans and to build predictions.
- Plug predictions into TyDi’s official evaluator without changing the gold side, as long as you respect the byte-offset conventions.

---

### 7. Triples Generation (`sdqa_triples_from_jsonl.py`)

For convenience, the repo includes a helper script to materialize the JSONL annotations into **(audio, question, answer)** triples.

- **Script**: `sdqa_triples_from_jsonl.py`
- **Location**: repository root (`SD-QA/sdqa_triples_from_jsonl.py`)

#### 7.1 What the script does

For each gold JSONL file (both `test/` and `dev/`):

- Reads `example_id`, `question_text`, `document_plaintext`, `annotations`, and `passage_answer_candidates`.
- Extracts **gold answer spans** per example using the TyDi logic described above:
  - Collects all non-null `minimal_answer` spans (using byte offsets into `document_plaintext`).
  - Also collects spans pointed to by `passage_answer.candidate_index` via `passage_answer_candidates`.
  - Deduplicates and sorts the spans; this set is the gold `answers` list.
- Resolves the corresponding WAV file if present:
  - Looks for a `wav_*` directory next to the JSONL file.
  - Builds a relative `audio_path` of the form `SD-QA/test/eng/gbr/wav_eng/{example_id}.wav`.

For every input `*.jsonl` / `*.jsonl.gz`, it writes a sibling `*.triples.jsonl`, e.g.:

- `english.test.gbr-en-GB.jsonl` → `english.test.gbr-en-GB.triples.jsonl`
- `english.test.aus-en-AU.jsonl.gz` → `english.test.aus-en-AU.triples.jsonl`

Each line in a `*.triples.jsonl` file looks like:

```json
{
  "example_id": -1886191442981300792,
  "question": "what is the most common type of rock on Earth ?",
  "answer": "basalt",                // first span or "-1" if no span
  "answers": ["basalt", "..."],      // all distinct gold spans
  "has_answer": true,
  "audio_path": "SD-QA/test/eng/gbr/wav_eng/-1886191442981300792.wav",
  "audio_exists": true
}
```

Notes:

- **`answers`** is the full set of gold spans for that `example_id`, coming from:
  - All `minimal_answer` spans across annotators, plus
  - All spans addressed by `passage_answer.candidate_index`.
- **`answer`** is just a convenience field:
  - If `answers` is non-empty, `answer` is set to the first span in `answers`.
  - If there is **no** gold span (all indices `-1`), `answer` is set to the string `"-1"` and `has_answer` is `false`.
- **`audio_path`** is repo-relative and always starts with `SD-QA/...`; `audio_exists` tells you whether the WAV file is actually present on disk.

#### 7.2 How to run the script

- **Recommended (all splits, all languages, all dialects):**

  ```bash
  cd /Users/faisal/Projects/SD-QA
  python3 sdqa_triples_from_jsonl.py .
  ```

  This will:
  - Walk `./test/` and `./dev/`
  - For every original `*.jsonl` / `*.jsonl.gz` file, create a corresponding `*.triples.jsonl` next to it.
  - It automatically skips any existing `*.triples.jsonl` so they are not re-processed.

- **Single file mode (debugging / ad hoc use):**

  ```bash
  cd /Users/faisal/Projects/SD-QA
  python3 sdqa_triples_from_jsonl.py test/eng/gbr/english.test.gbr-en-GB.jsonl
  ```

  This creates `test/eng/gbr/english.test.gbr-en-GB.triples.jsonl` only.

---

### 8. Summary

- SD-QA JSONL files follow the **TyDi QA gold annotation format** almost exactly.
- The critical link fields are:
  - `example_id` (connects audio, text, and JSONL)
  - `question_text`
  - `document_plaintext`
  - `passage_answer_candidates` (byte-offset candidates)
  - `annotations` (which candidate/minimal span each annotator picked)
- `plaintext_start_byte` / `plaintext_end_byte` are **UTF‑8 byte offsets** into `document_plaintext`.
- `-1` offsets and `candidate_index == -1` mean **no span (null answer)**.
- To get gold answers, you follow the same procedure that TyDi QA uses, as implemented in [`tydi_eval.py`](https://github.com/google-research-datasets/tydiqa/blob/master/tydi_eval.py).
- The helper script `sdqa_triples_from_jsonl.py` materializes these annotations into convenient triples that include:
  - question text
  - all gold answer spans (`answers`)
  - a single representative answer (`answer` or `"-1"`)
  - and the repo-relative path to the corresponding WAV file (`audio_path`).


