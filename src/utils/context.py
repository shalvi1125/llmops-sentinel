import re


_LIST_OR_SECTION_START = re.compile(r"^(?:[-*•]|\d{1,3}[.)]|[A-Z][A-Z0-9\s\-/]{3,}:?)$")
_SENTENCE_END = re.compile(r"[.!?:\"”')]\s*$")


def compact_context_text(text: str) -> str:
    """Normalize word-per-line and whitespace artifacts from PDF extraction while preserving structure.

    Stored ChromaDB chunk text is never mutated; compaction is applied only when building LLM prompts.
    """
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    raw_lines = [line.strip() for line in normalized.split("\n")]

    paragraphs: list[str] = []
    current_parts: list[str] = []

    def _flush_paragraph() -> None:
        if current_parts:
            paragraphs.append(" ".join(current_parts))
            current_parts.clear()

    saw_blank = False
    for line in raw_lines:
        if not line:
            saw_blank = True
            continue

        if not current_parts:
            current_parts.append(line)
            saw_blank = False
            continue

        prev_line = current_parts[-1]
        is_heading_prev = prev_line.isupper() and 2 <= len(prev_line) <= 64
        is_list_or_heading_curr = bool(_LIST_OR_SECTION_START.match(line)) or (
            line.isupper() and 2 <= len(line) <= 64
        )
        ends_sentence = bool(_SENTENCE_END.search(prev_line))

        if is_heading_prev or is_list_or_heading_curr or (saw_blank and ends_sentence):
            _flush_paragraph()
            current_parts.append(line)
        else:
            current_parts.append(line)

        saw_blank = False

    _flush_paragraph()
    return "\n\n".join(paragraphs).strip()


def format_context_for_llm(
    sources: list[str],
    max_chars: int = 3500,
    source_labels: list[str] | None = None,
) -> str:
    """Compact retrieved chunks cleanly and assemble prompt context within character bounds."""
    compacted_blocks: list[str] = []
    seen_blocks: set[str] = set()

    for idx, src in enumerate(sources):
        cleaned = compact_context_text(src)
        if not cleaned:
            continue
        dedup_key = cleaned.lower()
        if dedup_key in seen_blocks:
            continue
        seen_blocks.add(dedup_key)

        if source_labels and idx < len(source_labels) and source_labels[idx]:
            block = f"[{source_labels[idx]}] {cleaned}"
        else:
            block = cleaned
        compacted_blocks.append(block)

    combined = "\n\n".join(compacted_blocks)
    if len(combined) > max_chars:
        combined = combined[:max_chars].rstrip()
    return combined
