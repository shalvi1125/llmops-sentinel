from dataclasses import dataclass


@dataclass(frozen=True)
class PromptStrategy:
    id: str
    name: str
    version: str
    description: str
    template: str


DEFAULT_QA_TEMPLATE = """
Answer using ONLY the provided context, in the same language as the question.
If the context lacks sufficient evidence, state: "The provided document context does not contain sufficient information to answer this question."
Do not fabricate or extrapolate. Be concise and direct.

Context: {contexto}

Question: {question}
"""

STRICT_GROUNDING_TEMPLATE = """
Strict QA auditor: answer using ONLY explicit facts in the provided context, in the question's language.
If definitive evidence is missing, state: "The provided context does not contain sufficient information to answer this question."
Do not infer or use outside knowledge.

Context: {contexto}

Question: {question}
"""

CONCISE_FACTS_TEMPLATE = """
Present the core facts answering the question as concise bullet points using ONLY the provided context, in the question's language.
If the context is insufficient, state: "The provided document context does not contain sufficient information to answer this question."
Omit preamble and repetition.

Context: {contexto}

Question: {question}
"""

PROMPT_CATALOG: dict[str, PromptStrategy] = {
    "default_qa": PromptStrategy(
        id="default_qa",
        name="Balanced QA (Default)",
        version="1.1.0",
        description="Standard balanced QA with language matching, strict factual grounding, and insufficient-context refusal.",
        template=DEFAULT_QA_TEMPLATE.strip(),
    ),
    "strict_grounding": PromptStrategy(
        id="strict_grounding",
        name="Strict Grounding Auditor",
        version="1.1.0",
        description="Zero-hallucination auditor mode; strictly requires direct evidence and refuses ungrounded speculation.",
        template=STRICT_GROUNDING_TEMPLATE.strip(),
    ),
    "concise_facts": PromptStrategy(
        id="concise_facts",
        name="Concise Executive Facts",
        version="1.1.0",
        description="High-density bullet point summary of core facts grounded strictly in retrieved context.",
        template=CONCISE_FACTS_TEMPLATE.strip(),
    ),
}


def get_prompt_strategy(prompt_id: str | None = None) -> PromptStrategy:
    """Retrieve a prompt strategy by ID, falling back to default_qa."""
    if not prompt_id or prompt_id not in PROMPT_CATALOG:
        return PROMPT_CATALOG["default_qa"]
    return PROMPT_CATALOG[prompt_id]


def list_prompt_strategies() -> list[PromptStrategy]:
    """Return all available prompt strategies."""
    return list(PROMPT_CATALOG.values())
