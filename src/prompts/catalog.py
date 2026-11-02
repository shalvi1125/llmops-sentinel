from dataclasses import dataclass


@dataclass(frozen=True)
class PromptStrategy:
    id: str
    name: str
    version: str
    description: str
    template: str


DEFAULT_QA_TEMPLATE = """
You are a QA expert. Answer the question below using the provided context. Respond in the same language as the user's question (for example, answer in English if the question is in English, in Portuguese if the question is in Portuguese, etc.), regardless of the language of the retrieved context documents. Ensure the answer is accurate, strictly grounded in the provided context, concise, and direct without unnecessary filler.

Context: {contexto}

Question: {question}
"""

STRICT_GROUNDING_TEMPLATE = """
You are a strict QA compliance auditor. Answer the question using ONLY the explicit facts stated in the provided context.
If the provided context does not contain enough evidence to answer the question definitively, you MUST state:
"The provided context does not contain sufficient information to answer this question."
Do not extrapolate, infer, or use outside knowledge. Respond in the same language as the user's question.

Context: {contexto}

Question: {question}
"""

CONCISE_FACTS_TEMPLATE = """
You are an executive research analyst. Extract and present the core facts answering the question as concise, high-density bullet points.
Omit conversational filler, preamble, and restating the question. Every bullet must be directly verifiable against the provided context. Respond in the same language as the user's question.

Context: {contexto}

Question: {question}
"""

PROMPT_CATALOG: dict[str, PromptStrategy] = {
    "default_qa": PromptStrategy(
        id="default_qa",
        name="Balanced QA (Default)",
        version="1.0.0",
        description="Standard balanced QA with language matching, high factual grounding, and direct answers.",
        template=DEFAULT_QA_TEMPLATE.strip(),
    ),
    "strict_grounding": PromptStrategy(
        id="strict_grounding",
        name="Strict Grounding Auditor",
        version="1.0.0",
        description="Zero-hallucination auditor mode; strictly requires direct evidence and refuses ungrounded speculation.",
        template=STRICT_GROUNDING_TEMPLATE.strip(),
    ),
    "concise_facts": PromptStrategy(
        id="concise_facts",
        name="Concise Executive Facts",
        version="1.0.0",
        description="High-density bullet point summary of core facts without conversational preamble.",
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
