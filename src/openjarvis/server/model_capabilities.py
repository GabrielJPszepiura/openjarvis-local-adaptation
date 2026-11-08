"""Model capability helpers shared by server model-selection routes."""

import re

_EMBEDDING_MODEL_PREFIXES = (
    "all-minilm",
    "bge-",
    "bge_",
    "e5-",
    "e5_",
    "gte-",
    "gte_",
    "jina-embeddings",
    "nomic-bert",
    "sentence-transformers",
)


def is_embed_only_model(model_name: str) -> bool:
    """Return whether a model identifier denotes a non-chat embedder.

    Ollama does not expose capabilities through its model-list response, so
    model selection needs a conservative name-based guard.  Most embedding
    models contain ``embed``; the explicit prefixes cover common families
    such as MiniLM, BGE, E5, and GTE whose names do not.
    """
    name = (model_name or "").strip().lower()
    leaf = name.rsplit("/", 1)[-1].split(":", 1)[0]
    return (
        "embed" in leaf
        or "minilm" in leaf
        or leaf.startswith(_EMBEDDING_MODEL_PREFIXES)
    )


def requests_file_creation(text: str) -> bool:
    """Detect explicit requests to create a downloadable office file."""
    value = text or ""
    # A negated request must not trigger the capability fallback. The full
    # orchestrator performs the same constraint-aware routing later; this
    # early check only decides whether a temporary model switch is needed.
    value = re.sub(
        r"\b(?:sem|n[aã]o)\s+"
        r"(?:(?:quero|desejo|preciso|necessito)(?:\s+que|\s+de)?\s+)?"
        r"(?:(?:deve|deva|pode|possa)\s+)?"
        r"(?:(?:fa[cç]a|fazer|crie|criar|gere|gerar|produza|produzir|"
        r"monte|montar|exporte|exportar|salve|salvar|entregue|entregar|"
        r"retorne|retornar)\s+)?"
        r"(?:(?:de|um|uma|o|a|os|as|nenhum|nenhuma|qualquer)\s+)*"
        r"(?:arquivo|documento|relat[oó]rio|"
        r"planilha|word|docx|pdf|excel|xlsx|file|document|report|spreadsheet|"
        r"workbook)\b",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    if re.search(
        r"\b(?:responda|mostre|apresente|retorne)\s+(?:somente|apenas)\s+"
        r"(?:aqui\s+)?(?:no\s+)?chat\b",
        value,
        re.IGNORECASE,
    ):
        return False
    action = re.search(
        r"\b(crie|criar|gere|gerar|exporte|exportar|salve|salvar|create|generate|export|save)\b",
        value,
        re.IGNORECASE,
    )
    artifact = re.search(
        r"\b(arquivo|documento|word|docx|pdf|planilha|excel|xlsx|file|document|spreadsheet|workbook)\b",
        value,
        re.IGNORECASE,
    )
    return bool(action and artifact)


__all__ = ["is_embed_only_model", "requests_file_creation"]
