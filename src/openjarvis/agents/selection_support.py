"""Contextual choice and comparison support for the orchestrator.

This module owns the state-free rules that bind a follow-up such as
``qual dos três?`` to the option set that created it.  It intentionally does
not call models, tools, or the web: the orchestrator remains responsible for
routing and execution.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

from openjarvis.core.types import Message, Role

_NUMBER_WORDS = {
    "um": 1,
    "uma": 1,
    "dois": 2,
    "duas": 2,
    "tres": 3,
    "quatro": 4,
    "cinco": 5,
}
_CONTEXT_RE = re.compile(
    r"<!--\s*openjarvis-research-context:(?P<payload>.*?)\s*-->", re.DOTALL | re.I
)
_SOURCES_RE = re.compile(
    r"<!--\s*openjarvis-research-sources:(?P<payload>.*?)\s*-->", re.DOTALL | re.I
)
_ORDERED_ITEM_RE = re.compile(
    r"^\s*(?:[-*+]\s+)?(?:\*\*)?(?:\d+[.)])\s*(?:\*\*)?"
    r"(?P<item>[^\n:*]{2,100})(?:\*\*)?\s*(?::|[-–—]|$)",
    re.MULTILINE,
)


def _plain(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(
        char for char in normalized if not unicodedata.combining(char)
    ).lower()


def requested_item_kind(input_text: str) -> str | None:
    text = _plain(input_text)
    # The subject tied to the explicit quantity wins over incidental words
    # later in the request.  For example, "cinco notebooks ... cada opção"
    # must be stored as notebooks, not as generic options.
    match = re.search(
        r"\b(?:\d+|um|uma|dois|duas|tres|quatro|cinco|seis|sete|oito)\s+(?:(?:melhores|principais|recomendad[oa]s?)\s+)?(?P<kind>[a-z]{3,})",
        text,
    )
    if match:
        return match.group("kind")
    for singular, plural in (
        ("pais", "paises"),
        ("cidade", "cidades"),
        ("produto", "produtos"),
        ("tratamento", "tratamentos"),
        ("estudo", "estudos"),
        ("opcao", "opcoes"),
    ):
        if singular in text or plural in text:
            return plural
    return None


def requested_list_count(input_text: str) -> int | None:
    match = re.search(
        r"\b(?:(\d+)|(um|uma|dois|duas|tres|quatro|cinco|seis|sete|oito))\s+(?:(?:melhores|principais|recomendad[oa]s?)\s+)?[a-z]{3,}",
        _plain(input_text),
    )
    if not match:
        return None
    if match.group(1):
        return min(int(match.group(1)), 8)
    return {**_NUMBER_WORDS, "seis": 6, "sete": 7, "oito": 8}.get(match.group(2))


def presented_items(text: str) -> list[str]:
    """Extract labels from an answer without treating prose as options."""

    def collect(patterns: tuple[re.Pattern[str], ...]) -> list[str]:
        items: list[str] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                item = re.sub(r"\s+", " ", match.group("item")).strip(" .,:;-")
                if item and item not in items:
                    items.append(item)
        return items

    numbered = collect(
        (
            _ORDERED_ITEM_RE,
            re.compile(
                r"^\s*(?:\*\*)?\d+[.)]\s*(?P<item>[^\n:*]{2,100})(?:\*\*)?",
                re.MULTILINE,
            ),
        )
    )
    if len(numbered) >= 2:
        return numbered
    return collect(
        (
            re.compile(
                r"^\s*#{1,6}\s*\d+[.)]?\s*(?:\*\*)?(?P<item>[^\n:*]{2,100})",
                re.MULTILINE,
            ),
            re.compile(
                r"^\s*#{2,6}\s*(?:\*\*)?(?P<item>[^\n:*]{2,100})",
                re.MULTILINE,
            ),
        )
    )


def research_context_marker(input_text: str, answer: str) -> str:
    kind, items = requested_item_kind(input_text), presented_items(answer)
    if not kind or len(items) < 2:
        return ""
    return (
        "\n<!-- openjarvis-research-context:"
        + json.dumps({"kind": kind, "items": items[:8]}, ensure_ascii=False)
        + " -->"
    )


def _reference(input_text: str) -> tuple[int | None, str | None]:
    text = _plain(input_text)
    match = re.search(
        r"\b(?:entre|dentre)\s+(?:esses?|essas?|os|as)?\s*(?:(\d+)|(um|uma|dois|duas|tres|quatro|cinco))?",
        text,
    )
    count = None
    if match:
        count = (
            min(int(match.group(1)), 8)
            if match.group(1)
            else _NUMBER_WORDS.get(match.group(2) or "")
        )
    if count is None:
        match = re.search(
            r"\b(?:esses?|essas?|estes?|estas?|os|as|dos|das)\s+(?:(\d+)|(um|uma|dois|duas|tres|quatro|cinco))\s+(?:(?:melhores|principais)\s+)?[a-z]{3,}\b",
            text,
        )
        if match:
            count = (
                min(int(match.group(1)), 8)
                if match.group(1)
                else _NUMBER_WORDS.get(match.group(2) or "")
            )
    return count, requested_item_kind(input_text)


def _record(message: Message) -> dict[str, object] | None:
    match = _CONTEXT_RE.search(message.text)
    if not match:
        return None
    try:
        value = json.loads(match.group("payload"))
    except json.JSONDecodeError:
        return None
    items = value.get("items") if isinstance(value, dict) else None
    kind = str(value.get("kind", "")) if isinstance(value, dict) else ""
    if not kind or not isinstance(items, list):
        return None
    return {"kind": kind, "items": [str(item) for item in items if str(item).strip()]}


def _is_source_listing(message: Message) -> bool:
    return message.text.lstrip().startswith(
        "Estas são as fontes usadas na resposta anterior:"
    )


def resolve_options(input_text: str, messages: list[Message]) -> list[str]:
    count, requested_kind = _reference(input_text)
    candidates: list[dict[str, object]] = []
    legacy_generic_candidates: list[dict[str, object]] = []
    for message in reversed(messages):
        if message.role != Role.ASSISTANT or _is_source_listing(message):
            continue
        record = _record(message)
        if not record:
            continue
        items = list(record["items"])
        if count and len(items) < count:
            continue
        if requested_kind and record["kind"] != requested_kind:
            # Early versions could classify "cinco notebooks ... explique cada
            # opção" as generic ``opcoes``. Keep that old conversation usable
            # only when it is the sole compatible fallback; it never outranks
            # a correctly typed record and cannot make an ambiguous choice.
            if record["kind"] == "opcoes":
                legacy_generic_candidates.append(record)
            continue
        candidates.append(record)
    if not candidates:
        if requested_kind and len(legacy_generic_candidates) == 1:
            return list(legacy_generic_candidates[0]["items"])
        return []
    # Untyped references are ambiguous when more than one recorded set fits.
    if not requested_kind and len(candidates) != 1:
        return []
    return list(candidates[0]["items"])


def _sources(message: Message) -> list[dict[str, str]]:
    match = _SOURCES_RE.search(message.text)
    if not match:
        return []
    try:
        values = json.loads(match.group("payload"))
    except json.JSONDecodeError:
        return []
    return (
        [
            value
            for value in values
            if isinstance(value, dict) and value.get("title") and value.get("url")
        ]
        if isinstance(values, list)
        else []
    )


def prior_research_context(input_text: str, messages: list[Message]) -> str:
    options = resolve_options(input_text, messages)
    if not options:
        return ""
    for message in reversed(messages):
        record = _record(message) if message.role == Role.ASSISTANT else None
        if record and list(record["items"]) == options:
            body = _SOURCES_RE.sub("", _CONTEXT_RE.sub("", message.text)).strip()[:3000]
            source_lines = "\n".join(
                f"- {item['title']}: {item['url']}" for item in _sources(message)[:6]
            )
            return (
                "\n<prior_research_context>\nPesquisa anterior vinculada às opções referidas neste pedido. "
                "Use-a junto da pesquisa nova; não use nenhuma outra parte do histórico.\n"
                f"Opções permitidas para a escolha: {'; '.join(options)}.\n{body}"
                + ("\nFontes registradas:\n" + source_lines if source_lines else "")
                + "\n</prior_research_context>"
            )
    return ""


def choice_instruction(input_text: str) -> str:
    text = _plain(input_text)
    if not (
        re.search(r"\b(?:entre|dentre)\b", text)
        and re.search(r"\b(?:qual|melhor|indicaria|recomenda|escolhe|escolher)\b", text)
    ):
        return ""
    return (
        "\n<choice_follow_up>\nEsta é uma escolha entre opções já apresentadas na conversa. Use a pesquisa anterior "
        "vinculada a essas opções e a nova pesquisa para reforçar a decisão, mas não repita a apresentação completa "
        "de todas as opções. Comece escolhendo claramente uma única opção. Desenvolva mais essa escolhida com os "
        "detalhes novos e úteis para o critério do usuário. Depois, explique de forma breve por que as demais ficaram atrás.\n</choice_follow_up>"
    )


def comparison_instruction(input_text: str) -> str:
    text = _plain(input_text)
    count, kind = _reference(input_text)
    contextual = bool(
        re.search(
            r"\b(?:esse|esses|essa|essas|desse|desses|dessa|dessas|aquele|aqueles|aquela|aquelas|anterior|anteriores|mesmo|mesmos|mesma|mesmas|eles|elas)\b",
            text,
        )
    ) or bool(count and kind)
    if not (
        contextual
        and re.search(
            r"\b(?:compare|comparacao|vantagens?|desvantagens?|cuidados|pontos)\b", text
        )
    ):
        return ""
    return (
        "\n<contextual_comparison>\nCompare somente as opções já apresentadas e pesquise cada uma delas separadamente. "
        "Não introduza alternativas novas; para cada opção, apresente pontos a favor e cuidados relevantes.\n</contextual_comparison>"
    )


def research_criterion(input_text: str) -> str:
    match = re.search(r"\bpara\s+(.+)", _plain(input_text))
    if match:
        return match.group(1).strip(" .,:;?")
    return (
        re.sub(
            r"^.*?\b(?:qual|melhor|indicaria|recomenda)\b", "", _plain(input_text)
        ).strip(" .,:;?")
        or "comparação"
    )


@dataclass(frozen=True)
class SelectionFollowUp:
    options: list[str]
    choice_instruction: str
    comparison_instruction: str
    prior_research_context: str

    @property
    def active(self) -> bool:
        return bool(self.choice_instruction or self.comparison_instruction)


def resolve_follow_up(input_text: str, messages: list[Message]) -> SelectionFollowUp:
    choice = choice_instruction(input_text)
    comparison = comparison_instruction(input_text)
    options = resolve_options(input_text, messages) if choice or comparison else []
    return SelectionFollowUp(
        options,
        choice,
        comparison,
        prior_research_context(input_text, messages) if options else "",
    )


_INTERNAL_PROCESS_RE = re.compile(
    r"\b(?:fontes?|evid[eê]ncias?|hist[oó]rico|conjunto\s+(?:fechado|pr[eé]-selecionado)|"
    r"resultados?\s+(?:da|de)\s+pesquisa)\b.{0,90}\b(?:n[aã]o\s+(?:permitem|"
    r"identificaram|fornecem)|insuficient|imposs[ií]vel)\b",
    re.IGNORECASE,
)


def option_guard(options: list[str]) -> str:
    """Render the non-negotiable boundary passed to a local model."""
    return (
        "\n<locked_options>\n"
        "As únicas opções válidas neste pedido são: "
        + "; ".join(options)
        + ". Não introduza, recomende, trate como concorrente ou compare uma "
        "opção fora dessa lista. Não explique ao usuário esta regra, fontes, "
        "histórico ou limitações internas.\n</locked_options>"
    )


def response_needs_repair(
    response: str,
    *,
    options: list[str],
    is_choice: bool,
    is_comparison: bool,
) -> bool:
    """Detect when a small model ignored a contextual option boundary.

    The check intentionally favours a retry over trying to rewrite facts in
    Python.  It only verifies visible structure: a choice must name an allowed
    winner near the beginning; a comparison must visibly identify every item.
    """
    plain_response = _plain(response)
    normalized_options = [_plain(option) for option in options]
    if _INTERNAL_PROCESS_RE.search(response):
        return True
    if is_choice:
        # The winner belongs in the opening sentence. Looking through the
        # whole answer would accept "Brasil" as winner merely because China
        # appears later in the explanation of why it lost.
        opening = re.split(r"[.!?\n]", plain_response, maxsplit=1)[0][:420]
        return not any(option in opening for option in normalized_options)
    if is_comparison:
        # Merely naming every option is not a comparison. Each one needs its
        # own section with the two parts the user asked for.
        for option in normalized_options:
            heading = re.compile(
                rf"^\s*#{{1,6}}\s+(?:\*\*)?{re.escape(option)}(?:\*\*)?\s*$",
                re.MULTILINE,
            )
            match = heading.search(plain_response)
            if not match:
                return True
            following = plain_response[match.end() :]
            next_positions = [
                position
                for other in normalized_options
                if other != option
                for position in [following.find(other)]
                if position >= 0
            ]
            section = following[: min(next_positions)] if next_positions else following
            if "pontos positivos" not in section or "cuidados" not in section:
                return True
        return False
    return False


def repair_instruction(
    options: list[str], *, is_choice: bool, is_comparison: bool
) -> str:
    """Request one user-facing repair without exposing system mechanics."""
    if is_choice:
        structure = (
            "Comece a primeira frase com `Minha indicação: **NOME DA OPÇÃO**`. "
            "O nome deve ser exatamente uma das opções permitidas. Depois "
            "explique essa escolha e diga, em frases curtas, por que cada uma "
            "das demais ficou atrás."
        )
    elif is_comparison:
        structure = (
            "Use uma seção identificada pelo nome exato de cada opção e, em "
            "cada seção, apresente `Pontos positivos` e `Cuidados`."
        )
    else:
        structure = "Responda diretamente ao pedido com as opções permitidas."
    return (
        "\n<mandatory_contextual_repair>\n"
        "Reescreva a resposta de forma direta e natural para o usuário. "
        + structure
        + " Não mencione fontes, evidências, histórico, conjuntos de opções, "
        "buscas, limitações internas ou impossibilidade de comparar. "
        "Use somente estas opções: "
        + "; ".join(options)
        + ".\n</mandatory_contextual_repair>"
    )


def safe_failure_response(options: list[str], *, is_choice: bool) -> str:
    """Return a natural fallback when a retry still violates the boundary."""
    if is_choice:
        return (
            "Não consegui concluir uma indicação confiável entre "
            + ", ".join(options)
            + ". Posso comparar essas opções por orçamento, segurança ou "
            "facilidade para iniciantes."
        )
    return (
        "Não consegui apresentar uma comparação confiável entre "
        + ", ".join(options)
        + ". Posso reorganizar a comparação por um critério específico."
    )
