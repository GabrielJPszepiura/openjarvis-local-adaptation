"""SimpleAgent — single-turn query-to-response agent (no tool calling)."""

from __future__ import annotations

import re
from typing import Any, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.registry import AgentRegistry


_SOURCE_REQUEST_RE = re.compile(
    r"(?:\b(?:mostre|envie|inclua|liste|informe|forneca|cite)\b.{0,48}"
    r"\b(?:link|links|url|urls|fonte|fontes|referencia|referencias)\b|"
    r"\bcom\s+(?:link|links|url|urls|fonte|fontes|referencia|referencias)\b)",
    re.IGNORECASE,
)

_SOURCE_PRESENTATION_INSTRUCTION = """

<source_presentation>
O usuário pediu as fontes da conversa. Para cada fonte já disponível no
histórico, mantenha o título como link e acrescente, após um travessão, um
resumo curto de uma ou duas frases sobre o que aquela fonte informa e como ela
contribuiu para a resposta. Baseie o resumo somente no conteúdo já pesquisado.
Não exiba rótulos genéricos de classificação, como “fonte secundária ou
comercial”, e não invente URLs, títulos ou detalhes que não estejam no
histórico.
</source_presentation>
"""


def _source_presentation_input(input_text: str) -> str:
    """Add useful source-display guidance only when the user requests sources."""
    if _SOURCE_REQUEST_RE.search(input_text):
        return input_text + _SOURCE_PRESENTATION_INSTRUCTION
    return input_text


@AgentRegistry.register("simple")
class SimpleAgent(BaseAgent):
    """Single-turn agent: query -> model -> response.  No tool calling."""

    agent_id = "simple"
    supports_managed_tool_fallback = True

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Single-turn: build messages, call engine, return result."""
        self._emit_turn_start(input)

        messages = self._build_messages(_source_presentation_input(input), context)
        result = self._generate(messages)
        content = result.get("content", "")

        self._emit_turn_end(content_length=len(content))

        return AgentResult(content=content, turns=1)


__all__ = ["SimpleAgent"]
