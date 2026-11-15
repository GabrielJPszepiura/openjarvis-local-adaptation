"""Tests for the OrchestratorAgent."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from openjarvis.agents._stubs import AgentContext
from openjarvis.agents.orchestrator import (
    OrchestratorAgent,
    _attachment_research_terms,
    _chat_format_instruction,
    _countries_from_context,
    _filter_research_evidence,
    _hide_unrequested_links,
    _hide_unretrieved_links,
    _choice_follow_up_instruction,
    _choice_prior_research_context,
    _links_requested,
    _needs_evidence_research,
    _research_context_marker,
    _source_ledger_marker,
    _web_grounded_input,
    _web_search_query,
    _web_search_queries,
)
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import Conversation, Message, Role, ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec
from openjarvis.tools.document_create import (
    CreateDocumentTool,
    CreateSpreadsheetTool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CalculatorStub(BaseTool):
    tool_id = "calculator"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculator",
            description="Math calculator.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )

    def execute(self, **params) -> ToolResult:
        expr = params.get("expression", "0")
        try:
            val = eval(expr)  # noqa: S307 — safe in tests
        except Exception as e:
            return ToolResult(tool_name="calculator", content=str(e), success=False)
        return ToolResult(tool_name="calculator", content=str(val), success=True)


class _ThinkStub(BaseTool):
    tool_id = "think"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="think",
            description="Thinking tool.",
            parameters={
                "type": "object",
                "properties": {"thought": {"type": "string"}},
            },
        )

    def execute(self, **params) -> ToolResult:
        return ToolResult(
            tool_name="think",
            content=params.get("thought", ""),
            success=True,
        )


class _FileReadStub(BaseTool):
    tool_id = "file_read"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file_read",
            description="Read a local file.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )

    def execute(self, **params) -> ToolResult:
        return ToolResult(
            tool_name="file_read",
            content="should not be called",
            success=False,
        )


class _WebSearchStub(BaseTool):
    tool_id = "web_search"

    def __init__(self):
        self.calls = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_search",
            description="Search the web.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

    def execute(self, **params) -> ToolResult:
        self.calls.append(params)
        return ToolResult(
            tool_name="web_search",
            content="Resultado confiável da pesquisa.",
            success=True,
        )


def _make_engine_no_tools(content: str = "Final answer.") -> MagicMock:
    """Engine that never returns tool calls."""
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.return_value = {
        "content": content,
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "model": "test-model",
        "finish_reason": "stop",
    }
    return engine


def _make_engine_with_tool_call(
    tool_name: str = "calculator",
    arguments: str = '{"expression":"2+2"}',
    tool_call_id: str = "call_1",
    final_content: str = "The answer is 4.",
) -> MagicMock:
    """Engine that returns one tool call then a final answer."""
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = [
        # First call: tool call
        {
            "content": "",
            "tool_calls": [
                {"id": tool_call_id, "name": tool_name, "arguments": arguments}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "test-model",
            "finish_reason": "tool_calls",
        },
        # Second call: final answer
        {
            "content": final_content,
            "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
            "model": "test-model",
            "finish_reason": "stop",
        },
    ]
    return engine


def _make_engine_multi_tool() -> MagicMock:
    """Engine that calls multiple tools in one turn."""
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "calculator",
                    "arguments": '{"expression":"2+2"}',
                },
                {
                    "id": "call_2",
                    "name": "think",
                    "arguments": '{"thought":"thinking..."}',
                },
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "test-model",
            "finish_reason": "tool_calls",
        },
        {
            "content": "Done.",
            "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
            "model": "test-model",
            "finish_reason": "stop",
        },
    ]
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOrchestratorAgent:
    def test_contextual_choice_requests_a_single_focused_recommendation(self):
        instruction = _choice_follow_up_instruction(
            "Entre esses três, qual você indicaria para uma primeira viagem?"
        )

        assert "uma única opção" in instruction
        assert "não repita a apresentação completa" in instruction
        assert _choice_follow_up_instruction("Compare Japão e Croácia.") == ""

    def test_choice_uses_the_matching_prior_research_not_the_latest_topic(self):
        travel_answer = (
            "1. **Japão**: cultura e transporte.\n"
            "2. **Croácia**: litoral e história.\n"
            "3. **Espanha**: cidades e gastronomia."
        )
        travel_instruction = "Quais são três países para uma viagem de turismo?"
        debt_answer = "A dívida pública brasileira é acompanhada pelo Tesouro."
        messages = [
            Message(
                role=Role.ASSISTANT,
                content=(
                    travel_answer
                    + _research_context_marker(travel_instruction, travel_answer)
                    + _source_ledger_marker(
                        [{"title": "Turismo Japão", "url": "https://tour.example/jp"}]
                    )
                ),
            ),
            Message(role=Role.USER, content="Agora explique a dívida pública do Brasil."),
            Message(
                role=Role.ASSISTANT,
                content=(
                    debt_answer
                    + _research_context_marker(
                        "Explique dados sobre a dívida pública do Brasil.", debt_answer
                    )
                ),
            ),
        ]
        request = "Entre esses três países, qual você indicaria para uma primeira viagem?"

        prior = _choice_prior_research_context(request, messages)
        query = _web_search_query(request, messages)

        assert "Japão" in prior
        assert "Croácia" in prior
        assert "dívida pública" not in prior
        assert "Turismo Japão" in prior
        assert "Japão" in query
        assert "Croácia" in query
        assert "Brasil" not in query

    def test_ambiguous_choice_does_not_merge_two_untyped_option_sets(self):
        first = (
            "1. **Alpha**: primeiro.\n2. **Beta**: segundo.\n"
            "3. **Gama**: terceiro."
        )
        second = (
            "1. **Xenon**: primeiro.\n2. **Ypsilon**: segundo.\n"
            "3. **Zeta**: terceiro."
        )
        messages = [
            Message(
                role=Role.ASSISTANT,
                content=first + _research_context_marker("Liste três opções.", first),
            ),
            Message(
                role=Role.ASSISTANT,
                content=second + _research_context_marker("Liste três opções.", second),
            ),
        ]

        assert _choice_prior_research_context("Entre esses três, qual é melhor?", messages) == ""

    def test_explicit_word_request_creates_file_when_model_omits_tool(self, tmp_path):
        engine = _make_engine_no_tools("Resumo produzido a partir das planilhas.")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
        )

        result = agent.run(
            "Analise estas planilhas e me retorne um resumo em formato Word."
        )

        assert len(result.tool_results) == 1
        created = result.tool_results[0]
        assert created.success is True
        assert created.tool_name == "create_document"
        assert created.metadata["format"] == "docx"
        assert (tmp_path / created.metadata["filename"]).is_file()
        assert result.metadata["artifact_intent_fallback"] is True
        assert result.content.startswith("Arquivo criado:")

        messages = engine.generate.call_args.args[0]
        assert "tools" not in engine.generate.call_args.kwargs
        assert messages[-1].text.startswith("/no_think\n")
        assert "histórico da conversa" in messages[-1].text
        assert "somente o conteúdo final" in messages[-1].text

    def test_word_request_temporarily_disables_unneeded_tools(self, tmp_path):
        engine = _make_engine_no_tools("Conteúdo final do relatório.")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[
                _CalculatorStub(),
                CreateDocumentTool([str(tmp_path)]),
            ],
        )

        result = agent.run("Crie um documento Word com os resultados anteriores.")

        assert "tools" not in engine.generate.call_args.kwargs
        assert result.metadata["tool_route"] == "artifact_only"
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "create_document"
        assert result.tool_results[0].success is True

    def test_word_request_uses_larger_document_only_budget(self, tmp_path):
        engine = _make_engine_no_tools("Conteúdo final do relatório.")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
            max_tokens=256,
        )

        agent.run("Crie um documento Word com um resumo do projeto.")

        assert engine.generate.call_args.kwargs["max_tokens"] == 2048

    def test_truncated_document_is_not_created_or_offered(self, tmp_path):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "Trecho ainda incompleto do documento.",
            "tool_calls": [],
            "usage": {},
            "model": "test-model",
            "finish_reason": "length",
        }
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
        )
        agent._artifact_max_continuations = 1

        result = agent.run("Crie um documento Word sobre o OpenJarvis.")

        assert result.metadata["generation_truncated"] is True
        assert result.metadata["artifact_quality_blocked"]
        assert result.tool_results == []
        assert list(tmp_path.iterdir()) == []
        assert "Não consegui concluir todo o conteúdo" in result.content
        assert "não disponibilizei um arquivo parcial" in result.content
        assert "Versão mais curta" in result.content
        assert "Documento por partes" in result.content
        assert result.metadata["artifact_retry_suggestion"] == (
            "shorter_or_sectioned"
        )

    def test_requested_document_title_is_preserved(self, tmp_path):
        engine = _make_engine_no_tools("Texto explicando o OpenJarvis.")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
        )

        result = agent.run(
            'Crie um Word chamado teste.docx com o título "Teste de criação".'
        )

        assert result.tool_results[0].success is True
        created_path = tmp_path / result.tool_results[0].metadata["filename"]
        from docx import Document

        document = Document(created_path)
        assert document.paragraphs[0].text == "Teste de criação"

    def test_planilha_request_routes_directly_to_excel(self, tmp_path):
        engine = _make_engine_no_tools("País | Programa\nCanadá | Intercâmbio")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[CreateSpreadsheetTool([str(tmp_path)]), _WebSearchStub()],
        )

        result = agent.run("Crie uma planilha com estes dados.")

        assert "tools" not in engine.generate.call_args.kwargs
        assert result.metadata["tool_route"] == "artifact_only"
        assert result.tool_results[0].tool_name == "create_spreadsheet"
        assert result.tool_results[0].metadata["format"] == "xlsx"

    def test_explicit_web_request_forces_search_before_generation(self):
        engine = _make_engine_no_tools("Pesquisa concluída.")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub(), _WebSearchStub(), _FileReadStub()],
        )

        result = agent.run("Pesquise na internet os melhores destinos.")

        assert "tools" not in engine.generate.call_args.kwargs
        messages = engine.generate.call_args.args[0]
        assert "<web_search_results>" in messages[-1].text
        assert "Resultado confiável da pesquisa." in messages[-1].text
        assert [item.tool_name for item in result.tool_results] == ["web_search"]
        assert result.metadata["tool_route"] == "web_only"

    def test_recommendation_routes_to_research_without_model_rewriting_query(self):
        engine = _make_engine_no_tools("Três países.")
        web = _WebSearchStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])

        result = agent.run(
            "Quais são três países que você recomenda para turismo hoje?"
        )

        assert web.calls[0]["query"] == (
            "Quais são três países que você recomenda para turismo hoje?"
        )
        assert result.metadata["tool_route"] == "web_only"

    def test_low_cost_recommendation_routes_to_research(self):
        engine = _make_engine_no_tools("Uma recomendação fundamentada.")
        web = _WebSearchStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])

        result = agent.run("Qual é o melhor país para viajar gastando menos?")

        assert web.calls[0]["query"] == (
            "Qual é o melhor país para viajar gastando menos?"
        )
        assert result.metadata["tool_route"] == "web_only"

    def test_factual_requests_use_evidence_without_subject_word_lists(self):
        assert _needs_evidence_research("Qual é a dívida pública do Brasil?")
        assert _needs_evidence_research("O que é dipirona?")
        assert _needs_evidence_research("Compare estes dois tratamentos.")
        assert _needs_evidence_research("Explique a fotossíntese.")

    def test_non_factual_requests_do_not_start_web_research(self):
        assert not _needs_evidence_research("Oi, tudo bem?")
        assert not _needs_evidence_research("Resuma este texto para mim.")
        assert not _needs_evidence_research("Crie uma história de ficção curta.")
        assert not _needs_evidence_research(
            "Analise o documento anexado.", documents=[("dados.txt", "conteúdo")]
        )

    def test_sensitive_request_without_safe_evidence_keeps_limited_research(self):
        engine = _make_engine_no_tools("Tome 500 mg todos os dias.")
        web = _WebSearchStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])

        result = agent.run(
            "Quais efeitos colaterais podem ocorrer com este medicamento?"
        )

        assert "Tome 500 mg" not in result.content
        assert "profissional de saúde" in result.content
        assert engine.generate.call_count == 1
        assert result.metadata["tool_route"] == "web_only"

    def test_travel_result_that_mentions_visas_is_not_sensitive(self):
        class _TravelWebStub(_WebSearchStub):
            def execute(self, **params) -> ToolResult:
                self.calls.append(params)
                return ToolResult(
                    tool_name="web_search",
                    content=(
                        "### Turismo\nSource: https://travel.example/lista\n"
                        "Summary: Turismo cultural. Veja também regras de visto."
                    ),
                    success=True,
                )

        engine = _make_engine_no_tools("Resposta turística.")
        web = _TravelWebStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])

        result = agent.run("Quais países você recomenda para turismo hoje?")

        assert result.metadata["tool_route"] == "web_only"
        assert engine.generate.call_count == 1

    def test_sensitive_request_without_safe_evidence_does_not_prescribe(self):
        engine = _make_engine_no_tools("Tome 500 mg todos os dias.")
        web = _WebSearchStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])

        result = agent.run(
            "Quais efeitos colaterais podem ocorrer com este medicamento?"
        )

        assert "500 mg" not in result.content
        assert "profissional de saúde" in result.content

    def test_link_request_recovers_recorded_sources_without_new_search(self):
        engine = _make_engine_no_tools("Não deve gerar uma nova resposta.")
        web = _WebSearchStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])
        context = AgentContext()
        context.conversation.add(
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Resposta anterior.\n<!-- openjarvis-research-sources:"
                    '[{"title":"Fonte usada","url":"https://example.gov.br/dado",'
                    '"label":"domínio governamental provável"}] -->'
                ),
            )
        )

        result = agent.run(
            "Quais foram as fontes usadas? Mostre os links.", context=context
        )

        assert web.calls == []
        assert "[Fonte usada](https://example.gov.br/dado)" in result.content
        assert engine.generate.call_count == 0

    def test_plain_source_question_recovers_recorded_sources(self):
        engine = _make_engine_no_tools("Não deve gerar uma nova resposta.")
        web = _WebSearchStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])
        context = AgentContext()
        context.conversation.add(
            Message(
                role=Role.ASSISTANT,
                content=(
                    "Resposta anterior.\n<!-- openjarvis-research-sources:"
                    '[{"title":"Fonte usada","url":"https://example.gov.br/dado",'
                    '"label":"domínio governamental provável"}] -->'
                ),
            )
        )

        result = agent.run(
            "Quais fontes você usou para essa recomendação?", context=context
        )

        assert web.calls == []
        assert "[Fonte usada](https://example.gov.br/dado)" in result.content
        assert engine.generate.call_count == 0

    def test_source_question_without_ledger_does_not_generate_sources(self):
        engine = _make_engine_no_tools("Fonte inventada.")
        web = _WebSearchStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])
        context = AgentContext()
        context.conversation.add(
            Message(role=Role.ASSISTANT, content="Resposta antiga sem registro."),
        )

        result = agent.run("Quais fontes você usou?", context=context)

        assert web.calls == []
        assert "Não há fontes registradas" in result.content
        assert engine.generate.call_count == 0

    def test_web_search_expands_contextual_country_reference(self):
        engine = _make_engine_no_tools("Pesquisa concluída.")
        web = _WebSearchStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])
        context = AgentContext()
        context.conversation.add(
            Message(
                role=Role.USER,
                content="Quais são os melhores países para intercâmbio?",
            )
        )
        context.conversation.add(
            Message(
                role=Role.ASSISTANT,
                content="Canadá, Alemanha e Irlanda são boas opções.",
            )
        )

        agent.run(
            "Pesquise na web as vantagens e desvantagens desses países.",
            context=context,
        )

        assert len(web.calls) == 1
        query = web.calls[0]["query"]
        assert query == "as vantagens e desvantagens: Canadá, Alemanha, Irlanda"
        assert "Contexto da conversa" not in query
        messages = engine.generate.call_args.args[0]
        assert all(
            message.text != "Canadá, Alemanha e Irlanda são boas opções."
            for message in messages
        )

    def test_context_countries_deduplicate_england_from_united_kingdom(self):
        countries = _countries_from_context(
            [
                Message(
                    role=Role.ASSISTANT,
                    content=(
                        "Opções: Canadá, Austrália, Reino Unido, Ingllaterra, "
                        "Coreia do Sul e Irlanda."
                    ),
                )
            ]
        )

        assert countries == [
            "Canadá",
            "Austrália",
            "Reino Unido",
            "Coreia do Sul",
            "Irlanda",
        ]

    def test_web_response_hides_links_unless_the_user_asks(self):
        results = "### Fonte oficial\nSource: https://gov.example/atual\nSummary: Dado."
        ordinary = _web_grounded_input(
            "Pesquise a informação atual.", results, show_links=False
        )
        requested = _web_grounded_input(
            "Pesquise e mostre os links.", results, show_links=True
        )

        assert _links_requested("Pesquise a informação atual.") is False
        assert _links_requested("Pesquise e mostre os links.") is True
        assert "Não mostre URLs" in ordinary
        assert "pediu explicitamente" in requested
        assert "Data atual do sistema:" in ordinary
        assert "também é aceitável" in ordinary

    def test_chat_response_requests_readable_markdown(self):
        formatted = _chat_format_instruction("Explique o resultado.")

        assert formatted.startswith("Explique o resultado.")
        assert "<chat_presentation>" in formatted
        assert "linha em branco" in formatted
        assert "Data atual do sistema:" in formatted

    def test_attachment_search_uses_document_anchors_not_old_chat_topic(self):
        engine = _make_engine_no_tools("Complemento fundamentado nos anexos.")
        web = _WebSearchStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])
        context = AgentContext()
        context.conversation.add(
            Message(
                role=Role.ASSISTANT,
                content=(
                    "A inflação brasileira entre 1994 e 2005 caiu após o "
                    "Plano Real."
                ),
            )
        )
        request = """Com base no documento anexado, pesquise na web informações
recentes que complementem estes dados.
<attached_documents>
<document name="estrategia.xlsx" type="xlsx">
Programa\tDestino
Working Holiday\tNova Zelândia
Kaplan\tCanadá
</document>
</attached_documents>"""

        agent.run(request, context=context)

        assert len(web.calls) == 1
        query = web.calls[0]["query"]
        assert "Working Holiday" in query
        assert "Nova Zelândia" in query
        assert "inflação" not in query.lower()
        messages = engine.generate.call_args.args[0]
        assert all("Plano Real" not in message.text for message in messages)
        assert "<attachment_scope>" in messages[-1].text

    def test_attachment_terms_ignore_headers_and_long_prose(self):
        terms = _attachment_research_terms(
            [
                (
                    "dados.xlsx",
                    "Programa\tDestino\tStatus\n"
                    "Working Holiday\tNova Zelândia\tA verificar\n"
                    "Este é um comentário muito longo que não deve virar "
                    "consulta de pesquisa por conter informação descritiva demais.",
                )
            ]
        )

        assert terms == ["Working Holiday", "Nova Zelândia"]

    def test_user_style_ordinal_reference_keeps_exact_list_order(self):
        context = AgentContext()
        context.conversation.add(
            Message(
                role=Role.ASSISTANT,
                content=(
                    "1. **Japão**: cultura e transporte.\n"
                    "2. **França**: museus e gastronomia.\n"
                    "3. **Espanha**: praias e cidades.\n"
                    "4. **Portugal**: história e vinhos."
                ),
            )
        )

        queries = _web_search_queries(
            (
                "Agora pesquise na web as vantagens e desvantagens desses "
                "três primeiros países."
            ),
            context.conversation.messages,
        )

        assert queries == [
            "Japão vantagens e desvantagens",
            "França vantagens e desvantagens",
            "Espanha vantagens e desvantagens",
        ]

    def test_user_style_ordinal_reference_searches_each_item_separately(self):
        class _EvidenceWebStub(_WebSearchStub):
            def execute(self, **params) -> ToolResult:
                self.calls.append(params)
                entity = params["query"].split()[0]
                return ToolResult(
                    tool_name="web_search",
                    content=(
                        f"### Dados de {entity}\n"
                        f"Source: https://gov.example/{entity.lower()}\n"
                        f"Summary: Evidência turística de {entity}."
                    ),
                    success=True,
                    metadata={
                        "sources": [
                            {
                                "title": f"Dados de {entity}",
                                "url": f"https://gov.example/{entity.lower()}",
                            }
                        ],
                        "source_profiles": [],
                        "quality": {"trusted_results": 1, "primary_candidates": 1},
                    },
                )

        engine = _make_engine_no_tools("Resposta baseada nas três evidências.")
        web = _EvidenceWebStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])
        context = AgentContext()
        context.conversation.add(
            Message(
                role=Role.ASSISTANT,
                content=(
                    "1. **Japão**: opção um.\n"
                    "2. **França**: opção dois.\n"
                    "3. **Espanha**: opção três.\n"
                    "4. **Portugal**: opção quatro."
                ),
            )
        )

        result = agent.run(
            (
                "Agora pesquise na web as vantagens e desvantagens desses "
                "três primeiros países."
            ),
            context=context,
        )

        assert [call["query"] for call in web.calls[:3]] == [
            "Japão vantagens e desvantagens",
            "França vantagens e desvantagens",
            "Espanha vantagens e desvantagens",
        ]
        assert {
            call["query"] for call in web.calls[3:]
        } == {
            "https://gov.example/japão",
            "https://gov.example/frança",
            "https://gov.example/espanha",
        }
        assert result.tool_results[0].metadata["research_queries"] == [
            "Japão vantagens e desvantagens",
            "França vantagens e desvantagens",
            "Espanha vantagens e desvantagens",
        ]
        messages = engine.generate.call_args.args[0]
        assert "<research_scope>" in messages[-1].text
        assert "Portugal" not in messages[-1].text

    def test_user_style_follow_up_keeps_prior_subject_without_extra_items(self):
        messages = [
            Message(
                role=Role.USER,
                content="Pesquise na web quais são os melhores países para turismo.",
            ),
            Message(
                role=Role.ASSISTANT,
                content=(
                    "1. **Japão**: opção um.\n"
                    "2. **França**: opção dois.\n"
                    "3. **Espanha**: opção três.\n"
                    "4. **Portugal**: opção quatro."
                ),
            ),
        ]

        queries = _web_search_queries(
            "Agora pesquise as vantagens e desvantagens desses três primeiros países.",
            messages,
        )

        assert queries == [
            "Japão vantagens e desvantagens turismo",
            "França vantagens e desvantagens turismo",
            "Espanha vantagens e desvantagens turismo",
        ]

    def test_research_plan_discards_result_that_only_matches_entity(self):
        result = ToolResult(
            tool_name="web_search",
            content=(
                "### Futebol Japão\n"
                "Source: https://news.example/japan-football\n"
                "Summary: Japão venceu uma partida de futebol.\n\n---\n\n"
                "### Turismo no Japão\n"
                "Source: https://tourism.example/japan\n"
                "Summary: Turistas encontram transporte e segurança no Japão."
            ),
            success=True,
            metadata={
                "sources": [
                    {"url": "https://news.example/japan-football"},
                    {"url": "https://tourism.example/japan"},
                ],
                "source_profiles": [
                    {"url": "https://news.example/japan-football"},
                    {"url": "https://tourism.example/japan"},
                ],
            },
        )

        filtered = _filter_research_evidence(
            result,
            "Japão vantagens e desvantagens turismo",
        )

        assert "Futebol Japão" not in filtered.content
        assert "Turismo no Japão" in filtered.content
        assert filtered.metadata["evidence_blocks_removed"] == 1

    def test_user_style_sensitive_search_opens_relevant_evidence_before_writing(self):
        class _EvidencePipelineWebStub(_WebSearchStub):
            government_url = "https://www.canada.ca/student-permit"
            commercial_url = "https://agency.example/canada-study"

            def execute(self, **params) -> ToolResult:
                self.calls.append(params)
                query = params["query"]
                if query == self.government_url:
                    return ToolResult(
                        tool_name="web_search",
                        content=(
                            "Requisitos do visto de estudante no Canadá incluem "
                            "carta de aceitação e comprovação financeira."
                        ),
                        success=True,
                    )
                if query == self.commercial_url:
                    return ToolResult(
                        tool_name="web_search",
                        content=(
                            "O visto de estudante no Canadá pode exigir documentos "
                            "de apoio conforme o curso."
                        ),
                        success=True,
                    )
                return ToolResult(
                    tool_name="web_search",
                    content=(
                        "### Governo do Canadá: permissão de estudo\n"
                        f"Source: {self.government_url}\n"
                        "Summary: Requisitos do visto de estudante no Canadá."
                        "\n\n---\n\n"
                        "### Guia comercial\n"
                        f"Source: {self.commercial_url}\n"
                        "Summary: Visto de estudante no Canadá para intercâmbio."
                    ),
                    success=True,
                    metadata={
                        "sources": [
                            {"url": self.government_url},
                            {"url": self.commercial_url},
                        ],
                        "source_profiles": [
                            {
                                "url": self.government_url,
                                "label": "instituição pública reconhecida",
                                "institutional": True,
                                "primary_candidate": True,
                            },
                            {
                                "url": self.commercial_url,
                                "label": "fonte secundária ou comercial",
                                "institutional": False,
                                "primary_candidate": False,
                            },
                        ],
                        "quality": {"sensitive_query": True},
                    },
                )

        engine = _make_engine_no_tools("Resposta fundamentada.")
        web = _EvidencePipelineWebStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])

        result = agent.run(
            "Pesquise informações atuais sobre os requisitos do visto de "
            "estudante no Canadá."
        )

        assert [call["query"] for call in web.calls] == [
            "informações atuais sobre os requisitos do visto de estudante no Canadá.",
            web.government_url,
        ]
        evidence = result.tool_results[0]
        assert evidence.metadata["quality"]["safe_source_count"] == 1
        assert evidence.metadata["quality"]["evidence_mode"] == "page_verified"
        prompt = engine.generate.call_args.args[0][-1].text
        assert "<evidence_contract>" in prompt
        assert "página aberta" in prompt
        assert "Requisitos do visto de estudante no Canadá incluem" in prompt

    def test_page_that_does_not_cover_the_query_is_not_page_verified(self):
        class _OffTopicPageWebStub(_WebSearchStub):
            source_url = "https://gov.example/japan-tourism"

            def execute(self, **params) -> ToolResult:
                self.calls.append(params)
                if params["query"] == self.source_url:
                    return ToolResult(
                        tool_name="web_search",
                        content="Resultados de futebol e calendário de partidas.",
                        success=True,
                    )
                return ToolResult(
                    tool_name="web_search",
                    content=(
                        "### Turismo no Japão\n"
                        f"Source: {self.source_url}\n"
                        "Summary: Turismo no Japão: vantagens de transporte "
                        "e segurança."
                    ),
                    success=True,
                    metadata={
                        "source_profiles": [
                            {
                                "url": self.source_url,
                                "label": "domínio governamental provável",
                                "institutional": True,
                                "primary_candidate": True,
                            }
                        ]
                    },
                )

        engine = _make_engine_no_tools("Resposta.")
        web = _OffTopicPageWebStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])

        result = agent.run("Pesquise as vantagens do turismo no Japão.")

        evidence = result.tool_results[0]
        assert evidence.metadata["quality"]["safe_source_count"] == 0
        assert evidence.metadata["evidence_records"][0]["mode"] == "summary"
        assert "Resultados de futebol" not in evidence.content
        assert "Verificação: resumo de busca" in evidence.content

    def test_sensitive_search_retries_once_for_a_safe_source(self):
        class _SafeRetryWebStub(_WebSearchStub):
            commercial_url = "https://agency.example/medicine"
            official_url = "https://gov.example/medicine"

            def execute(self, **params) -> ToolResult:
                self.calls.append(params)
                query = params["query"]
                if query == self.official_url:
                    return ToolResult(
                        tool_name="web_search",
                        content=(
                            "Dipirona é um medicamento. A orientação de uso deve "
                            "seguir a bula e um profissional de saúde."
                        ),
                        success=True,
                    )
                if "órgão regulador" in query:
                    return ToolResult(
                        tool_name="web_search",
                        content=(
                            "### Bula de dipirona\n"
                            f"Source: {self.official_url}\n"
                            "Summary: Informações da bula oficial sobre dipirona."
                        ),
                        success=True,
                        metadata={
                            "source_profiles": [
                                {
                                    "url": self.official_url,
                                    "label": "domínio governamental provável",
                                    "institutional": True,
                                    "primary_candidate": True,
                                }
                            ]
                        },
                    )
                return ToolResult(
                    tool_name="web_search",
                    content=(
                        "### Guia comercial de dipirona\n"
                        f"Source: {self.commercial_url}\n"
                        "Summary: Dipirona e informações de medicamento."
                    ),
                    success=True,
                    metadata={
                        "source_profiles": [
                            {
                                "url": self.commercial_url,
                                "label": "fonte secundária ou comercial",
                                "institutional": False,
                                "primary_candidate": False,
                            }
                        ]
                    },
                )

        engine = _make_engine_no_tools("Resposta.")
        web = _SafeRetryWebStub()
        agent = OrchestratorAgent(engine, "test-model", tools=[web])

        result = agent.run("Pesquise informações sobre dipirona.")

        evidence = result.tool_results[0]
        assert len(web.calls) == 3
        assert "órgão regulador" in web.calls[1]["query"]
        assert web.calls[2]["query"] == web.official_url
        assert evidence.metadata["secure_source_retry"] is True
        assert evidence.metadata["quality"]["safe_source_count"] == 1

    def test_web_response_removes_links_not_returned_by_the_tool(self):
        response = (
            "Veja a [fonte obtida](https://gov.example/dado) e a "
            "[fonte inventada](https://inventada.example/fato)."
        )

        assert _hide_unretrieved_links(response, {"https://gov.example/dado"}) == (
            "Veja a [fonte obtida](https://gov.example/dado) e a fonte inventada."
        )

    def test_web_response_strips_links_when_not_requested(self):
        response = (
            "Veja a [fonte oficial](https://gov.example/dado) e "
            "https://example.com/extra."
        )

        assert _hide_unrequested_links(response) == "Veja a fonte oficial e."

    def test_compound_web_and_word_routes_search_then_python(self, tmp_path):
        engine = _make_engine_no_tools("Relatório baseado na pesquisa confiável.")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[
                _CalculatorStub(),
                _WebSearchStub(),
                CreateDocumentTool([str(tmp_path)]),
            ],
        )

        result = agent.run(
            "Pesquise na internet o OpenJarvis e crie um documento Word."
        )

        assert "tools" not in engine.generate.call_args.kwargs
        messages = engine.generate.call_args.args[0]
        assert "<web_search_results>" in messages[-1].text
        assert result.metadata["tool_route"] == "web_then_artifact"
        assert [item.tool_name for item in result.tool_results] == [
            "web_search",
            "create_document",
        ]
        assert result.tool_results[-1].success is True

    def test_tool_route_is_recomputed_for_each_message(self, tmp_path):
        engine = _make_engine_no_tools("Conteúdo do documento.")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_WebSearchStub(), CreateDocumentTool([str(tmp_path)])],
        )

        first = agent.run("Crie um documento Word simples.")
        assert "tools" not in engine.generate.call_args.kwargs
        assert first.metadata["tool_route"] == "artifact_only"

        second = agent.run("Pesquise na web as notícias atuais.")
        assert "tools" not in engine.generate.call_args.kwargs
        assert second.tool_results[0].tool_name == "web_search"
        assert second.metadata["tool_route"] == "web_only"

    def test_web_request_without_tool_fails_instead_of_hallucinating(self):
        engine = _make_engine_no_tools("Fonte inventada.")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
        )

        result = agent.run("Pesquise na internet o OpenJarvis.")

        engine.generate.assert_not_called()
        assert result.turns == 0
        assert result.metadata["tool_route"] == "web_search_failed"
        assert result.tool_results[0].success is False
        assert "não está disponível" in result.content

    def test_max_turns_error_is_never_written_to_word(self, tmp_path):
        engine = _make_engine_no_tools("Maximum turns reached without a final answer.")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
        )

        result = agent.run("Crie um documento Word simples.")

        assert result.tool_results == []
        assert "limite interno" in result.metadata["artifact_quality_blocked"]
        assert list(tmp_path.iterdir()) == []

    def test_attached_word_request_uses_one_analysis_pass_and_semantic_name(
        self, tmp_path
    ):
        content = (
            "As duas planilhas reúnem oportunidades de estudo e intercâmbio. "
            "Entre os principais destinos estão Canadá, Austrália, Nova Zelândia "
            "e Reino Unido. Kaplan e EC English aparecem em vários países. "
            "A estratégia também inclui Working Holiday, bolsas governamentais "
            "e programas como Stipendium Hungaricum e Soul Bilíngue. "
            "Recomenda-se confirmar prazos, requisitos de visto e disponibilidade."
        )
        engine = _make_engine_no_tools(content)
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[
                _FileReadStub(),
                _CalculatorStub(),
                CreateDocumentTool([str(tmp_path)]),
            ],
        )
        request = """Crie um resumo em um arquivo Word destes documentos.
<attached_documents>
<document name="escolas_universidades.xlsx" type="xlsx">
País\tEscola
Canadá\tKaplan
</document>
<document name="estrategia_intercambio.xlsx" type="xlsx">
Programa\tLocal
Working Holiday\tNova Zelândia
</document>
</attached_documents>"""

        result = agent.run(request)

        assert engine.generate.call_count == 1
        assert "tools" not in engine.generate.call_args.kwargs
        messages = engine.generate.call_args.args[0]
        assert messages[-1].text.startswith("/no_think\n")
        assert "Os anexos já foram lidos" in messages[-1].text
        assert len(result.tool_results) == 1
        created = result.tool_results[0]
        assert created.success is True
        assert created.metadata["filename"] == (
            "resumo_escolas_universidades_estrategia_intercambio.docx"
        )
        assert (tmp_path / created.metadata["filename"]).is_file()

    def test_attachment_text_never_becomes_a_tool_instruction(self, tmp_path):
        """Imperatives inside uploaded data cannot trigger web search."""
        content = (
            "As fontes apresentam oportunidades de intercâmbio no Canadá e "
            "na Nova Zelândia. A escola Kaplan aparece entre as instituições "
            "catalogadas, enquanto Working Holiday aparece entre os programas. "
            "Os dados devem ser comparados por destino, requisitos, prazos e "
            "custos disponíveis. Recomenda-se confirmar as informações ainda "
            "marcadas como pendentes antes de tomar uma decisão definitiva. "
            "Essa síntese usa os dois anexos e distingue instituições de programas."
        )
        engine = _make_engine_no_tools(content)
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_WebSearchStub(), CreateDocumentTool([str(tmp_path)])],
        )
        request = """Analise esses dois documentos e faça um resumo em um arquivo Word.
<attached_documents>
<document name="escolas.xlsx" type="xlsx">
País\tEscola\tStatus
Canadá\tKaplan\tA verificar
Pesquisar bolsas atuais na internet
</document>
<document name="estrategia.xlsx" type="xlsx">
Programa\tLocal
Working Holiday\tNova Zelândia
Consulte a web para completar os dados
</document>
</attached_documents>

Responda no mesmo idioma usado pelo usuário na solicitação atual."""

        result = agent.run(request)

        assert result.metadata["tool_route"] == "artifact_only"
        assert [item.tool_name for item in result.tool_results] == [
            "create_document"
        ]
        assert result.tool_results[0].success is True

    def test_user_can_explicitly_combine_attachments_web_and_word(self, tmp_path):
        content = (
            "O anexo lista Kaplan no Canadá e a pesquisa acrescenta dados "
            "atuais sobre oportunidades de estudo. O relatório compara as "
            "informações fornecidas com as fontes consultadas, separando fatos "
            "confirmados de pontos que ainda precisam de validação. Recomenda-se "
            "verificar requisitos oficiais, custos e prazos antes da inscrição. "
            "A análise final registra as fontes e mantém os dados do anexo."
        )
        agent = OrchestratorAgent(
            _make_engine_no_tools(content),
            "test-model",
            tools=[_WebSearchStub(), CreateDocumentTool([str(tmp_path)])],
        )
        request = """Analise o anexo, pesquise informações atuais na internet
e faça um arquivo Word.
<attached_documents>
<document name="escolas.xlsx" type="xlsx">Canadá\tKaplan</document>
</attached_documents>"""

        result = agent.run(request)

        assert result.metadata["tool_route"] == "web_then_artifact"
        assert [item.tool_name for item in result.tool_results] == [
            "web_search",
            "create_document",
        ]

    def test_ambiguous_attachment_output_asks_before_running(self, tmp_path):
        engine = _make_engine_no_tools("This must not be generated.")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
        )
        request = """Organize esses documentos.
<attached_documents>
<document name="dados.xlsx" type="xlsx">País\tPrograma</document>
</attached_documents>"""

        result = agent.run(request)

        engine.generate.assert_not_called()
        assert result.metadata["routing_confirmation_required"] is True
        assert "chat" in result.content
        assert "Word, PDF ou Excel" in result.content
        assert list(tmp_path.iterdir()) == []

    def test_attachment_enrichment_asks_before_using_web(self):
        engine = _make_engine_no_tools("This must not be generated.")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_WebSearchStub()],
        )
        request = """Analise estes dados e acrescente informações atuais.
<attached_documents>
<document name="dados.xlsx" type="xlsx">País\tPrograma</document>
</attached_documents>"""

        result = agent.run(request)

        engine.generate.assert_not_called()
        assert result.metadata["routing_confirmation_required"] is True
        assert "somente os anexos" in result.content
        assert "fontes na internet" in result.content
        assert result.tool_results == []

    def test_casual_word_delivery_request_is_understood(self, tmp_path):
        agent = OrchestratorAgent(
            _make_engine_no_tools("Conteúdo final para o documento."),
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
        )

        result = agent.run("Agora eu quero isso em Word.")

        assert result.metadata["tool_route"] == "artifact_only"
        assert result.tool_results[0].tool_name == "create_document"
        assert result.tool_results[0].success is True

    def test_operational_statuses_are_emitted_for_file_workflow(self, tmp_path):
        bus = EventBus(record_history=True)
        agent = OrchestratorAgent(
            _make_engine_no_tools("Conteúdo final do documento."),
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
            bus=bus,
        )

        agent.run("Faça um documento Word chamado teste_status.docx.")

        stages = [
            event.data["stage"]
            for event in bus.history
            if event.event_type == EventType.AGENT_STATUS
        ]
        assert stages == [
            "routing",
            "generation",
            "artifact_creation",
            "validation",
        ]

    def test_explicit_web_negation_keeps_attachment_workflow_offline(self, tmp_path):
        engine = _make_engine_no_tools(
            "Resumo consolidado dos anexos com informações suficientes "
            "para produzir um documento válido e verificável. O material "
            "compara os programas disponíveis no Canadá, incluindo a Bolsa "
            "Horizonte, e registra as informações "
            "presentes na planilha e organiza os resultados em seções claras. "
            "Os dados ausentes permanecem identificados como não informados, "
            "sem consulta externa e sem invenção de conteúdo adicional."
        )
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_WebSearchStub(), CreateDocumentTool([str(tmp_path)])],
        )
        request = """Analise os anexos e crie um arquivo Word,
sem pesquisar na internet.
<attached_documents>
<document name="dados.xlsx" type="xlsx">
País\tPrograma
Canadá\tBolsa Horizonte
</document>
</attached_documents>"""

        result = agent.run(request)

        assert result.metadata["tool_route"] == "artifact_only"
        assert [item.tool_name for item in result.tool_results] == [
            "create_document"
        ]

    def test_web_negation_overrides_document_section_named_web_research(
        self, tmp_path
    ):
        engine = _make_engine_no_tools(
            "# OpenJarvis personalizado\n\n"
            "## 1. Objetivo\n\nDescrição suficientemente detalhada do projeto.\n\n"
            "## 10. Pesquisa na internet\n\nDescrição da funcionalidade existente.\n\n"
            "## 12. Conclusão\n\nSíntese final suficientemente detalhada."
        )
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_WebSearchStub(), CreateDocumentTool([str(tmp_path)])],
        )
        request = """Crie um documento Word sobre o OpenJarvis com estas seções:
1. Objetivo
10. Pesquisa na internet
12. Conclusão
Não pesquise na internet e use somente o contexto desta conversa."""

        result = agent.run(request)

        assert result.metadata["tool_route"] == "artifact_only"
        assert [item.tool_name for item in result.tool_results] == [
            "create_document"
        ]

    def test_explicit_artifact_negation_answers_only_in_chat(self, tmp_path):
        engine = _make_engine_no_tools("Resumo exibido somente no chat.")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
        )

        result = agent.run(
            "Analise os dados. Não crie arquivo; responda apenas no chat."
        )

        assert result.content == "Resumo exibido somente no chat."
        assert result.tool_results == []
        assert list(tmp_path.iterdir()) == []

    def test_artifact_negation_with_determiners_has_priority(self, tmp_path):
        engine = _make_engine_no_tools("Resumo exibido somente no chat.")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
        )
        request = """Analise estes dois documentos e me apresente um resumo no chat.
Não crie nenhum arquivo e não pesquise na internet.
<attached_documents>
<document name="dados.xlsx" type="xlsx">País\tPrograma</document>
</attached_documents>"""

        result = agent.run(request)

        assert result.content == "Resumo exibido somente no chat."
        assert result.metadata["tool_route"] == "attached_content_only"
        assert result.tool_results == []
        assert list(tmp_path.iterdir()) == []

    def test_negated_format_can_be_replaced_by_later_positive_format(self, tmp_path):
        agent = OrchestratorAgent(
            _make_engine_no_tools("Conteúdo final do documento."),
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
        )

        result = agent.run("Não crie nenhum PDF; crie um arquivo Word.")

        assert result.metadata["tool_route"] == "artifact_only"
        assert result.tool_results[0].success is True
        assert result.tool_results[0].metadata["format"] == "docx"

    def test_attached_word_request_blocks_file_for_invalid_analysis(self, tmp_path):
        engine = _make_engine_no_tools(
            "Os arquivos mencionados não foram encontrados no seu sistema."
        )
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
        )
        request = """Crie um resumo em um arquivo Word.
<attached_documents>
<document name="dados.xlsx" type="xlsx">País\tPrograma</document>
</attached_documents>"""

        result = agent.run(request)

        assert result.tool_results == []
        assert "não encont" in result.metadata["artifact_quality_blocked"]
        assert list(tmp_path.iterdir()) == []

    def test_attached_word_request_blocks_unrelated_historical_answer(self, tmp_path):
        engine = _make_engine_no_tools(
            "A inflação brasileira entre 1994 e 2005 caiu após o Plano Real. "
            "O período teve mudanças econômicas importantes, inflação e juros. "
            "Esses dados históricos devem ser comparados com cautela antes de "
            "tomar decisões futuras sobre a economia brasileira. A análise "
            "macroeconômica considera os efeitos da política monetária, da "
            "taxa de câmbio e dos índices de preços no comportamento dos "
            "agentes econômicos durante a estabilização."
        )
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
        )
        request = """Crie um resumo em Word a partir do anexo.
<attached_documents>
<document name="intercambio.xlsx" type="xlsx">
País\tPrograma
Canadá\tWorking Holiday
</document>
</attached_documents>"""

        result = agent.run(request)

        assert result.tool_results == []
        assert (
            "temas específicos dos anexos"
            in result.metadata["artifact_quality_blocked"]
        )
        assert list(tmp_path.iterdir()) == []

    def test_empty_word_response_without_attachments_has_accurate_error(self, tmp_path):
        engine = _make_engine_no_tools("")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[CreateDocumentTool([str(tmp_path)])],
        )

        result = agent.run("Crie um documento Word usando a pesquisa anterior.")

        assert result.tool_results == []
        assert result.metadata["artifact_quality_blocked"] == (
            "o modelo não produziu conteúdo"
        )
        assert "histórico da conversa continua disponível" in result.content
        assert "anexos continuam" not in result.content.lower()
        assert list(tmp_path.iterdir()) == []

    def test_agent_id(self):
        engine = _make_engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model")
        assert agent.agent_id == "orchestrator"

    def test_capability_policy_is_forwarded_to_tool_executor(self):
        policy = MagicMock()
        agent = OrchestratorAgent(
            _make_engine_no_tools(),
            "test-model",
            capability_policy=policy,
        )
        assert agent._executor._capability_policy is policy

    def test_no_tools_single_turn(self):
        engine = _make_engine_no_tools("Hello!")
        agent = OrchestratorAgent(engine, "test-model")
        result = agent.run("Hello")
        assert result.content == "Hello!"
        assert result.turns == 1
        assert result.tool_results == []

    def test_single_tool_call(self):
        engine = _make_engine_with_tool_call()
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
        )
        result = agent.run("What is 2+2?")
        assert result.content == "The answer is 4."
        assert result.turns == 2
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "calculator"
        assert result.tool_results[0].content == "4"

    def test_multiple_tool_calls_same_turn(self):
        engine = _make_engine_multi_tool()
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub(), _ThinkStub()],
        )
        result = agent.run("Think and calculate.")
        assert result.content == "Done."
        assert result.turns == 2
        assert len(result.tool_results) == 2

    def test_with_context_conversation(self):
        engine = _make_engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model")
        conv = Conversation()
        conv.add(Message(role=Role.SYSTEM, content="Be helpful."))
        ctx = AgentContext(conversation=conv)
        agent.run("Hi", context=ctx)
        call_args = engine.generate.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0].role == Role.SYSTEM

    def test_tools_passed_to_engine(self):
        engine = _make_engine_no_tools()
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
        )
        agent.run("Hello")
        call_kwargs = engine.generate.call_args[1]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1

    def test_no_tools_no_tools_kwarg(self):
        engine = _make_engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model")
        agent.run("Hello")
        call_kwargs = engine.generate.call_args[1]
        assert "tools" not in call_kwargs

    def test_max_turns_exceeded(self):
        """When the engine keeps returning tool calls, stop after max_turns."""
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "calculator", "arguments": '{"expression":"1+1"}'}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "test-model",
            "finish_reason": "tool_calls",
        }
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
            max_turns=3,
        )
        result = agent.run("Loop forever")
        assert result.turns == 3
        assert result.metadata.get("max_turns_exceeded") is True

    def test_unknown_tool_in_response(self):
        engine = _make_engine_with_tool_call(
            tool_name="unknown_tool",
            arguments="{}",
            final_content="Handled.",
        )
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
        )
        result = agent.run("Use unknown tool")
        assert result.content == "Handled."
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False

    def test_temperature_passthrough(self):
        engine = _make_engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model", temperature=0.1)
        agent.run("Hello")
        call_kwargs = engine.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.1

    def test_max_tokens_passthrough(self):
        engine = _make_engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model", max_tokens=256)
        agent.run("Hello")
        call_kwargs = engine.generate.call_args[1]
        assert call_kwargs["max_tokens"] == 256

    def test_event_bus_agent_events(self):
        bus = EventBus(record_history=True)
        engine = _make_engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model", bus=bus)
        agent.run("Hello")
        event_types = [e.event_type for e in bus.history]
        assert EventType.AGENT_TURN_START in event_types
        assert EventType.AGENT_TURN_END in event_types

    def test_event_bus_inference_events(self):
        """INFERENCE_START/END are now published by InstrumentedEngine,
        not by agents directly.  Agent tests verify agent-level events."""
        bus = EventBus(record_history=True)
        engine = _make_engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model", bus=bus)
        agent.run("Hello")
        event_types = [e.event_type for e in bus.history]
        assert EventType.AGENT_TURN_START in event_types
        assert EventType.AGENT_TURN_END in event_types

    def test_event_bus_tool_events(self):
        bus = EventBus(record_history=True)
        engine = _make_engine_with_tool_call()
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
            bus=bus,
        )
        agent.run("Calc 2+2")
        event_types = [e.event_type for e in bus.history]
        assert EventType.TOOL_CALL_START in event_types
        assert EventType.TOOL_CALL_END in event_types

    def test_messages_accumulate(self):
        """After tool call, messages include assistant + tool messages."""
        engine = _make_engine_with_tool_call()
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
        )
        agent.run("What is 2+2?")
        # Second call should include accumulated messages
        second_call = engine.generate.call_args_list[1]
        messages = second_call[0][0]
        roles = [m.role for m in messages]
        assert Role.ASSISTANT in roles
        assert Role.TOOL in roles

    def test_tool_message_has_tool_call_id(self):
        engine = _make_engine_with_tool_call(tool_call_id="abc123")
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
        )
        agent.run("What is 2+2?")
        second_call = engine.generate.call_args_list[1]
        messages = second_call[0][0]
        tool_msgs = [m for m in messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "abc123"

    def test_no_bus_works(self):
        engine = _make_engine_with_tool_call()
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
        )
        result = agent.run("What is 2+2?")
        assert result.content == "The answer is 4."

    def test_empty_tools_list(self):
        engine = _make_engine_no_tools()
        agent = OrchestratorAgent(engine, "test-model", tools=[])
        result = agent.run("Hello")
        assert result.content == "Final answer."

    def test_three_turn_conversation(self):
        """Engine calls a tool twice before answering."""
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.side_effect = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "calculator",
                        "arguments": '{"expression":"2+2"}',
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
                "model": "test-model",
                "finish_reason": "tool_calls",
            },
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "calculator",
                        "arguments": '{"expression":"4*3"}',
                    }
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 3,
                    "total_tokens": 18,
                },
                "model": "test-model",
                "finish_reason": "tool_calls",
            },
            {
                "content": "2+2=4, 4*3=12",
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 5,
                    "total_tokens": 30,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
        ]
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
        )
        result = agent.run("Calculate")
        assert result.turns == 3
        assert len(result.tool_results) == 2
        assert result.tool_results[0].content == "4"
        assert result.tool_results[1].content == "12"

    def test_tool_result_latency_tracked(self):
        engine = _make_engine_with_tool_call()
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
        )
        result = agent.run("What is 2+2?")
        assert result.tool_results[0].latency_seconds >= 0

    def test_max_turns_1(self):
        """With max_turns=1 and a tool call, should stop after 1 turn."""
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "",
            "tool_calls": [
                {"id": "c1", "name": "calculator", "arguments": '{"expression":"1"}'}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "test-model",
            "finish_reason": "tool_calls",
        }
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
            max_turns=1,
        )
        result = agent.run("Calc")
        assert result.turns == 1
        assert result.metadata.get("max_turns_exceeded") is True

    def test_agent_turn_end_data_no_tools(self):
        bus = EventBus(record_history=True)
        engine = _make_engine_no_tools("reply")
        agent = OrchestratorAgent(engine, "test-model", bus=bus)
        agent.run("Hi")
        end = [e for e in bus.history if e.event_type == EventType.AGENT_TURN_END][0]
        assert end.data["turns"] == 1
        assert end.data["content_length"] == 5

    def test_result_content_on_max_turns(self):
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "partial",
            "tool_calls": [
                {"id": "c1", "name": "calculator", "arguments": '{"expression":"1"}'}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "test-model",
            "finish_reason": "tool_calls",
        }
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
            max_turns=2,
        )
        result = agent.run("Calc")
        # Should use the partial content if available
        assert result.content == "partial"


class TestOrchestratorStructuredMode:
    """Tests for the structured (THOUGHT/TOOL/INPUT/FINAL_ANSWER) mode."""

    def test_structured_mode_final_answer(self):
        """Structured mode should parse FINAL_ANSWER: correctly."""
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "THOUGHT: Easy question.\nFINAL_ANSWER: Paris",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "test-model",
            "finish_reason": "stop",
        }
        agent = OrchestratorAgent(
            engine,
            "test-model",
            mode="structured",
        )
        result = agent.run("What is the capital of France?")
        assert result.content == "Paris"
        assert result.turns == 1
        assert result.tool_results == []

    def test_structured_mode_tool_call(self):
        """Parse TOOL:/INPUT:, execute tool, return final answer."""
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.side_effect = [
            {
                "content": (
                    "THOUGHT: Need to calculate.\n"
                    "TOOL: calculator\n"
                    'INPUT: {"expression":"2+2"}'
                ),
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
            {
                "content": ("THOUGHT: Got 4.\nFINAL_ANSWER: The answer is 4."),
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
        ]
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
            mode="structured",
        )
        result = agent.run("What is 2+2?")
        assert result.content == "The answer is 4."
        assert result.turns == 2
        assert len(result.tool_results) == 1
        assert result.tool_results[0].tool_name == "calculator"
        assert result.tool_results[0].content == "4"

    def test_structured_mode_enriched_descriptions(self):
        """Structured mode system prompt should contain enriched tool descriptions."""
        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.return_value = {
            "content": "FINAL_ANSWER: ok",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "test-model",
            "finish_reason": "stop",
        }
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
            mode="structured",
        )
        agent.run("Hello")
        call_args = engine.generate.call_args
        messages = call_args[0][0]
        system_msg = messages[0].content
        assert "### calculator" in system_msg
        assert "expression" in system_msg


class TestOrchestratorParallelTools:
    """Tests for parallel tool execution."""

    def test_parallel_tool_execution(self):
        """Multiple tool calls execute in parallel and return in correct order."""

        class _SlowTool(BaseTool):
            tool_id = "slow"

            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(
                    name="slow",
                    description="Slow tool.",
                    parameters={
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                    },
                )

            def execute(self, **params) -> ToolResult:
                time.sleep(0.1)  # Simulate slow operation
                return ToolResult(
                    tool_name="slow",
                    content=f"result_{params.get('id', '')}",
                    success=True,
                )

        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.side_effect = [
            {
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "slow", "arguments": '{"id":"1"}'},
                    {"id": "c2", "name": "slow", "arguments": '{"id":"2"}'},
                    {"id": "c3", "name": "slow", "arguments": '{"id":"3"}'},
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                },
                "model": "test-model",
                "finish_reason": "tool_calls",
            },
            {
                "content": "All done.",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 3,
                    "total_tokens": 23,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
        ]

        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_SlowTool()],
            parallel_tools=True,
        )
        t0 = time.time()
        result = agent.run("Do things")
        elapsed = time.time() - t0

        assert result.content == "All done."
        assert len(result.tool_results) == 3
        # Results should be in original order
        assert result.tool_results[0].content == "result_1"
        assert result.tool_results[1].content == "result_2"
        assert result.tool_results[2].content == "result_3"
        # Should be parallel — 3 tools at 0.1s each should take < 0.25s, not 0.3s+
        assert elapsed < 0.25

    def test_sequential_tool_execution(self):
        """parallel_tools=False runs tools sequentially."""
        engine = _make_engine_multi_tool()
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub(), _ThinkStub()],
            parallel_tools=False,
        )
        result = agent.run("Do things")
        assert result.content == "Done."
        assert len(result.tool_results) == 2

    def test_single_tool_call_no_parallel(self):
        """Single tool call should not use parallel path even if parallel_tools=True."""
        engine = _make_engine_with_tool_call()
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
            parallel_tools=True,
        )
        result = agent.run("What is 2+2?")
        assert result.content == "The answer is 4."


class TestOrchestratorGovernanceHook:
    """Tests for the optional before_tool_call governance hook."""

    def test_no_hook_runs_tool_normally(self):
        """Without a hook, tool execution is unchanged."""
        engine = _make_engine_with_tool_call()
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
        )
        result = agent.run("What is 2+2?")
        assert result.content == "The answer is 4."
        assert result.tool_results[0].success is True
        assert result.tool_results[0].content == "4"

    def test_hook_returning_true_allows_tool(self):
        """Hook returning True allows the tool to execute normally."""
        engine = _make_engine_with_tool_call()
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
            before_tool_call=lambda name, args: True,
        )
        result = agent.run("What is 2+2?")
        assert result.tool_results[0].success is True
        assert result.tool_results[0].content == "4"

    def test_hook_returning_false_blocks_tool(self):
        """Hook returning False injects a denial; the tool is never executed."""
        executed = []

        class _TrackedCalculator(_CalculatorStub):
            def execute(self, **params) -> ToolResult:
                executed.append(params)
                return super().execute(**params)

        engine = _make_engine_with_tool_call()
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_TrackedCalculator()],
            before_tool_call=lambda name, args: False,
        )
        result = agent.run("What is 2+2?")
        assert executed == [], "tool must not be called when hook returns False"
        assert result.tool_results[0].success is False
        assert "[Governance]" in result.tool_results[0].content

    def test_hook_receives_correct_tool_name_and_args(self):
        """The hook is called with the exact tool name and parsed argument dict."""
        calls: list[tuple] = []

        def _hook(name: str, args: dict) -> bool:
            calls.append((name, args))
            return True

        engine = _make_engine_with_tool_call(
            tool_name="calculator",
            arguments='{"expression":"3*7"}',
        )
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_CalculatorStub()],
            before_tool_call=_hook,
        )
        agent.run("Calculate")
        assert len(calls) == 1
        assert calls[0][0] == "calculator"
        assert calls[0][1] == {"expression": "3*7"}

    def test_hook_denial_in_parallel_mode(self):
        """Hook returning False blocks tools in the parallel execution path."""
        executed = []

        class _TrackedCalculator(_CalculatorStub):
            def execute(self, **params) -> ToolResult:
                executed.append(params)
                return super().execute(**params)

        engine = _make_engine_multi_tool()
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_TrackedCalculator(), _ThinkStub()],
            parallel_tools=True,
            before_tool_call=lambda name, args: False,
        )
        result = agent.run("Think and calculate.")
        assert executed == [], "no tool must execute when hook denies"
        assert len(result.tool_results) == 2
        assert all(not tr.success for tr in result.tool_results)
        assert all("[Governance]" in tr.content for tr in result.tool_results)

    def test_hook_denial_in_structured_mode(self):
        """Hook returning False blocks a tool in structured (ReAct) mode."""
        executed = []

        class _TrackedCalculator(_CalculatorStub):
            def execute(self, **params) -> ToolResult:
                executed.append(params)
                return super().execute(**params)

        engine = MagicMock()
        engine.engine_id = "mock"
        engine.generate.side_effect = [
            {
                "content": (
                    "THOUGHT: Need to calculate.\n"
                    "TOOL: calculator\n"
                    'INPUT: {"expression":"2+2"}'
                ),
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
            {
                "content": "THOUGHT: Blocked.\nFINAL_ANSWER: Could not calculate.",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
                "model": "test-model",
                "finish_reason": "stop",
            },
        ]
        agent = OrchestratorAgent(
            engine,
            "test-model",
            tools=[_TrackedCalculator()],
            mode="structured",
            before_tool_call=lambda name, args: False,
        )
        result = agent.run("What is 2+2?")
        assert executed == [], "tool must not execute when hook denies"
        assert len(result.tool_results) == 1
        assert result.tool_results[0].success is False
        assert "[Governance]" in result.tool_results[0].content

    def test_hook_exception_fails_closed_without_aborting_agent(self):
        """A broken policy integration denies the call instead of crashing."""
        executed = []

        class _TrackedCalculator(_CalculatorStub):
            def execute(self, **params) -> ToolResult:
                executed.append(params)
                return super().execute(**params)

        def _broken_hook(name: str, args: dict) -> bool:
            raise RuntimeError("policy service unavailable")

        agent = OrchestratorAgent(
            _make_engine_with_tool_call(),
            "test-model",
            tools=[_TrackedCalculator()],
            before_tool_call=_broken_hook,
        )

        result = agent.run("What is 2+2?")

        assert result.content == "The answer is 4."
        assert executed == []
        assert result.tool_results[0].success is False
        assert "governance check failed" in result.tool_results[0].content

    def test_parallel_hook_callbacks_are_serialized(self):
        """Policy callbacks run deterministically before parallel tool work."""
        state_lock = threading.Lock()
        active_callbacks = 0
        overlapped = False
        callback_threads: list[int] = []
        caller_thread = threading.get_ident()

        def _stateful_hook(name: str, args: dict) -> bool:
            nonlocal active_callbacks, overlapped
            with state_lock:
                active_callbacks += 1
                overlapped = overlapped or active_callbacks > 1
                callback_threads.append(threading.get_ident())
            time.sleep(0.03)
            with state_lock:
                active_callbacks -= 1
            return True

        agent = OrchestratorAgent(
            _make_engine_multi_tool(),
            "test-model",
            tools=[_CalculatorStub(), _ThinkStub()],
            parallel_tools=True,
            before_tool_call=_stateful_hook,
        )

        result = agent.run("Think and calculate.")

        assert all(tool_result.success for tool_result in result.tool_results)
        assert overlapped is False
        assert callback_threads == [caller_thread, caller_thread]

    def test_parallel_hook_failure_denies_only_that_call(self):
        """One failed decision cannot abort other calls in the same batch."""

        def _partially_broken_hook(name: str, args: dict) -> bool:
            if name == "calculator":
                raise RuntimeError("policy service unavailable")
            return True

        agent = OrchestratorAgent(
            _make_engine_multi_tool(),
            "test-model",
            tools=[_CalculatorStub(), _ThinkStub()],
            parallel_tools=True,
            before_tool_call=_partially_broken_hook,
        )

        result = agent.run("Think and calculate.")

        assert result.tool_results[0].success is False
        assert "governance check failed" in result.tool_results[0].content
        assert result.tool_results[1].success is True

    def test_non_object_arguments_fail_closed(self):
        """The documented dict callback never receives a list or scalar."""
        calls = []

        def _hook(name: str, args: dict) -> bool:
            calls.append((name, args))
            return True

        agent = OrchestratorAgent(
            _make_engine_with_tool_call(arguments='["2+2"]'),
            "test-model",
            tools=[_CalculatorStub()],
            before_tool_call=_hook,
        )

        result = agent.run("What is 2+2?")

        assert calls == []
        assert result.tool_results[0].success is False
        assert "not an object" in result.tool_results[0].content
