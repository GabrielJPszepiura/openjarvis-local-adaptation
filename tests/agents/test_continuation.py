"""Tests for continuation handling (Phase 14.2)."""

from __future__ import annotations

from typing import Any, Dict, List

from openjarvis.agents._stubs import AgentResult, BaseAgent


class MockEngine:
    """Mock engine that simulates finish_reason='length'."""

    def __init__(self, responses: List[Dict[str, Any]]):
        self._responses = list(responses)
        self._call_count = 0
        self.calls = []
        self.message_calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(kwargs)
        self.message_calls.append(list(messages))
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = {"content": "done", "finish_reason": "stop"}
        self._call_count += 1
        return resp

    def list_models(self):
        return ["test-model"]

    def health(self):
        return True


class ContinuationAgent(BaseAgent):
    """Agent that exercises _check_continuation."""

    agent_id = "test_continuation"

    def run(self, input, context=None, **kwargs):
        messages = self._build_messages(input, context)
        result = self._generate(messages)
        content = self._check_continuation(result, messages)
        return AgentResult(content=content, turns=1)


class TestContinuation:
    def test_no_continuation_needed(self):
        engine = MockEngine(
            [
                {"content": "Hello world", "finish_reason": "stop"},
            ]
        )
        agent = ContinuationAgent(engine, "test-model")
        result = agent.run("Hi")
        assert result.content == "Hello world"

    def test_single_continuation(self):
        engine = MockEngine(
            [
                {"content": "Part 1...", "finish_reason": "length"},
                {"content": " Part 2.", "finish_reason": "stop"},
            ]
        )
        agent = ContinuationAgent(engine, "test-model")
        result = agent.run("Hi")
        assert result.content == "Part 1... Part 2."

    def test_heading_continuation_is_not_glued_to_previous_paragraph(self):
        engine = MockEngine(
            [
                {
                    "content": "O sistema usa motores de processamento",
                    "finish_reason": "length",
                },
                {
                    "content": "## 9. Criação de arquivos\n\nNovo conteúdo.",
                    "finish_reason": "stop",
                },
            ]
        )
        agent = ContinuationAgent(engine, "test-model")

        result = agent.run("Crie um documento")

        assert result.content == (
            "O sistema usa motores de processamento\n\n"
            "## 9. Criação de arquivos\n\nNovo conteúdo."
        )

    def test_mid_sentence_continuation_prompt_requests_completion(self):
        engine = MockEngine(
            [
                {
                    "content": "O parágrafo termina durante o processamento",
                    "finish_reason": "length",
                },
                {"content": " do pedido.", "finish_reason": "stop"},
            ]
        )
        agent = ContinuationAgent(engine, "test-model")

        agent.run("Crie um documento")

        continuation_prompt = engine.message_calls[1][-1].content
        assert "meio de uma frase" in continuation_prompt
        assert "processamento" in continuation_prompt
        assert "linha em branco antes de um novo título" in continuation_prompt

    def test_silent_token_cap_triggers_continuation(self):
        """A false Ollama ``stop`` at the exact cap must not truncate files."""
        engine = MockEngine(
            [
                {
                    "content": "Part 1...",
                    "finish_reason": "stop",
                    "usage": {"completion_tokens": 128},
                },
                {
                    "content": " Part 2.",
                    "finish_reason": "stop",
                    "usage": {"completion_tokens": 12},
                },
            ]
        )
        agent = ContinuationAgent(engine, "test-model", max_tokens=128)
        result = agent.run("Hi")

        assert result.content == "Part 1... Part 2."
        assert engine._call_count == 2

    def test_silent_token_cap_respects_continuation_limit(self):
        engine = MockEngine(
            [
                {
                    "content": "A",
                    "finish_reason": "stop",
                    "usage": {"completion_tokens": 64},
                },
                {
                    "content": "B",
                    "finish_reason": "stop",
                    "usage": {"completion_tokens": 64},
                },
            ]
        )
        agent = ContinuationAgent(engine, "test-model", max_tokens=64)
        messages = agent._build_messages("Hi")
        initial = agent._generate(messages)

        outcome = agent._continue_generation(
            initial,
            messages,
            max_continuations=1,
        )

        assert outcome.content == "AB"
        assert outcome.truncated is True
        assert outcome.stop_reason == "continuation_limit"

    def test_multiple_continuations(self):
        engine = MockEngine(
            [
                {"content": "A", "finish_reason": "length"},
                {"content": "B", "finish_reason": "length"},
                {"content": "C", "finish_reason": "stop"},
            ]
        )
        agent = ContinuationAgent(engine, "test-model")
        result = agent.run("Hi")
        assert result.content == "ABC"

    def test_max_continuations_respected(self):
        engine = MockEngine(
            [
                {"content": "A", "finish_reason": "length"},
                {"content": "B", "finish_reason": "length"},
                {"content": "C", "finish_reason": "length"},  # 3rd continuation
                {"content": "D", "finish_reason": "stop"},
            ]
        )
        agent = ContinuationAgent(engine, "test-model")
        # Default max_continuations=2, so should stop after 2 continuations
        messages = agent._build_messages("Hi")
        result_dict = agent._generate(messages)
        content = agent._check_continuation(result_dict, messages, max_continuations=2)
        assert content == "ABC"  # A + B + C, but not D

    def test_empty_finish_reason(self):
        engine = MockEngine(
            [
                {"content": "Done", "finish_reason": ""},
            ]
        )
        agent = ContinuationAgent(engine, "test-model")
        result = agent.run("Hi")
        assert result.content == "Done"

    def test_reports_truncation_when_continuation_limit_is_exhausted(self):
        engine = MockEngine(
            [
                {"content": "A", "finish_reason": "length"},
                {"content": "B", "finish_reason": "length"},
            ]
        )
        agent = ContinuationAgent(engine, "test-model")
        messages = agent._build_messages("Hi")
        initial = agent._generate(messages)

        outcome = agent._continue_generation(
            initial,
            messages,
            max_continuations=1,
        )

        assert outcome.content == "AB"
        assert outcome.truncated is True
        assert outcome.stop_reason == "continuation_limit"

    def test_repeated_continuation_stops_the_loop(self):
        engine = MockEngine(
            [
                {"content": "Mesmo trecho", "finish_reason": "length"},
                {"content": "Mesmo trecho", "finish_reason": "length"},
                {"content": "não deve ser chamado", "finish_reason": "stop"},
            ]
        )
        agent = ContinuationAgent(engine, "test-model")
        messages = agent._build_messages("Hi")
        initial = agent._generate(messages)

        outcome = agent._continue_generation(initial, messages)

        assert outcome.truncated is True
        assert outcome.stop_reason == "repeated_continuation"
        assert engine._call_count == 2

    def test_continuation_uses_document_token_override(self):
        engine = MockEngine(
            [
                {"content": "A", "finish_reason": "length"},
                {"content": "B", "finish_reason": "stop"},
            ]
        )
        agent = ContinuationAgent(engine, "test-model", max_tokens=128)
        messages = agent._build_messages("Hi")
        initial = agent._generate(messages)

        outcome = agent._continue_generation(
            initial,
            messages,
            max_tokens_override=2048,
        )

        assert outcome.content == "AB"
        assert engine.calls[0]["max_tokens"] == 128
        assert engine.calls[1]["max_tokens"] == 2048
