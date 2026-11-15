"""Conversation data-deletion coverage for the local web API."""

# Added by the OpenJarvis local adaptation (2026-11-13): deletion cascade coverage.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openjarvis.core.types import Trace
from openjarvis.server.routes import delete_all_conversation_data, delete_conversation_data
from openjarvis.traces.store import TraceStore


def test_delete_conversation_purges_only_its_server_data(tmp_path) -> None:
    store = TraceStore(tmp_path / "traces.db")
    deleted = Trace(query="delete", metadata={"conversation_id": "chat-a"})
    retained = Trace(query="retain", metadata={"conversation_id": "chat-b"})
    store.save(deleted)
    store.save(retained)
    (tmp_path / "report.docx").write_bytes(b"test")
    store.register_generated_artifacts("chat-a", ["report.docx"], trace_id=deleted.trace_id)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                trace_store=store,
                config=SimpleNamespace(
                    security=SimpleNamespace(allowed_dirs=[str(tmp_path)])
                ),
            )
        )
    )
    coroutine = delete_conversation_data("chat-a", request)
    with pytest.raises(StopIteration) as finished:
        coroutine.send(None)
    response = finished.value.value

    assert response["deleted_traces"] == 1
    assert response["deleted_files"] == 1
    assert store.get(deleted.trace_id) is None
    assert store.get(retained.trace_id) is not None
    assert not (tmp_path / "report.docx").exists()
    store.close()


def test_delete_all_conversations_purges_legacy_traces(tmp_path) -> None:
    store = TraceStore(tmp_path / "traces.db")
    legacy = Trace(query="legacy")
    linked = Trace(query="linked", metadata={"conversation_id": "chat-a"})
    store.save(legacy)
    store.save(linked)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                trace_store=store,
                memory_service=None,
                config=SimpleNamespace(security=SimpleNamespace(allowed_dirs=[str(tmp_path)])),
            )
        )
    )

    coroutine = delete_all_conversation_data(request)
    with pytest.raises(StopIteration) as finished:
        coroutine.send(None)
    response = finished.value.value

    assert response["deleted_traces"] == 2
    assert store.count() == 0
    store.close()
