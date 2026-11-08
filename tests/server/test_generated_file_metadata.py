from types import SimpleNamespace

from openjarvis.server.routes import _generated_files_from_agent_result


def _result(*tool_results):
    return SimpleNamespace(tool_results=list(tool_results))


def test_generated_file_comes_from_direct_tool_result():
    tool_result = SimpleNamespace(
        tool_name="create_document",
        success=True,
        metadata={
            "filename": "relatorio.docx",
            "format": "docx",
            "size_bytes": 37043,
        },
    )

    files = _generated_files_from_agent_result(_result(tool_result))

    assert len(files) == 1
    assert files[0].filename == "relatorio.docx"
    assert files[0].download_url == "/v1/generated-files/relatorio.docx"


def test_generated_file_falls_back_to_successful_trace_step():
    step = SimpleNamespace(
        step_type="tool_call",
        input={"tool": "create_document"},
        output={"success": True},
        metadata={
            "filename": "relatorio final.docx",
            "format": "docx",
            "size_bytes": 42,
        },
    )
    trace = SimpleNamespace(steps=[step])

    files = _generated_files_from_agent_result(_result(), trace)

    assert len(files) == 1
    assert files[0].filename == "relatorio final.docx"
    assert files[0].download_url == "/v1/generated-files/relatorio%20final.docx"


def test_model_claim_without_successful_tool_has_no_download():
    failed_step = SimpleNamespace(
        step_type="tool_call",
        input={"tool": "create_document"},
        output={"success": False},
        metadata={"filename": "nao-existe.docx", "format": "docx"},
    )

    files = _generated_files_from_agent_result(
        _result(), SimpleNamespace(steps=[failed_step])
    )

    assert files == []

