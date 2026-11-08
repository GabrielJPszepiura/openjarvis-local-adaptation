from fastapi import FastAPI
from fastapi.testclient import TestClient

from openjarvis.server.generated_files_router import router


def test_download_generated_file(monkeypatch, tmp_path):
    output = tmp_path / "generated"
    output.mkdir()
    artifact = output / "relatorio.docx"
    artifact.write_bytes(b"docx-test")
    config = tmp_path / "config.toml"
    config.write_text(
        '[security]\nallowed_dirs = ["' + output.as_posix() + '"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENJARVIS_CONFIG", str(config))

    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/v1/generated-files/relatorio.docx")

    assert response.status_code == 200
    assert response.content == b"docx-test"
    assert 'attachment; filename="relatorio.docx"' in response.headers[
        "content-disposition"
    ]


def test_download_rejects_unsupported_extension(monkeypatch, tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("[security]\nallowed_dirs = []\n", encoding="utf-8")
    monkeypatch.setenv("OPENJARVIS_CONFIG", str(config))

    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/v1/generated-files/secrets.txt")

    assert response.status_code == 400
