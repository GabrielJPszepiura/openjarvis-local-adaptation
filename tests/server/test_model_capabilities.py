from openjarvis.server.model_capabilities import requests_file_creation


def test_detects_portuguese_file_creation_request():
    assert requests_file_creation("Crie um documento Word chamado relatorio.docx")
    assert requests_file_creation("Gere uma planilha Excel com os produtos")


def test_ignores_ordinary_creation_request():
    assert not requests_file_creation("Crie uma história sobre inteligência artificial")
    assert not requests_file_creation("Explique o que é um PDF")


def test_ignores_negated_file_creation_request():
    assert not requests_file_creation("Analise os anexos sem criar um arquivo Word")
    assert not requests_file_creation("Não crie arquivo; responda apenas no chat")
    assert not requests_file_creation("Não crie nenhum arquivo")
    assert not requests_file_creation("Não gere qualquer documento PDF")
    assert not requests_file_creation("Não quero que crie uma planilha Excel")
