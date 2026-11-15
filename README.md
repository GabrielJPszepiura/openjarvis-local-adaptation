# OpenJarvis

Assistente de IA local em português, com interface web, pesquisa na internet,
recuperação de fontes e criação de documentos.

Esta é uma versão adaptada do [OpenJarvis](https://github.com/open-jarvis/OpenJarvis),
projeto open source que serve como base para a arquitetura de agentes, ferramentas
e execução local. Esta versão concentra-se em uma experiência prática no Windows,
com pesquisa web contextual, fontes rastreáveis e geração de arquivos.

## O que esta versão faz

- Conversa local pela interface web.
- Pesquisa web para perguntas factuais e atuais.
- Recuperação dos links realmente usados quando o usuário pede as fontes.
- Continuidade em listas pesquisadas: depois de receber opções, o usuário pode
  pedir comparação, pontos positivos, cuidados ou uma recomendação entre elas.
- Proteção contra troca indevida de opções: uma consulta de fontes não substitui
  a lista original da conversa.
- Mudança de assunto com nova pesquisa, sem reutilizar fatos do tema anterior
  como evidência.
- Criação de documentos Word para download a partir de pesquisas e conteúdos do chat.

## Estado atual

Os fluxos principais foram testados com o modelo local Qwen3.5:2B:

- pesquisa de listas e recomendações;
- exibição posterior das fontes usadas;
- comparação entre itens apresentados anteriormente;
- mudança de assunto, como turismo para cotação do dólar;
- geração de arquivo Word com nome relacionado ao assunto.

O principal ponto de evolução é o tempo de geração do modelo local. A busca web
costuma ser rápida, mas respostas longas podem levar alguns minutos conforme o
modelo e o hardware disponíveis. A qualidade final da redação também depende do
modelo selecionado.

## Requisitos locais

- Windows 10 ou 11.
- Python e ambiente virtual do projeto configurados.
- Node.js, usado pela interface web.
- [Ollama](https://ollama.com/) em execução com um modelo local disponível.

O ambiente usado nos testes possui o modelo `qwen3.5:2b`.

## Executar no Windows

No PowerShell, a partir da pasta do projeto:

```powershell
cd C:\Users\gabri\OpenJarvis
.\.venv\Scripts\jarvis.exe gui
```

A interface ficará disponível em:

```text
http://127.0.0.1:5173
```

Para encerrar o backend corretamente:

```powershell
.\.venv\Scripts\jarvis.exe stop
```

Para reiniciar após mudanças no código:

```powershell
.\.venv\Scripts\jarvis.exe stop
.\.venv\Scripts\jarvis.exe gui
```

## Sequência curta de teste

Envie as mensagens abaixo no mesmo chat:

```text
Pesquise 3 melhores países para turismo.
```

```text
Quais foram as suas fontes?
```

```text
Agora compare os três países e explique os pontos positivos e os cuidados de cada um para uma viagem de turismo.
```

```text
Qual é a cotação do dólar atualmente?
```

Essa sequência valida pesquisa, recuperação de fontes, contexto de opções e
mudança de assunto.

## Estrutura relevante

- `src/openjarvis/agents/orchestrator.py`: roteamento de ferramentas, pesquisa
  web, contexto e regras de resposta.
- `tests/agents/test_orchestrator.py`: testes automatizados das regras de
  pesquisa e continuidade.
- `frontend/`: interface web do chat.
- `docs/`: documentação técnica herdada e referências do projeto-base.

## Licença e atribuição

O código-base OpenJarvis é distribuído sob a licença Apache 2.0. Esta versão
mantém a licença aplicável e reconhece o projeto original como base open source.
Consulte [LICENSE](LICENSE) e o [repositório original](https://github.com/open-jarvis/OpenJarvis).
