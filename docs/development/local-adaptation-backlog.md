# Backlog da adaptação local

Pendências abertas registradas em 14 de novembro de 2026 para a versão local
do OpenJarvis. Este arquivo não descreve funcionalidades já concluídas; ele
serve como guia de testes, melhorias e decisões antes da publicação.

## Concluído e validado manualmente

- **Limpar todos os dados**: a remoção de conversas e rastros vinculados foi
  verificada na interface e no backend.
- **Aviso de lista vazia**: quando não há conversas, a interface informa isso
  sem abrir uma confirmação destrutiva indevida.
- **Configuração de contexto entre chats**: criada desligada por padrão, com
  ativação explícita do usuário.

## Prioridade: validação de dados e memória

- Validar a retenção de 30 dias dos logs do servidor.

## Prioridade: escolhas, contexto e pesquisa

- Fazer perguntas vagas como “qual dos cinco?” recuperarem o assunto correto
  mesmo que outros assuntos tenham aparecido no mesmo chat; quando houver
  ambiguidade real, pedir esclarecimento.
- Impedir que uma resposta derivada de escolha ou comparação sobrescreva o
  conjunto original de opções bloqueadas. Somente a resposta que cria uma
  lista deve registrar ou atualizar esse conjunto.
- Garantir que listas respeitem quantidade e tipo pedidos: cinco notebooks
  devem ser cinco notebooks concretos, e não uma categoria genérica como
  quinto item.
- Tornar a pesquisa menos literal: reunir evidências de mais de uma fonte e
  compor a recomendação, sem exigir que uma única página traga exatamente o
  mesmo número solicitado pelo usuário.
- Reduzir o tempo de resposta: objetivo futuro de aproximadamente **um minuto
  por solicitação**, identificando o custo de busca, abertura de páginas,
  pesquisa por item e geração local. Também estabelecer tratamento para
  evidência insuficiente, sem entrar em loop.
- Durante os testes de escolha contextual, medir se a pesquisa complementar
  trouxe informação realmente útil para a decisão ou apenas aumentou o tempo
  e introduziu ruído. Registrar tempo total, consultas feitas, escolha final
  e valor concreto acrescentado pela evidência nova.
- Avaliar posteriormente dois níveis de escolha: **rápida**, para “qual você
  escolheria e por quê?” com justificativa curta baseada no material já
  disponível; e **aprofundada**, quando o usuário pedir detalhamento,
  comparação por critérios ou justificativa extensa, podendo pesquisar cada
  opção original sob um limite global de duração.

## Próxima bateria: arquivos e análise de anexos

- Criar pelo chat uma planilha Excel financeira de teste: dados mensais,
  totais, fórmulas, formatação e download.
- Anexar essa mesma planilha em um chat novo e validar a análise dos dados:
  totais, maior despesa, saldo e tendências identificadas a partir do arquivo.
- Com a planilha anexada, gerar um PDF com relatório financeiro resumido;
  validar conteúdo, renderização, download e se a análise usa os dados do
  anexo, não números inventados.
- Anexar o PDF gerado em outro chat para validar leitura de PDF e, depois,
  criar outro formato de entrega com base nele (por exemplo, Word ou Excel).

## Criação de arquivos e visualizações futuras

- Criar dashboards **nativos na tela do OpenJarvis**, vinculados à análise ou
  ao projeto/conversa — não uma página HTML externa.
- Expor uma ação explícita, como “Criar dashboard desta análise”, apenas
  quando a análise contiver dados estruturados suficientes.
- Fazer a IA produzir uma especificação controlada de métricas, categorias,
  comparações, séries e fontes; a interface renderiza gráficos com componentes
  aprovados, sem executar HTML ou JavaScript livre gerado pelo modelo.
- Avaliar exportação posterior do dashboard para PDF, imagem ou HTML somente
  como formato de entrega, sem tirá-lo da experiência nativa.

## Experiência visual e interfaces futuras

- Planejar o modo visual do assistente: avatar/animação reativa durante fala,
  resposta e estados do sistema, inspirado na ideia de um Jarvis visual.
- Definir como dashboards, painel do sistema e assistente visual convivem na
  mesma interface sem sobrecarregar a tela de conversa.

## Manutenção, privacidade e publicação

- Limpar, depois dos testes, `rust/target-memory-build/` e
  `rust/target-memory-build2/`; manter `rust/target/` por enquanto para evitar
  recompilação desnecessária.
- Excluir o backup local somente após a validação completa da limpeza de
  dados; ele não deve entrar no Git.
- Antes do GitHub, revisar README, histórico de modificações, licença e a
  estratégia de marca/nome/logotipo/cores para a futura versão própria do
  projeto.
- Planejar opções explícitas de consentimento para conectar fontes de dados e
  usar contexto entre chats.
