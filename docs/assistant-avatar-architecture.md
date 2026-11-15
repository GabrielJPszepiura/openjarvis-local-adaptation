# Arquitetura do Avatar 3D Conversacional

Este documento registra a direção inicial para adicionar uma experiência de
assistente pessoal ao OpenJarvis. A funcionalidade será adicional: ela não
substitui a interface de chat, os agentes nem o mecanismo de inferência.

## O orquestrador

O agente `orchestrator` recebe o pedido do usuário e executa um ciclo limitado:

1. monta a conversa e a lista de ferramentas permitidas;
2. pede uma resposta ao modelo de IA;
3. se o modelo solicitar ferramentas, verifica a política de autorização;
4. executa as ferramentas, adiciona os resultados à conversa e repete;
5. quando não há nova ferramenta, entrega a resposta final.

Guardas de segurança impedem chamadas não autorizadas, ciclos repetitivos e
execução além do limite de turnos. O avatar não participa dessas decisões.

## Integração proposta

O núcleo já possui um `EventBus` local. A interface do avatar deve observar
eventos existentes e refletir estados visuais, sem acoplar a renderização ao
orquestrador:

| Evento do núcleo | Estado do avatar |
| --- | --- |
| `AGENT_TURN_START` | pensando |
| `INFERENCE_START` | pensando |
| `TOOL_CALL_START` | executando |
| `TOOL_CALL_END` | pensando |
| `AGENT_TURN_END` | pronto |
| TTS iniciado | falando |
| Microfone em gravação | ouvindo |

O frontend receberá esses estados por uma API/evento próprio, com dados mínimos
e sem incluir raciocínio interno, conteúdo sensível de ferramentas ou chaves.

## Voz e animação

O OpenJarvis já separa captura de voz, transcrição e TTS. A primeira versão do
avatar usará os estados `ouvindo`, `pensando`, `falando`, `executando` e
`pronto`. Lip-sync detalhado será uma fase posterior: o TTS deverá expor
temporização ou amplitude de áudio, e o avatar consumirá somente esses sinais.

## Licença e referência técnica

O MARK JARVIS é uma referência para comportamento visual (visemas, olhar,
expressões e cartões), mas seu código e assets não serão copiados para este
projeto. O OpenJarvis é Apache-2.0; a implementação aqui deve ser original e
usar dependências e assets com licenças compatíveis.

## Primeira implementação

1. criar um contrato de estados de presença no frontend;
2. exibir um painel de assistente não-3D para validar os eventos;
3. conectar microfone e TTS aos estados;
4. trocar o painel por um renderizador 3D em uma etapa separada.
