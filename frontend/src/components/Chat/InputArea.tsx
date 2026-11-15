import { useState, useRef, useCallback, useEffect } from 'react';
// Modified by the OpenJarvis local adaptation (2026-11-13): optional cross-chat context.
import { useTranslation } from 'react-i18next';
import { AssistantPresence } from '../Assistant/AssistantPresence';
import { FileText, FileUp, LoaderCircle, Paperclip, Search, Send, Square, X } from 'lucide-react';
import { toast } from 'sonner';
import { useAppStore, generateId } from '../../lib/store';
import { streamChat, streamResearch } from '../../lib/sse';
import { extractChatAttachments, fetchSavings, getBase } from '../../lib/api';
import { listConnectors, getSyncStatus } from '../../lib/connectors-api';
import { serializeToolCallArguments } from '../../lib/tool-call';
import { messageContentForApi, responseLanguageInstruction } from '../../lib/chat-attachments';
import {
  engineFromCompletionChunk,
  resolveChatEngine,
} from '../../lib/chat-telemetry';
import { MicButton } from './MicButton';
import { useSpeech } from '../../hooks/useSpeech';
import type {
  ChatAttachment,
  ChatMessage,
  GeneratedFile,
  MessageTelemetry,
  ResearchSearchTrace,
  ResearchSource,
  TokenUsage,
  ToolCallInfo,
} from '../../types';

const CROSS_CHAT_CONTEXT_KEY = 'openjarvis-cross-chat-context';
const MAX_CONNECTED_CHATS = 3;
const MAX_CONNECTED_MESSAGES = 4;
const MAX_CONNECTED_CHARS = 6000;

const CROSS_CHAT_STOP_WORDS = new Set([
  'agora', 'antes', 'com', 'como', 'dos', 'das', 'esse', 'esses', 'essa',
  'essas', 'entre', 'para', 'qual', 'quais', 'sobre', 'uma', 'umas', 'uns',
  'melhor', 'melhores', 'opcao', 'opcoes', 'opção', 'opções',
]);

function crossChatTerms(query: string): string[] {
  return [...new Set(
    query
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .toLocaleLowerCase()
      .match(/[a-z0-9]{4,}/g)
      ?.filter((term) => !CROSS_CHAT_STOP_WORDS.has(term)) ?? [],
  )];
}

function relevanceScore(content: string, terms: string[]): number {
  const normalized = content
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLocaleLowerCase();
  return terms.reduce((score, term) => score + (normalized.includes(term) ? 1 : 0), 0);
}

function connectedChatContext(activeConversationId: string | null, query: string): string {
  try {
    if (localStorage.getItem(CROSS_CHAT_CONTEXT_KEY) !== 'true') return '';
    const raw = localStorage.getItem('openjarvis-conversations');
    if (!raw) return '';
    const parsed = JSON.parse(raw) as {
      conversations?: Record<string, { id: string; title: string; updatedAt: number; messages?: ChatMessage[] }>;
    };
    const terms = crossChatTerms(query);
    if (terms.length === 0) return '';
    const excerpts = Object.values(parsed.conversations ?? {})
      .filter((conversation) => conversation.id !== activeConversationId)
      .map((conversation) => ({
        conversation,
        score: relevanceScore(
          `${conversation.title}\n${(conversation.messages ?? []).map((message) => message.content).join('\n')}`,
          terms,
        ),
      }))
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score || b.conversation.updatedAt - a.conversation.updatedAt)
      .slice(0, MAX_CONNECTED_CHATS)
      .map(({ conversation }) => {
        const turns = (conversation.messages ?? [])
          .map((message, index) => ({ message, index, score: relevanceScore(message.content, terms) }))
          .filter(({ score }) => score > 0)
          .sort((a, b) => b.score - a.score || b.index - a.index)
          .slice(0, MAX_CONNECTED_MESSAGES)
          .sort((a, b) => a.index - b.index)
          .map(({ message }) => `${message.role}: ${message.content}`)
          .join('\n');
        return turns ? `Chat anterior — ${conversation.title}:\n${turns}` : '';
      })
      .filter(Boolean)
      .join('\n\n');
    if (!excerpts) return '';
    return [
      'O usuário ativou explicitamente o contexto entre chats.',
      'Use os trechos abaixo apenas quando forem relevantes; não misture assuntos sem relação.',
      excerpts.slice(0, MAX_CONNECTED_CHARS),
    ].join('\n\n');
  } catch {
    return '';
  }
}

// While Deep Research is toggled on, poll connected sources for sync
// progress so we can surface "Searching over N items — sync in progress"
// next to the toggle. Polling is gated on `enabled` so toggling DR off
// stops the network chatter immediately.
function useResearchCorpusSync(enabled: boolean): {
  syncing: boolean;
  itemsSynced: number;
} {
  const [state, setState] = useState({ syncing: false, itemsSynced: 0 });

  useEffect(() => {
    if (!enabled) {
      setState({ syncing: false, itemsSynced: 0 });
      return;
    }
    let cancelled = false;

    const poll = async () => {
      try {
        const list = await listConnectors();
        const connected = list.filter((c) => c.connected);
        if (connected.length === 0) {
          if (!cancelled) setState({ syncing: false, itemsSynced: 0 });
          return;
        }
        const results = await Promise.all(
          connected.map(async (c) => {
            try {
              return await getSyncStatus(c.connector_id);
            } catch {
              return null;
            }
          }),
        );
        let syncing = false;
        let itemsSynced = 0;
        for (const r of results) {
          if (!r) continue;
          if (r.state === 'syncing') syncing = true;
          itemsSynced += r.items_synced ?? 0;
        }
        if (!cancelled) setState({ syncing, itemsSynced });
      } catch {
        // Network blip — leave previous state intact.
      }
    };

    poll();
    const interval = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [enabled]);

  return state;
}

export function InputArea() {
  const { t, i18n } = useTranslation();
  const [input, setInput] = useState('');
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [attachmentLoading, setAttachmentLoading] = useState(false);
  const [isDraggingFiles, setIsDraggingFiles] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const dragDepthRef = useRef(0);

  const activeId = useAppStore((s) => s.activeId);
  const selectedModel = useAppStore((s) => s.selectedModel);
  const streamState = useAppStore((s) => s.streamState);
  const messages = useAppStore((s) => s.messages);
  const speechEnabled = useAppStore((s) => s.settings.speechEnabled);
  const maxTokens = useAppStore((s) => s.settings.maxTokens);
  const temperature = useAppStore((s) => s.settings.temperature);
  const createConversation = useAppStore((s) => s.createConversation);
  const addMessage = useAppStore((s) => s.addMessage);
  const updateLastAssistant = useAppStore((s) => s.updateLastAssistant);
  const setStreamState = useAppStore((s) => s.setStreamState);
  const resetStream = useAppStore((s) => s.resetStream);
  const modelLoading = useAppStore((s) => s.modelLoading);
  const deepResearch = useAppStore((s) => s.deepResearch);
  const setDeepResearch = useAppStore((s) => s.setDeepResearch);
  const corpusSync = useResearchCorpusSync(deepResearch);
  const isCurrentChatStreaming = streamState.isStreaming && streamState.conversationId === activeId;

  const {
    state: speechState,
    error: speechError,
    available: speechAvailable,
    startRecording,
    stopRecording,
  } = useSpeech();

  // Abort in-flight stream when the user switches models mid-generation.
  // This prevents errors from trying to continue a stream with a stale model.
  const prevModelRef = useRef(selectedModel);
  useEffect(() => {
    if (prevModelRef.current !== selectedModel && streamState.isStreaming) {
      abortRef.current?.abort();
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      resetStream();
      abortRef.current = null;
    }
    prevModelRef.current = selectedModel;
  }, [selectedModel, streamState.isStreaming, resetStream]);

  const micDisabled = !speechEnabled || !speechAvailable || streamState.isStreaming;
  const micReason: 'not-enabled' | 'no-backend' | 'streaming' | undefined =
    !speechEnabled ? 'not-enabled'
    : !speechAvailable ? 'no-backend'
    : streamState.isStreaming ? 'streaming'
    : undefined;

  useEffect(() => {
    if (speechError) {
      toast.error(speechError, { duration: 8000 });
    }
  }, [speechError]);

  const handleMicClick = useCallback(async () => {
    if (speechState === 'recording') {
      try {
        const text = await stopRecording();
        if (text) {
          setInput((prev) => (prev ? prev + ' ' + text : text));
        }
      } catch {
        // Error is captured in useSpeech
      }
    } else {
      await startRecording();
    }
  }, [speechState, startRecording, stopRecording]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }, [input]);

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    resetStream();
  }, [resetStream]);

  const sendMessage = useCallback(async () => {
    const typedContent = input.trim();
    if ((!typedContent && attachments.length === 0) || streamState.isStreaming || attachmentLoading) return;
    const content = typedContent || t('chat.analyzeAttachments');
    if (!selectedModel) {
      toast.error('Pick a model first (⌘K)');
      return;
    }

    const sentAttachments = attachments;
    setInput('');
    setAttachments([]);

    let convId = activeId;
    if (!convId) {
      convId = createConversation(selectedModel);
    }

    const userMsg: ChatMessage = {
      id: generateId(),
      role: 'user',
      content,
      timestamp: Date.now(),
      attachments: sentAttachments.length ? sentAttachments : undefined,
    };
    addMessage(convId, userMsg);

    // Keep model output aligned with the language selected in the interface.
    // This is separate from UI translation: local models otherwise infer the
    // response language from document contents, which may be English.
    const responseLanguage = i18n.resolvedLanguage || i18n.language;
    const languageInstruction = responseLanguageInstruction(responseLanguage);
    const currentMessages = useAppStore.getState().messages;
    const crossChatContext = connectedChatContext(convId, content);
    const apiMessages = [
      { role: 'system', content: languageInstruction },
      ...(crossChatContext ? [{ role: 'system', content: crossChatContext }] : []),
      ...currentMessages.map((m, index) => ({
        role: m.role,
        content: index === currentMessages.length - 1 && m.role === 'user'
          ? `${messageContentForApi(m, responseLanguage)}\n\n${languageInstruction}`
          : messageContentForApi(m, responseLanguage),
      })),
    ];

    const assistantMsg: ChatMessage = {
      id: generateId(),
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      isResearch: deepResearch || undefined,
    };
    addMessage(convId, assistantMsg);

    // Start streaming
    const startTime = Date.now();
    const timer = setInterval(() => {
      setStreamState({ elapsedMs: Date.now() - startTime });
    }, 100);
    timerRef.current = timer;

    const controller = new AbortController();
    abortRef.current = controller;

    let accumulatedContent = '';
    let usage: TokenUsage | undefined;
    let complexity: { score: number; tier: string; suggested_max_tokens: number } | undefined;
    let routedEngine: string | undefined;
    let effectiveModel = selectedModel;
    const toolCalls: ToolCallInfo[] = [];
    const researchTraces: ResearchSearchTrace[] = [];
    const researchSourcesByRef = new Map<number, ResearchSource>();
    const generatedFiles: GeneratedFile[] = [];
    const flushSources = () =>
      Array.from(researchSourcesByRef.values()).sort((a, b) => a.ref - b.ref);
    let lastFlush = 0;
    let ttftMs: number | undefined;

    setStreamState({
      conversationId: convId,
      isStreaming: true,
      phase: deepResearch ? t('chat.statusWeb') : t('chat.statusRouting'),
      effectiveModel: selectedModel,
      elapsedMs: 0,
      activeToolCalls: [],
      content: '',
    });
    useAppStore.getState().addLogEntry({
      timestamp: Date.now(),
      level: 'info',
      category: 'chat',
      message: deepResearch
        ? `Research: "${content.slice(0, 80)}${content.length > 80 ? '...' : ''}"`
        : `Request: "${content.slice(0, 80)}${content.length > 80 ? '...' : ''}" → ${selectedModel}`,
    });

    try {
      if (deepResearch) {
        for await (const ev of streamResearch(
          `${messageContentForApi(userMsg, responseLanguage)}\n\n${languageInstruction}`,
          selectedModel,
          controller.signal,
        )) {
          if (ev.type === 'search_call') {
            const trace: ResearchSearchTrace = {
              id: generateId(),
              query: ev.arguments?.query ?? '',
              person: ev.arguments?.person,
              timeRange: ev.arguments?.time_range,
              status: 'pending',
            };
            researchTraces.push(trace);
            setStreamState({ phase: `Searching: ${trace.query}` });
            updateLastAssistant(
              convId,
              accumulatedContent,
              undefined,
              undefined,
              undefined,
              undefined,
              [...researchTraces],
              flushSources(),
            );
            useAppStore.getState().addLogEntry({
              timestamp: Date.now(),
              level: 'info',
              category: 'tool',
              message: `Search: "${trace.query}"${trace.person ? ` (person: ${trace.person})` : ''}`,
            });
          } else if (ev.type === 'search_result') {
            const pending = [...researchTraces].reverse().find((t) => t.status === 'pending');
            if (pending) {
              pending.status = 'complete';
              pending.numHits = ev.num_hits;
              pending.topTitles = ev.top_titles;
            }
            if (ev.sources) {
              for (const src of ev.sources) {
                if (src && typeof src.ref === 'number' && !researchSourcesByRef.has(src.ref)) {
                  researchSourcesByRef.set(src.ref, src);
                }
              }
            }
            updateLastAssistant(
              convId,
              accumulatedContent,
              undefined,
              undefined,
              undefined,
              undefined,
              [...researchTraces],
              flushSources(),
            );
          } else if (ev.type === 'synthesis') {
            if (!ttftMs) ttftMs = Date.now() - startTime;
            accumulatedContent += ev.text;
            setStreamState({ content: accumulatedContent, phase: '' });
            const now = Date.now();
            if (now - lastFlush >= 80) {
              updateLastAssistant(
                convId,
                accumulatedContent,
                undefined,
                undefined,
                undefined,
                undefined,
                [...researchTraces],
                flushSources(),
              );
              lastFlush = now;
            }
          } else if (ev.type === 'system_metrics') {
            // Live GPU sample — feed straight to the System panel so Power
            // (W) and Energy (kJ) tick up in real time as the agent runs.
            useAppStore.getState().setLiveEnergy({
              power_w: ev.power_w,
              energy_j: ev.energy_j,
              duration_s: ev.duration_s,
            });
          } else if (ev.type === 'error') {
            // Backend setup/worker failure (Ollama down, planner model
            // missing, KnowledgeStore locked, etc.). Without surfacing the
            // message, the user sees only the generic "No response was
            // generated" fallback and has no way to self-diagnose.
            const msg = ev.message || 'Research failed (no detail provided)';
            accumulatedContent = accumulatedContent
              ? `${accumulatedContent}\n\n**Research stopped:** ${msg}`
              : `**Research failed:** ${msg}`;
            setStreamState({ content: accumulatedContent, phase: '' });
            useAppStore.getState().addLogEntry({
              timestamp: Date.now(),
              level: 'error',
              category: 'chat',
              message: `Deep Research error: ${msg}`,
            });
            toast.error(msg, { duration: 8000 });
          } else if (ev.type === 'done') {
            if (ev.usage) {
              usage = {
                prompt_tokens: ev.usage.prompt_tokens ?? 0,
                completion_tokens: ev.usage.completion_tokens ?? 0,
                total_tokens:
                  ev.usage.total_tokens ??
                  (ev.usage.prompt_tokens ?? 0) +
                    (ev.usage.completion_tokens ?? 0),
              };
              // Optimistically roll this research turn into the session
              // counters so the Session panel updates the moment the
              // stream finishes, regardless of how /v1/savings aggregates
              // research telemetry server-side.
              useAppStore.getState().incrementSavings(usage);
            }
            // Hold the final live numbers visible for a beat so the panel
            // doesn't flash to 0 between the SSE close and the next
            // /v1/telemetry/energy poll picking up the persisted record.
            window.setTimeout(() => {
              useAppStore.getState().setLiveEnergy(null);
            }, 1500);
            break;
          }
        }
      } else {
      for await (const sseEvent of streamChat(
        {
          model: selectedModel,
          messages: apiMessages,
          stream: true,
          conversation_id: convId,
          temperature,
          max_tokens: maxTokens,
        },
        controller.signal,
      )) {
        const eventName = sseEvent.event;

        if (eventName === 'agent_turn_start') {
          // The orchestrator may publish a precise phase (routing,
          // attachments, web, artifact creation...) immediately before the
          // lower-level agent announces a new turn.  A turn-start event is
          // only a lifecycle marker, so it must not erase that more useful
          // user-facing status.
          const currentPhase = useAppStore.getState().streamState.phase;
          if (!currentPhase || currentPhase === 'Generating...' || currentPhase === 'Researching...') {
            setStreamState({ phase: t('chat.statusThinking') });
          }
        } else if (eventName === 'model_switch') {
          try {
            const data = JSON.parse(sseEvent.data) as {
              from_model?: string;
              to_model?: string;
            };
            if (data.to_model) {
              effectiveModel = data.to_model;
              setStreamState({
                effectiveModel,
                phase: t('chat.statusModelSwitch', {
                  from: data.from_model || selectedModel,
                  to: effectiveModel,
                }),
              });
              useAppStore.getState().addLogEntry({
                timestamp: Date.now(), level: 'info', category: 'chat',
                message: `Temporary model route: ${data.from_model || selectedModel} → ${effectiveModel}`,
              });
            }
          } catch {}
        } else if (eventName === 'agent_status') {
          try {
            const data = JSON.parse(sseEvent.data) as { stage?: string; message?: string };
            const stageKeys: Record<string, string> = {
              routing: 'chat.statusRouting',
              confirm: 'chat.statusConfirm',
              attachments: 'chat.statusAttachments',
              web: 'chat.statusWeb',
              generation: 'chat.statusGeneration',
              artifact_creation: 'chat.statusArtifact',
              validation: 'chat.statusValidation',
            };
            const phase = data.stage && stageKeys[data.stage]
              ? t(stageKeys[data.stage])
              : data.message;
            if (phase) setStreamState({ phase });
          } catch {}
        } else if (eventName === 'inference_start') {
          try {
            const data = JSON.parse(sseEvent.data) as { model?: string };
            if (data.model) effectiveModel = data.model;
          } catch {}
          // The orchestrator has already published the concrete operation
          // (analyzing attachments, searching the web, preparing content...).
          // Do not replace that useful status with the generic
          // "Generating the response" label when inference begins.
          const currentPhase = useAppStore.getState().streamState.phase;
          setStreamState({
            ...(currentPhase ? {} : { phase: t('chat.statusGenerating') }),
            effectiveModel,
          });
          useAppStore.getState().addLogEntry({
            timestamp: Date.now(), level: 'info', category: 'chat',
            message: `Generating with ${effectiveModel}...`,
          });
        } else if (eventName === 'tool_call_start') {
          try {
            const data = JSON.parse(sseEvent.data);
            const tc: ToolCallInfo = {
              id: generateId(),
              tool: data.tool,
              arguments: serializeToolCallArguments(data.arguments),
              status: 'running',
            };
            toolCalls.push(tc);
            const toolPhase = data.tool === 'web_search'
              ? t('chat.statusWeb')
              : data.tool === 'create_document' || data.tool === 'create_spreadsheet'
                ? t('chat.statusArtifact')
                : t('chat.statusCallingTool', { tool: data.tool });
            setStreamState({
              phase: toolPhase,
              activeToolCalls: [...toolCalls],
            });
            updateLastAssistant(convId, accumulatedContent, [...toolCalls]);
            useAppStore.getState().addLogEntry({
              timestamp: Date.now(), level: 'info', category: 'tool',
              message: `Calling ${data.tool}(${serializeToolCallArguments(data.arguments)})`,
            });
          } catch {}
        } else if (eventName === 'tool_call_end') {
          try {
            const data = JSON.parse(sseEvent.data);
            const tc = toolCalls.find(
              (t) => t.tool === data.tool && t.status === 'running',
            );
            if (tc) {
              tc.status = data.success ? 'success' : 'error';
              tc.latency = data.latency;
              tc.result = data.result;
            }
            const nextPhase = data.tool === 'web_search'
              ? t('chat.statusAnalyzingSources')
              : data.tool === 'create_document' || data.tool === 'create_spreadsheet'
                ? t('chat.statusValidation')
                : t('chat.statusGeneration');
            setStreamState({
              phase: nextPhase,
              activeToolCalls: [...toolCalls],
            });
            updateLastAssistant(convId, accumulatedContent, [...toolCalls]);
          } catch {}
        } else if (eventName === 'generated_files') {
          try {
            const data = JSON.parse(sseEvent.data) as { files?: GeneratedFile[] };
            for (const file of data.files ?? []) {
              if (!generatedFiles.some((item) => item.download_url === file.download_url)) {
                generatedFiles.push(file);
              }
            }
          } catch {}
        } else {
          try {
            const data = JSON.parse(sseEvent.data);
            const delta = data.choices?.[0]?.delta;
            if (data.usage) usage = data.usage;
            if (data.complexity) complexity = data.complexity;
            routedEngine = engineFromCompletionChunk(data) ?? routedEngine;
            if (delta?.content) {
              if (!ttftMs) ttftMs = Date.now() - startTime;
              accumulatedContent += delta.content;
              setStreamState({ content: accumulatedContent, phase: '' });

              const now = Date.now();
              if (now - lastFlush >= 80) {
                updateLastAssistant(
                  convId,
                  accumulatedContent,
                  toolCalls.length > 0 ? [...toolCalls] : undefined,
                );
                lastFlush = now;
              }
            }
            if (data.choices?.[0]?.finish_reason === 'stop') break;
          } catch {}
        }
      }
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        // User cancelled or model switch — keep whatever was accumulated
        if (!accumulatedContent) accumulatedContent = '(Generation stopped)';
      } else {
        const errMsg = err?.message || String(err);
        accumulatedContent =
          accumulatedContent || `Error: ${errMsg}`;
        useAppStore.getState().addLogEntry({
          timestamp: Date.now(), level: 'error', category: 'chat',
          message: `Stream error: ${errMsg}`,
        });
      }
      // If we tore out mid-research, make sure the live System panel
      // numbers don't get stuck on the last sample.
      useAppStore.getState().setLiveEnergy(null);
    } finally {
      if (!accumulatedContent) {
        accumulatedContent = 'No response was generated. Please try again.';
      }
      const totalMs = Date.now() - startTime;
      const appState = useAppStore.getState();
      const selectedOwner = appState.models.find((m) => m.id === selectedModel)?.owned_by;
      const engineLabel = resolveChatEngine({
        routedEngine,
        serverEngine: appState.serverInfo?.engine,
        selectedModel,
        selectedOwner,
      });
      const telemetry: MessageTelemetry = {
        engine: engineLabel,
        model_id: effectiveModel,
        total_ms: totalMs,
        ttft_ms: ttftMs,
        tokens_per_sec: usage?.completion_tokens
          ? usage.completion_tokens / (totalMs / 1000)
          : undefined,
        complexity_score: complexity?.score,
        complexity_tier: complexity?.tier,
        suggested_max_tokens: complexity?.suggested_max_tokens,
      };
      // Check if the response has digest audio available
      let audioMeta: { url: string } | undefined;
      try {
        const digestRes = await fetch(`${getBase()}/api/digest`);
        if (digestRes.ok) {
          const digest = await digestRes.json();
          if (digest.audio_available) {
            audioMeta = { url: `${getBase()}/api/digest/audio` };
          }
        }
      } catch {
        // Not a digest response or server unavailable — skip
      }

      updateLastAssistant(
        convId,
        accumulatedContent,
        toolCalls.length > 0 ? toolCalls : undefined,
        usage,
        telemetry,
        audioMeta,
        researchTraces.length > 0 ? researchTraces : undefined,
        researchSourcesByRef.size > 0 ? flushSources() : undefined,
        generatedFiles.length > 0 ? generatedFiles : undefined,
      );
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      resetStream();
      useAppStore.getState().addLogEntry({
        timestamp: Date.now(), level: 'info', category: 'chat',
        message: `Response: ${accumulatedContent.length} chars`,
      });
      abortRef.current = null;

      // Research path updates session counters optimistically from the
      // `done` event's usage payload — re-fetching here would overwrite
      // it with a potentially stale snapshot if the server's research
      // telemetry hasn't been merged into /v1/savings yet.
      if (!deepResearch) {
        fetchSavings()
          .then((data) => useAppStore.getState().setSavings(data))
          .catch(() => {});
      }
    }
  }, [
    input,
    attachments,
    attachmentLoading,
    t,
    i18n,
    activeId,
    selectedModel,
    streamState.isStreaming,
    createConversation,
    addMessage,
    updateLastAssistant,
    setStreamState,
    resetStream,
    deepResearch,
    temperature,
    maxTokens,
  ]);

  const handleFilesSelected = useCallback(async (files: File[]) => {
    if (files.length === 0) return;
    if (attachments.length + files.length > 4) {
      toast.error(t('chat.tooManyAttachments'));
      return;
    }
    setAttachmentLoading(true);
    try {
      const extracted = await extractChatAttachments(files);
      setAttachments((current) => [...current, ...extracted]);
      if (extracted.some((attachment) => attachment.truncated)) {
        toast.warning(t('chat.attachmentTruncated'));
      }
    } catch (error: any) {
      toast.error(error?.message || t('chat.attachmentError'), { duration: 8000 });
    } finally {
      setAttachmentLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }, [attachments.length, t]);

  useEffect(() => {
    const containsFiles = (event: DragEvent) =>
      Array.from(event.dataTransfer?.types ?? []).includes('Files');

    const handleDragEnter = (event: DragEvent) => {
      if (!containsFiles(event)) return;
      event.preventDefault();
      dragDepthRef.current += 1;
      setIsDraggingFiles(true);
    };
    const handleDragOver = (event: DragEvent) => {
      if (!containsFiles(event)) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
    };
    const handleDragLeave = (event: DragEvent) => {
      if (!containsFiles(event)) return;
      event.preventDefault();
      dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
      if (dragDepthRef.current === 0) setIsDraggingFiles(false);
    };
    const handleDrop = (event: DragEvent) => {
      if (!containsFiles(event)) return;
      event.preventDefault();
      dragDepthRef.current = 0;
      setIsDraggingFiles(false);
      if (streamState.isStreaming || modelLoading || attachmentLoading) return;
      void handleFilesSelected(Array.from(event.dataTransfer?.files ?? []));
    };

    window.addEventListener('dragenter', handleDragEnter);
    window.addEventListener('dragover', handleDragOver);
    window.addEventListener('dragleave', handleDragLeave);
    window.addEventListener('drop', handleDrop);
    return () => {
      window.removeEventListener('dragenter', handleDragEnter);
      window.removeEventListener('dragover', handleDragOver);
      window.removeEventListener('dragleave', handleDragLeave);
      window.removeEventListener('drop', handleDrop);
    };
  }, [attachmentLoading, handleFilesSelected, modelLoading, streamState.isStreaming]);
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div
      className="px-4 pb-4 pt-2"
      style={{ maxWidth: 'calc(var(--chat-max-width) + 20rem)', margin: '0 auto', width: '100%' }}
    >
      {isDraggingFiles && (
        <div
          className="pointer-events-none fixed inset-0 z-[100] flex items-center justify-center p-8"
          style={{ background: 'color-mix(in srgb, var(--color-bg) 78%, transparent)' }}
        >
          <div
            className="flex min-h-56 w-full max-w-2xl flex-col items-center justify-center gap-3 rounded-3xl border-2 border-dashed p-8 text-center"
            style={{
              background: 'var(--color-bg-secondary)',
              borderColor: 'var(--color-accent)',
              boxShadow: 'var(--shadow-lg)',
            }}
          >
            <FileUp size={42} style={{ color: 'var(--color-accent)' }} />
            <strong className="text-lg" style={{ color: 'var(--color-text)' }}>
              {t('chat.dropDocuments')}
            </strong>
            <span className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
              {t('chat.dropDocumentsHint')}
            </span>
          </div>
        </div>
      )}
      <div className="mb-2 flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setDeepResearch(!deepResearch)}
            disabled={streamState.isStreaming}
            aria-pressed={deepResearch}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs transition-colors cursor-pointer disabled:cursor-default disabled:opacity-50"
            style={{
              background: deepResearch ? 'var(--color-accent-subtle)' : 'transparent',
              border: `1px solid ${deepResearch ? 'var(--color-accent)' : 'var(--color-border)'}`,
              color: deepResearch ? 'var(--color-accent)' : 'var(--color-text-tertiary)',
            }}
            title={deepResearch ? t('chat.deepResearchOn') : t('chat.deepResearchOff')}
          >
            <Search size={12} />
            {t('nav.deepResearch')}
          </button>
        </div>
        {deepResearch && corpusSync.syncing && corpusSync.itemsSynced > 0 && (
          <div
            className="text-[11px] leading-snug"
            style={{ color: 'var(--color-text-tertiary)' }}
          >
            Searching over{' '}
            <span key={corpusSync.itemsSynced} className="sync-bump" style={{ color: 'var(--color-text-secondary)' }}>
              {corpusSync.itemsSynced.toLocaleString()}
            </span>{' '}
            items — sync in progress, results will improve as more data is indexed.
          </div>
        )}
      </div>
      {attachments.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-2">
          {attachments.map((attachment, index) => (
            <div
              key={`${attachment.name}-${index}`}
              className="flex max-w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-xs"
              style={{ background: 'var(--color-bg-secondary)', border: '1px solid var(--color-border)' }}
            >
              <FileText size={14} style={{ color: 'var(--color-accent)' }} />
              <span className="max-w-52 truncate" title={attachment.name}>{attachment.name}</span>
              <button
                type="button"
                onClick={() => setAttachments((current) => current.filter((_, itemIndex) => itemIndex !== index))}
                disabled={streamState.isStreaming}
                className="cursor-pointer rounded p-0.5 disabled:cursor-default disabled:opacity-50"
                title={t('chat.removeAttachment')}
                aria-label={t('chat.removeAttachment')}
              >
                <X size={13} />
              </button>
            </div>
          ))}
        </div>
      )}
      <div className="flex items-center gap-3">
        <div
          className="flex min-w-0 flex-1 items-center gap-2 rounded-2xl px-4 py-3 transition-shadow"
          style={{
            background: 'var(--color-input-bg)',
            border: '1px solid var(--color-input-border)',
            boxShadow: 'var(--shadow-sm)',
          }}
        >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".txt,.md,.csv,.pdf,.docx,.xlsx"
          className="hidden"
          onChange={(event) => handleFilesSelected(Array.from(event.target.files ?? []))}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={streamState.isStreaming || modelLoading || attachmentLoading}
          className="shrink-0 cursor-pointer rounded-xl p-2 transition-colors disabled:cursor-default disabled:opacity-40"
          style={{ color: 'var(--color-text-tertiary)' }}
          title={t('chat.attachDocuments')}
          aria-label={t('chat.attachDocuments')}
        >
          {attachmentLoading ? <LoaderCircle size={16} className="animate-spin" /> : <Paperclip size={16} />}
        </button>
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={selectedModel ? t('chat.messagePlaceholder') : t('chat.pickModelPlaceholder')}
          rows={1}
          className="flex-1 bg-transparent outline-none resize-none text-sm leading-relaxed"
          style={{ color: 'var(--color-text)', maxHeight: '200px' }}
          disabled={streamState.isStreaming || modelLoading}
        />
        {isCurrentChatStreaming ? (
          <button
            onClick={stopStreaming}
            className="p-2 rounded-xl transition-colors shrink-0 cursor-pointer"
            style={{ background: 'var(--color-error)', color: 'var(--color-on-accent)' }}
            title={t('chat.stopGenerating')}
          >
            <Square size={16} />
          </button>
        ) : (
          <div className="flex items-center gap-1">
            <MicButton
              state={speechState}
              onClick={handleMicClick}
              disabled={micDisabled}
              reason={micReason}
            />
            <button
              onClick={sendMessage}
              disabled={streamState.isStreaming || attachmentLoading || (!input.trim() && attachments.length === 0) || modelLoading || !selectedModel}
              title={selectedModel ? t('chat.sendMessage') : t('chat.pickModelPlaceholder')}
              className="p-2 rounded-xl transition-colors shrink-0 cursor-pointer disabled:opacity-30 disabled:cursor-default"
              style={{
                background: (input.trim() || attachments.length > 0) ? 'var(--color-accent)' : 'var(--color-bg-tertiary)',
                color: (input.trim() || attachments.length > 0) ? 'white' : 'var(--color-text-tertiary)',
              }}
            >
              <Send size={16} />
            </button>
          </div>
        )}
        </div>
        <AssistantPresence />
      </div>
      <div className="flex items-center justify-center mt-2 text-[11px]" style={{ color: 'var(--color-text-tertiary)' }}>
        <span>
          <kbd className="font-mono">Enter</kbd> {t('chat.enterToSend')} &middot;{' '}
          <kbd className="font-mono">Shift+Enter</kbd> {t('chat.newLine')}
        </span>
      </div>
    </div>
  );
}
