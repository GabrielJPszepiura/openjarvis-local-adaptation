// Modified by Gabriel_JP for the OpenJarvis local adaptation (2026-11-15).
import { Brain, CircleDot, Wrench } from 'lucide-react';
import { useAppStore } from '../../lib/store';

export type AssistantPresenceState = 'ready' | 'thinking' | 'working';

interface PresenceView {
  state: AssistantPresenceState;
  label: string;
  detail: string;
}

/**
 * Small visual bridge between the agent runtime and the future 3D avatar.
 * It observes existing frontend state only and never influences agent choices.
 */
function resolvePresence(
  isStreaming: boolean,
  activeToolCalls: unknown[],
  latestEventType?: string,
): PresenceView {
  if (!isStreaming) {
    return {
      state: 'ready',
      label: 'Assistente pronto',
      detail: 'Aguardando sua próxima solicitação',
    };
  }

  const runningTool = activeToolCalls.length > 0 || latestEventType === 'tool_call_start';
  if (runningTool) {
    return {
      state: 'working',
      label: 'Executando tarefa',
      detail: 'Usando uma ferramenta solicitada pelo agente',
    };
  }

  return {
    state: 'thinking',
    label: 'Pensando',
    detail: 'Analisando sua solicitação',
  };
}

const styles: Record<AssistantPresenceState, { ring: string; icon: string }> = {
  ready: { ring: 'border-cyan-400/35 bg-cyan-400/10', icon: 'text-cyan-300' },
  thinking: { ring: 'border-violet-400/45 bg-violet-400/10', icon: 'text-violet-300' },
  working: { ring: 'border-amber-400/45 bg-amber-400/10', icon: 'text-amber-300' },
};

/**
 * Lightweight placeholder for the Avatar 3D Conversacional. A future
 * Three.js/Babylon renderer can replace its inner visual without coupling to
 * chat or to the agent runtime.
 */
export function AssistantPresence() {
  const streamState = useAppStore((s) => s.streamState);
  const agentEvents = useAppStore((s) => s.agentEvents);
  const latestEventType = agentEvents[agentEvents.length - 1]?.type;
  const presence = resolvePresence(
    streamState.isStreaming,
    streamState.activeToolCalls,
    latestEventType,
  );
  const style = styles[presence.state];
  const Icon = presence.state === 'working'
    ? Wrench
    : presence.state === 'thinking'
      ? Brain
      : CircleDot;

  return (
    <aside
      aria-live="polite"
      aria-label={`Estado do assistente: ${presence.label}`}
      className="z-10 flex shrink-0 items-center gap-3 rounded-2xl border px-3 py-2 shadow-lg backdrop-blur-md"
      style={{ background: 'color-mix(in srgb, var(--color-surface) 82%, transparent)' }}
    >
      <div className={`relative flex size-9 items-center justify-center rounded-full border ${style.ring}`}>
        {presence.state !== 'ready' && (
          <span className={`absolute inset-0 rounded-full border ${style.ring} animate-ping`} aria-hidden="true" />
        )}
        <Icon className={`size-4 ${style.icon}`} aria-hidden="true" />
      </div>
      <div className="min-w-0">
        <p className="text-xs font-medium" style={{ color: 'var(--color-text)' }}>{presence.label}</p>
        <p className="max-w-52 truncate text-[11px]" style={{ color: 'var(--color-text-muted)' }}>{presence.detail}</p>
      </div>
    </aside>
  );
}
