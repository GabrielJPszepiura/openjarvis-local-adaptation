// Modified by Gabriel_JP for the OpenJarvis local adaptation (2026-11-15).
import { ChatArea } from '../components/Chat/ChatArea';
import { SystemPanel } from '../components/Chat/SystemPanel';
import { useAppStore } from '../lib/store';

export function ChatPage() {
  const systemPanelOpen = useAppStore((s) => s.systemPanelOpen);

  return (
    <div className="relative flex h-full overflow-hidden">
      <div className="flex-1 min-w-0">
        <ChatArea />
      </div>
      <aside
        aria-hidden={!systemPanelOpen}
        className={`h-full shrink-0 overflow-hidden transition-all duration-200 ease-in-out ${
          systemPanelOpen ? 'w-[280px] visible' : 'w-0 invisible pointer-events-none'
        }`}
      >
        <SystemPanel />
      </aside>
    </div>
  );
}
