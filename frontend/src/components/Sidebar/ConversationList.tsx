import { useState } from 'react';
import { AlertTriangle, Trash2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { TFunction } from 'i18next';
import { useNavigate } from 'react-router';
import { useAppStore } from '../../lib/store';
import { Button } from '../ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog';

interface Props {
  searchQuery: string;
}

function formatRelativeTime(timestamp: number, t: TFunction, locale: string): string {
  const diff = Date.now() - timestamp;
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return t('chat.justNow');
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(timestamp).toLocaleDateString(locale);
}

export function ConversationList({ searchQuery }: Props) {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();
  const conversations = useAppStore((s) => s.conversations);
  const activeId = useAppStore((s) => s.activeId);
  const streamingConversationId = useAppStore((s) =>
    s.streamState.isStreaming ? s.streamState.conversationId : null,
  );
  const selectConversation = useAppStore((s) => s.selectConversation);
  const deleteConversation = useAppStore((s) => s.deleteConversation);
  const [pendingDeletion, setPendingDeletion] = useState<{ id: string; title: string } | null>(null);

  const filtered = searchQuery
    ? conversations.filter((c) =>
        c.title.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    : conversations;

  if (filtered.length === 0) {
    return (
      <div className="px-3 py-8 text-center text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
        {searchQuery ? t('chat.noMatching') : t('chat.noConversations')}
      </div>
    );
  }

  return (
    <>
      <div className="flex flex-col gap-0.5 py-1">
        {filtered.map((conv) => {
          const isActive = conv.id === activeId;
          const isStreaming = conv.id === streamingConversationId;
          return (
            <div
              key={conv.id}
              className="group flex items-center rounded-lg cursor-pointer transition-colors"
              style={{
                background: isActive ? 'var(--color-bg-tertiary)' : 'transparent',
              }}
              onMouseEnter={(e) => {
                if (!isActive) e.currentTarget.style.background = 'var(--color-bg-secondary)';
              }}
              onMouseLeave={(e) => {
                if (!isActive) e.currentTarget.style.background = 'transparent';
              }}
            >
              <button
                onClick={() => {
                  selectConversation(conv.id);
                  navigate('/');
                }}
                className="flex-1 text-left px-3 py-2 min-w-0 cursor-pointer"
              >
                <div
                  className="text-sm truncate"
                  style={{
                    color: isActive ? 'var(--color-text)' : 'var(--color-text-secondary)',
                    fontWeight: isActive ? 500 : 400,
                  }}
                >
                  {conv.title}
                </div>
                <div className="text-[11px] mt-0.5" style={{ color: 'var(--color-text-tertiary)' }}>
                  {formatRelativeTime(conv.updatedAt, t, i18n.language)}
                </div>
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setPendingDeletion({ id: conv.id, title: conv.title });
                }}
                disabled={isStreaming}
                className="p-1.5 mr-1 rounded opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer disabled:cursor-not-allowed disabled:opacity-30"
                style={{ color: 'var(--color-text-tertiary)' }}
                onMouseEnter={(e) => {
                  if (!isStreaming) e.currentTarget.style.color = 'var(--color-error)';
                }}
                onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--color-text-tertiary)')}
                title={
                  isStreaming
                    ? t('chat.stopBeforeDelete')
                    : t('chat.deleteConversation')
                }
              >
                <Trash2 size={14} />
              </button>
            </div>
          );
        })}
      </div>
      <Dialog
        open={pendingDeletion !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDeletion(null);
        }}
      >
        <DialogContent
          showCloseButton={false}
          className="overflow-hidden rounded-2xl p-0 sm:max-w-md"
          style={{
            background: 'var(--color-bg-secondary)',
            border: '1px solid var(--color-border)',
            color: 'var(--color-text)',
            boxShadow: '0 24px 80px rgba(0, 0, 0, 0.45)',
          }}
        >
          <div className="p-5">
            <DialogHeader className="gap-0">
              <div className="flex items-start gap-3">
                <div
                  className="flex size-10 shrink-0 items-center justify-center rounded-xl"
                  style={{
                    background: 'color-mix(in srgb, var(--color-error) 14%, transparent)',
                    color: 'var(--color-error)',
                  }}
                >
                  <AlertTriangle size={20} />
                </div>
                <div className="min-w-0 pt-0.5">
                  <DialogTitle style={{ color: 'var(--color-text)' }}>
                    {t('chat.deleteConversationTitle')}
                  </DialogTitle>
                  <DialogDescription
                    className="mt-2 leading-relaxed"
                    style={{ color: 'var(--color-text-secondary)' }}
                  >
                    {t('chat.deleteConversationDescription', { title: pendingDeletion?.title ?? '' })}
                  </DialogDescription>
                </div>
              </div>
            </DialogHeader>
          </div>
          <DialogFooter
            className="m-0 rounded-none border-t px-5 py-4"
            style={{
              background: 'var(--color-bg-tertiary)',
              borderColor: 'var(--color-border)',
            }}
          >
            <Button
              variant="outline"
              className="min-w-24"
              style={{
                background: 'var(--color-bg-secondary)',
                borderColor: 'var(--color-border)',
                color: 'var(--color-text)',
              }}
              onClick={() => setPendingDeletion(null)}
              autoFocus
            >
              {t('common.cancel')}
            </Button>
            <Button
              variant="destructive"
              className="min-w-36"
              style={{
                background: 'color-mix(in srgb, var(--color-error) 14%, transparent)',
                borderColor: 'color-mix(in srgb, var(--color-error) 28%, transparent)',
                color: 'var(--color-error)',
              }}
              onClick={() => {
                if (!pendingDeletion) return;
                deleteConversation(pendingDeletion.id);
                setPendingDeletion(null);
              }}
            >
              <Trash2 size={14} />
              {t('chat.confirmDeleteConversation')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
