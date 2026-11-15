import { useState, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';
import { Check, Copy, Download, FileText } from 'lucide-react';
import { AudioPlayer } from './AudioPlayer';
import { ResearchTimeline } from './ResearchTimeline';
import { rehypeCitations } from '../../lib/rehype-citations';
import { XRayFooter } from './XRayFooter';
import { SpeakMessageButton } from './SpeakMessageButton';
import type { ChatMessage } from '../../types';
import { stripThinkTags } from '../../lib/message-text';
import { escapeCurrencyDollars } from '../../lib/markdown-currency';
import { downloadGeneratedFile } from '../../lib/api';

interface Props {
  message: ChatMessage;
  isLive?: boolean;
}

function getTextContent(node: any): string {
  if (typeof node === 'string' || typeof node === 'number') {
    return String(node);
  }
  if (Array.isArray(node)) {
    return node.map(getTextContent).join('');
  }
  if (node?.props?.children) {
    return getTextContent(node.props.children);
  }
  return '';
}

function CodeBlockPre({ children, ...props }: any) {
  const [copied, setCopied] = useState(false);
  const codeElement = Array.isArray(children) ? children[0] : children;
  const className = codeElement?.props?.className || '';
  const match = /language-([\w-]+)/.exec(className);
  const lang = match ? match[1] : '';
  const code = getTextContent(codeElement?.props?.children).replace(/\n$/, '');

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className="code-block-wrapper relative my-3"
      style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden' }}
    >
      <div
        className="flex items-center justify-between px-4 py-1.5 text-xs"
        style={{ background: 'var(--color-bg-tertiary)', color: 'var(--color-text-tertiary)' }}
      >
        <span className="font-mono">{lang || 'code'}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 px-2 py-0.5 rounded transition-colors cursor-pointer"
          style={{ color: 'var(--color-text-tertiary)' }}
          onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--color-text-secondary)')}
          onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--color-text-tertiary)')}
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre {...props} style={{ margin: 0, borderRadius: 0 }}>
        {children}
      </pre>
    </div>
  );
}

function CopyMessageButton({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className="p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity cursor-pointer"
      style={{ color: 'var(--color-text-tertiary)' }}
      title="Copy message"
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  );
}

export function MessageBubble({ message, isLive = false }: Props) {
  const isUser = message.role === 'user';

  const cleanContent = useMemo(() => stripThinkTags(message.content), [message.content]);
  const markdownContent = useMemo(
    () => escapeCurrencyDollars(cleanContent),
    [cleanContent],
  );

  // Build a ref→source lookup once per render. Memoized so the rehype plugin
  // identity stays stable until the source list actually changes.
  const sourcesMap = useMemo(() => {
    const m = new Map<number, NonNullable<ChatMessage['researchSources']>[number]>();
    for (const s of message.researchSources ?? []) {
      if (typeof s.ref === 'number') m.set(s.ref, s);
    }
    return m;
  }, [message.researchSources]);

  const rehypePlugins = useMemo(() => {
    const base: any[] = [[rehypeHighlight, { detect: true }], rehypeKatex];
    if (sourcesMap.size > 0) base.push([rehypeCitations, { sources: sourcesMap }]);
    return base;
  }, [sourcesMap]);

  if (isUser) {
    return (
      <div className="flex justify-end mb-4">
        <div
          className="max-w-[85%] px-4 py-2.5 text-sm leading-relaxed"
          style={{
            background: 'var(--color-user-bubble)',
            color: 'var(--color-user-bubble-text)',
            borderRadius: 'var(--radius-xl) var(--radius-xl) var(--radius-sm) var(--radius-xl)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {message.attachments && message.attachments.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1.5">
              {message.attachments.map((attachment, index) => (
                <span
                  key={`${attachment.name}-${index}`}
                  className="inline-flex max-w-full items-center gap-1 rounded-md px-2 py-1 text-[11px]"
                  style={{ background: 'color-mix(in srgb, var(--color-user-bubble-text) 10%, transparent)' }}
                  title={attachment.name}
                >
                  <FileText size={12} />
                  <span className="max-w-48 truncate">{attachment.name}</span>
                </span>
              ))}
            </div>
          )}
          <div>{message.content}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="group mb-6">
      {/* Deep Research timeline (steps + status) */}
      {(message.isResearch || (message.researchTraces && message.researchTraces.length > 0)) && (
        <ResearchTimeline
          traces={message.researchTraces ?? []}
          isLive={isLive}
          hasContent={cleanContent.length > 0}
        />
      )}

      {/* Audio player (e.g. morning digest) */}
      {message.audio?.url && <AudioPlayer src={message.audio.url} />}

      {/* Assistant message */}
      {cleanContent && (
        <div className="prose chat-response max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={rehypePlugins}
            components={{
              pre: CodeBlockPre,
            }}
          >
            {markdownContent}
          </ReactMarkdown>
        </div>
      )}

      {message.generatedFiles && message.generatedFiles.length > 0 && (
        <div className="mt-3 flex flex-col gap-2">
          {message.generatedFiles.map((file) => (
            <button
              key={file.download_url}
              type="button"
              onClick={() => void downloadGeneratedFile(file)}
              className="flex w-full max-w-md items-center gap-3 rounded-xl border px-4 py-3 text-left transition-colors cursor-pointer"
              style={{
                background: 'var(--color-bg-secondary)',
                borderColor: 'var(--color-border)',
                color: 'var(--color-text-primary)',
              }}
              title={file.filename}
            >
              <FileText size={20} style={{ color: 'var(--color-accent)' }} />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium">{file.filename}</span>
                <span className="block text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
                  {file.format.toUpperCase()} · {Math.max(1, Math.round(file.size_bytes / 1024))} KB
                </span>
              </span>
              <Download size={18} style={{ color: 'var(--color-accent)' }} />
            </button>
          ))}
        </div>
      )}

      {/* Footer: copy + read aloud + x-ray */}
      <div className="flex items-center gap-2 mt-1.5">
        <CopyMessageButton content={cleanContent} />
        {!isUser && !isLive && (
          <SpeakMessageButton messageId={message.id} content={cleanContent} />
        )}
      </div>
      <XRayFooter
        usage={message.usage}
        telemetry={message.telemetry}
        isResearch={message.isResearch}
      />
    </div>
  );
}
