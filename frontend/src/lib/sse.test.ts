import { afterEach, describe, expect, it, vi } from 'vitest';

import { streamChat } from './sse';

describe('streamChat', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('preserves a named SSE event when event and data arrive in separate reads', async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('event: generated_files\n'));
        controller.enqueue(
          encoder.encode(
            'data: {"files":[{"filename":"teste.docx","format":"docx","size_bytes":42,"download_url":"/v1/generated-files/teste.docx"}]}\n\n',
          ),
        );
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      },
    });
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(body, { status: 200 })),
    );

    const events = [];
    for await (const event of streamChat({
      model: 'test-model',
      messages: [{ role: 'user', content: 'crie um arquivo' }],
      stream: true,
    })) {
      events.push(event);
    }

    expect(events).toEqual([
      {
        event: 'generated_files',
        data: '{"files":[{"filename":"teste.docx","format":"docx","size_bytes":42,"download_url":"/v1/generated-files/teste.docx"}]}',
      },
    ]);
  });
});
