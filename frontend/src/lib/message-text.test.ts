import { describe, expect, it } from 'vitest';
import { stripThinkTags } from './message-text';

describe('stripThinkTags', () => {
  it('keeps research markers in stored messages but hides them from chat text', () => {
    const visible = stripThinkTags(
      'Resposta útil.\n<!-- openjarvis-research-context:{"kind":"paises"} -->',
    );

    expect(visible).toBe('Resposta útil.');
  });
});
