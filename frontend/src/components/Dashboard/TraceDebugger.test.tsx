import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { StepDetail } from './TraceDebugger';

describe('dashboard trace details', () => {
  it('renders an incomplete legacy step without crashing', () => {
    const html = renderToStaticMarkup(
      <StepDetail
        index={0}
        step={{ step_type: null, duration_ms: null, data: null }}
      />,
    );

    expect(html).toContain('unknown');
    expect(html).toContain('0ms');
    expect(html).not.toContain('NaN');
  });
});
