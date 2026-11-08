import { renderToStaticMarkup } from 'react-dom/server';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import { describe, expect, it } from 'vitest';
import { escapeCurrencyDollars } from './markdown-currency';

function renderMath(markdown: string) {
  return renderToStaticMarkup(
    <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
      {markdown}
    </ReactMarkdown>,
  );
}

describe('patched KaTeX with the Markdown plugins', () => {
  it('renders inline and display equations', () => {
    const html = renderMath('Inline $x^2$\n\n$$\n\\frac{1}{2}\n$$');
    expect(html).toContain('class="katex"');
    expect(html).toContain('class="katex-display"');
    expect(html).toContain('<math');
    expect(html).not.toContain('katex-error');
  });

  it('does not enable untrusted HTML commands', () => {
    const html = renderMath('$\\href{javascript:alert(1)}{click}$');
    expect(html).not.toContain('href="javascript:');
    expect(html).not.toContain('<a ');
  });

  it('renders currency markers literally without disabling equations', () => {
    const html = renderMath(
      escapeCurrencyDollars('De R$ 5.000 a US$20.000 e $ 50; fórmula $x^2$.'),
    );

    expect(html).toContain('R$ 5.000');
    expect(html).toContain('US$20.000');
    expect(html).toContain('$ 50');
    expect(html).toContain('class="katex"');
  });
});
