/**
 * Protect currency notation from the Markdown math parser.
 *
 * `remark-math` treats the pair of dollar signs in `R$ 5.000 ... R$ 20.000`
 * as one inline equation. Escaping only a currency marker followed by a
 * number keeps ordinary mathematical expressions (for example `$x^2$`)
 * available in chat responses.
 */
export function escapeCurrencyDollars(markdown: string): string {
  // A monetary value may be written as R$ 10, US$10, or simply $10.  Escape
  // the dollar sign whenever a numeric amount follows it; `$x^2$` remains a
  // math expression because it does not start with a number.
  return markdown.replace(/(?<!\\)\$(?=\s*(?:\d|[.,]\d))/g, '\\$');
}
