/**
 * Protect currency notation from the Markdown math parser.
 *
 * `remark-math` treats the pair of dollar signs in `R$ 5.000 ... R$ 20.000`
 * as one inline equation. Escaping only a currency marker followed by a
 * number keeps ordinary mathematical expressions (for example `$x^2$`)
 * available in chat responses.
 */
export function escapeCurrencyDollars(markdown: string): string {
  return markdown.replace(
    /\b([A-Z]{1,3})\$(?=\s*(?:\d|[.,]\d))/g,
    '$1\\$',
  );
}
