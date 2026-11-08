/** Text helpers shared between the message renderer and voice output. */

/** Remove <think> reasoning blocks so they are neither shown nor spoken. */
export function stripThinkTags(text: string): string {
  let cleaned = text.replace(/<think>[\s\S]*?<\/think>\s*/gi, '');
  cleaned = cleaned.replace(/^[\s\S]*?<\/think>\s*/i, '');
  // Provenance stays in the stored assistant turn so a later "mostre as
  // fontes" request can retrieve the exact sources used, but must never be
  // rendered or spoken to the user.
  cleaned = cleaned.replace(
    /<!--\s*openjarvis-research-sources:[\s\S]*?-->/gi,
    '',
  );
  return cleaned.trim();
}
