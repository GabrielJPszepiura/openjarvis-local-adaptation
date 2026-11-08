import { afterEach, describe, expect, it } from 'vitest';
import i18n from './i18n';

describe('application translations', () => {
  afterEach(async () => {
    await i18n.changeLanguage('en-US');
  });

  it('switches between English and Brazilian Portuguese', async () => {
    await i18n.changeLanguage('en-US');
    expect(i18n.t('nav.settings')).toBe('Settings');
    expect(i18n.t('chat.morning')).toBe('Good morning');
    expect(i18n.t('chat.deleteConversationTitle')).toBe('Delete conversation?');
    expect(i18n.t('chat.deleteConversationDescription', { title: 'Project notes' }))
      .toContain('Project notes');
    expect(i18n.t('dashboard.title')).toBe('System Overview');

    await i18n.changeLanguage('pt-BR');
    expect(i18n.t('nav.settings')).toBe('Configurações');
    expect(i18n.t('chat.morning')).toBe('Bom dia');
    expect(i18n.t('chat.deleteConversationTitle')).toBe('Excluir conversa?');
    expect(i18n.t('chat.deleteConversationDescription', { title: 'Notas do projeto' }))
      .toContain('Notas do projeto');
    expect(i18n.t('dashboard.title')).toBe('Visão geral do sistema');
  });

  it('falls back to English for an unsupported language', async () => {
    await i18n.changeLanguage('es');
    expect(i18n.t('nav.settings')).toBe('Settings');
  });
});
