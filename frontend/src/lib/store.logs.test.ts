import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

class MemoryStorage {
  private store = new Map<string, string>();

  getItem(key: string): string | null {
    return this.store.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.store.set(key, String(value));
  }

  removeItem(key: string): void {
    this.store.delete(key);
  }
}

let storage: MemoryStorage;

beforeEach(() => {
  vi.resetModules();
  storage = new MemoryStorage();
  (globalThis as unknown as { localStorage: MemoryStorage }).localStorage = storage;
});

afterEach(() => {
  (globalThis as unknown as { localStorage?: MemoryStorage }).localStorage = undefined;
});

describe('activity log persistence', () => {
  it('restores entries after the store is reloaded', async () => {
    const firstStore = (await import('./store')).useAppStore;
    firstStore.getState().addLogEntry({
      timestamp: 123,
      level: 'info',
      category: 'chat',
      message: 'Request sent',
    });

    vi.resetModules();
    const reloadedStore = (await import('./store')).useAppStore;

    expect(reloadedStore.getState().logEntries).toEqual([
      {
        timestamp: 123,
        level: 'info',
        category: 'chat',
        message: 'Request sent',
      },
    ]);
  });

  it('removes persisted entries when logs are cleared', async () => {
    const firstStore = (await import('./store')).useAppStore;
    firstStore.getState().addLogEntry({
      timestamp: 456,
      level: 'warn',
      category: 'server',
      message: 'Example warning',
    });
    firstStore.getState().clearLogs();

    vi.resetModules();
    const reloadedStore = (await import('./store')).useAppStore;

    expect(reloadedStore.getState().logEntries).toEqual([]);
  });
});
