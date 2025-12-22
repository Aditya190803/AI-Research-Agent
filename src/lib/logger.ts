const isDev = process.env.NODE_ENV === 'development';

export const logger = {
  info: (message: string, ...args: unknown[]) => {
    if (isDev) {
      console.log(`[INFO] ${message}`, ...args);
    }
  },
  error: (message: string, error?: unknown, ...args: unknown[]) => {
    // In production, you might want to send this to Sentry or another service
    console.error(`[ERROR] ${message}`, error, ...args);
  },
  warn: (message: string, ...args: unknown[]) => {
    if (isDev) {
      console.warn(`[WARN] ${message}`, ...args);
    }
  },
};
