export interface RateLimitInfo {
  limit: number;
  remaining: number;
  reset: number;
}

const cache = new Map<string, { count: number; reset: number }>();

export function rateLimit(ip: string, limit: number, windowMs: number): RateLimitInfo | null {
  const now = Date.now();
  const key = ip;
  const record = cache.get(key);

  if (!record || now > record.reset) {
    const newRecord = { count: 1, reset: now + windowMs };
    cache.set(key, newRecord);
    return {
      limit,
      remaining: limit - 1,
      reset: newRecord.reset,
    };
  }

  if (record.count >= limit) {
    return null;
  }

  record.count++;
  return {
    limit,
    remaining: limit - record.count,
    reset: record.reset,
  };
}

// Cleanup cache periodically
if (typeof setInterval !== 'undefined') {
  setInterval(() => {
    const now = Date.now();
    for (const [key, record] of cache.entries()) {
      if (now > record.reset) {
        cache.delete(key);
      }
    }
  }, 60000); // Every minute
}
