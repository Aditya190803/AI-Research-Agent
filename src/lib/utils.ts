import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { ResearchResult } from '@/types';
import { logger } from "./logger";

export function getFriendlyErrorMessage(error: string): string {
  if (error.includes("OpenRouter API error")) {
    if (error.includes("401")) return "Authentication failed. Please check the API configuration.";
    if (error.includes("429")) return "The AI service is currently busy. Please try again in a few moments.";
    if (error.includes("500")) return "The AI service encountered an error. We're retrying automatically.";
    return "We're having trouble connecting to our AI provider. Please try again later.";
  }

  if (error.includes("API rate limit reached")) {
    return "Financial data limit reached. Please wait a minute before trying again.";
  }

  if (error.includes("Invalid stock symbol")) {
    return "We couldn't find that stock symbol. Please check the spelling and try again.";
  }

  if (error.includes("Failed to fetch search results")) {
    return "We're having trouble searching the web right now. Please try again in a moment.";
  }

  if (error.includes("Failed to fetch") || error.includes("NetworkError")) {
    return "Network error. Please check your internet connection and try again.";
  }

  if (error.includes("Rate limit exceeded")) {
    return "You're doing that too fast! Please wait a moment before your next request.";
  }

  return error || "An unexpected error occurred. Please try again.";
}

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function detectQueryType(query: string): "finance" | "general" {
  const financeKeywords = [
    "stock",
    "stocks",
    "share",
    "shares",
    "price",
    "market",
    "trading",
    "invest",
    "investment",
    "dividend",
    "earnings",
    "revenue",
    "profit",
    "loss",
    "nasdaq",
    "nyse",
    "dow",
    "s&p",
    "ticker",
    "portfolio",
    "bull",
    "bear",
    "ipo",
    "etf",
    "mutual fund",
    "bond",
    "forex",
    "crypto",
    "bitcoin",
    "ethereum",
    "capitalization",
    "pe ratio",
    "eps",
    "quarterly",
    "annual report",
    "sec filing",
    "financial",
    "balance sheet",
    "income statement",
    "cash flow",
  ];

  const stockSymbolPattern = /\b[A-Z]{1,5}\b/g;
  const companyPhrases = [
    "stock of",
    "shares of",
    "price of",
    "how is .* performing",
    "should i buy",
    "should i sell",
    "market cap of",
  ];

  const lowerQuery = query.toLowerCase();

  // Check for finance keywords
  for (const keyword of financeKeywords) {
    if (lowerQuery.includes(keyword)) {
      return "finance";
    }
  }

  // Check for company phrases
  for (const phrase of companyPhrases) {
    if (new RegExp(phrase, "i").test(query)) {
      return "finance";
    }
  }

  // Check for stock symbols (uppercase letters 1-5 chars)
  const symbols = query.match(stockSymbolPattern);
  if (symbols && symbols.some((s) => s.length >= 2 && s.length <= 5)) {
    // Common words that look like tickers but aren't
    const commonWords = [
      "THE",
      "AND",
      "FOR",
      "ARE",
      "BUT",
      "NOT",
      "YOU",
      "ALL",
      "CAN",
      "HER",
      "WAS",
      "ONE",
      "OUR",
      "OUT",
      "HAS",
      "HIS",
      "HOW",
      "ITS",
      "MAY",
      "NEW",
      "NOW",
      "OLD",
      "SEE",
      "WAY",
      "WHO",
      "BOY",
      "DID",
      "GET",
      "HIM",
      "LET",
      "PUT",
      "SAY",
      "SHE",
      "TOO",
      "USE",
      "OR",
      "AND",
      "NOT",
      "IF",
      "OF",
      "TO",
      "IN",
      "IS",
      "IT",
      "AS",
      "AT",
      "BY",
      "BE",
      "ON",
      "SO",
      "WE",
      "AN",
      "DO",
      "NO",
      "UP",
      "MY",
      "GO",
      "ME",
      "HE",
    ];
    const potentialTickers = symbols.filter((s) => !commonWords.includes(s));
    if (potentialTickers.length > 0) {
      return "finance";
    }
  }

  return "general";
}

export function extractStockSymbol(query: string): string | null {
  const symbols = extractStockSymbols(query);
  return symbols.length > 0 ? symbols[0] : null;
}

export function extractStockSymbols(query: string): string[] {
  // Common stock symbols pattern
  const symbolPattern = /\b([A-Z]{1,5})\b/g;
  const matches = query.match(symbolPattern);

  if (matches) {
    const commonWords = [
      "THE",
      "AND",
      "FOR",
      "ARE",
      "BUT",
      "NOT",
      "YOU",
      "ALL",
      "CAN",
      "HER",
      "WAS",
      "ONE",
      "OUR",
      "OUT",
      "HAS",
      "HIS",
      "HOW",
      "ITS",
      "MAY",
      "NEW",
      "NOW",
      "OLD",
      "SEE",
      "WAY",
      "WHO",
      "BOY",
      "DID",
      "GET",
      "HIM",
      "LET",
      "PUT",
      "SAY",
      "SHE",
      "TOO",
      "USE",
      "AI",
      "OR",
      "AND",
      "NOT",
      "IF",
      "OF",
      "TO",
      "IN",
      "IS",
      "IT",
      "AS",
      "AT",
      "BY",
      "BE",
      "ON",
      "SO",
      "WE",
      "AN",
      "DO",
      "NO",
      "UP",
      "MY",
      "GO",
      "ME",
      "HE",
    ];
    
    const symbols: string[] = [];
    for (const match of matches) {
      if (!commonWords.includes(match) && match.length >= 2 && !symbols.includes(match)) {
        symbols.push(match);
      }
    }
    return symbols;
  }

  return [];
}

export function formatNumber(num: number): string {
  if (num >= 1e12) {
    return (num / 1e12).toFixed(2) + "T";
  }
  if (num >= 1e9) {
    return (num / 1e9).toFixed(2) + "B";
  }
  if (num >= 1e6) {
    return (num / 1e6).toFixed(2) + "M";
  }
  if (num >= 1e3) {
    return (num / 1e3).toFixed(2) + "K";
  }
  return num.toFixed(2);
}

export function generateResearchId(): string {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < 8; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

const STORAGE_KEY = 'ai-research-agent-history';

export function saveToLocalStorage(results: ResearchResult[]): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(results));
  } catch (e) {
    logger.error('Failed to save to localStorage:', e);
  }
}

export function loadFromLocalStorage(): ResearchResult[] {
  if (typeof window === 'undefined') return [];
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    const parsed = JSON.parse(stored);
    // Convert timestamp strings back to Date objects and migrate financeData
    return parsed.map((r: ResearchResult) => {
      let financeData = r.financeData;
      if (financeData && !Array.isArray(financeData)) {
        financeData = [financeData];
      }
      
      return {
        ...r,
        financeData,
        timestamp: new Date(r.timestamp)
      };
    });
  } catch (e) {
    logger.error('Failed to load from localStorage:', e);
    return [];
  }
}

export function getResearchById(id: string): ResearchResult | null {
  const history = loadFromLocalStorage();
  return history.find(r => r.id === id) || null;
}

export function debounce<T extends (...args: never[]) => unknown>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}
