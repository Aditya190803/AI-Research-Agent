import { ResearchStrategy, ClarificationQuestion, SearchResult, FinanceData } from "@/types";

export interface ExploreResponse {
  assessment: string;
  keyAreas: string[];
  needsClarification: boolean;
  clarificationQuestions: ClarificationQuestion[];
}

export interface ResearchApiResponse {
  summary: string;
  keyFindings: string[];
  detailedAnalysis: string;
  searchResults: SearchResult[];
  newsResults: SearchResult[];
  financeData: FinanceData[];
  type: "finance" | "general";
}

export interface ResearchClient {
  explore: (query: string, context?: string, signal?: AbortSignal) => Promise<ExploreResponse>;
  generateStrategy: (query: string, context?: string, signal?: AbortSignal) => Promise<ResearchStrategy>;
  executeResearch: (strategy: ResearchStrategy, signal?: AbortSignal) => Promise<ResearchApiResponse>;
}

export const researchClient: ResearchClient = {
  async explore(query, context, signal) {
    const response = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "explore", query, context }),
      signal,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to explore query");
    }

    return response.json();
  },

  async generateStrategy(query, context, signal) {
    const response = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "strategy", query, context }),
      signal,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to generate strategy");
    }

    return response.json();
  },

  async executeResearch(strategy, signal) {
    const response = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "execute",
        query: strategy.optimizedQuery,
        subQueries: strategy.subQueries,
        stockSymbols: strategy.stockSymbols,
      }),
      signal,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to execute research");
    }

    return response.json();
  },
};
