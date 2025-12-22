export type ResearchPhase =
  | "idle"
  | "exploring"
  | "clarifying"
  | "confirming"
  | "researching"
  | "complete"
  | "error";

export interface Source {
  title: string;
  url: string;
  snippet: string;
}

export interface FinanceData {
  symbol: string;
  companyName: string;
  price: number;
  change: number;
  changePercent: string;
  high: number;
  low: number;
  volume: string;
  marketCap: string;
  peRatio: string;
  dividendYield: string;
  fiftyTwoWeekHigh: number;
  fiftyTwoWeekLow: number;
  description: string;
  sector: string;
  industry: string;
  exchange: string;
  currency: string;
  currencySymbol: string;
}

export interface ResearchResult {
  id: string;
  type: "finance" | "general";
  query: string;
  summary?: string;
  keyFindings?: string[];
  detailedAnalysis?: string;
  sources?: Source[];
  financeData?: FinanceData[];
  timestamp: Date;
  phase: ResearchPhase;
  initialAssessment?: string;
  strategy?: ResearchStrategy | null;
  clarificationQuestions?: ClarificationQuestion[];
  error?: string | null;
}

export interface ClarificationQuestion {
  id: string;
  question: string;
  options: string[];
}

export interface ResearchStrategy {
  scope: string;
  optimizedQuery: string;
  subQueries: string[];
  investigationPath: string[];
}

export interface SearchResult {
  title: string;
  url: string;
  content: string;
}

export interface LangSearchItem {
  name: string;
  url: string;
  snippet: string;
  summary?: string;
}
