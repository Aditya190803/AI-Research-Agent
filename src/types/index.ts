export type ResearchPhase =
  | "idle"
  | "exploring"
  | "clarifying"
  | "confirming"
  | "researching"
  | "news"
  | "complete"
  | "error";

export interface Source {
  title: string;
  url: string;
  snippet: string;
}

export interface HistoricalPoint {
  date: string;
  price: number;
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
  historicalData?: HistoricalPoint[];
}

export interface ResearchResult {
  id: string;
  type: "finance" | "general";
  query: string;
  summary?: string;
  keyFindings?: string[];
  detailedAnalysis?: string;
  sources?: Source[];
  newsResults?: SearchResult[];
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
  stockSymbols?: string[];
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
