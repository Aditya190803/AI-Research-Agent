export const APP_NAME = "Deep Research";
export const APP_DESCRIPTION = "AI-powered deep research and financial analysis tool";

export const API_URLS = {
  OPENROUTER: "https://openrouter.ai/api/v1/chat/completions",
  LANGSEARCH: "https://api.langsearch.com/v1/web-search",
  ALPHA_VANTAGE: "https://www.alphavantage.co/query",
};

export const LLM_MODELS = {
  PRIMARY: "xiaomi/mimo-v2-flash:free",
};

export const ANIMATION_DURATIONS = {
  FAST: 0.2,
  NORMAL: 0.3,
  SLOW: 0.5,
};

export const COLORS = {
  PRIMARY: "#3b82f6", // blue-500
  SECONDARY: "#10b981", // emerald-500
  ACCENT: "#8b5cf6", // violet-500
  ERROR: "#ef4444", // red-500
  WARNING: "#f59e0b", // amber-500
  SUCCESS: "#10b981", // emerald-500
};

export const CACHE_TIMES = {
  FINANCE_DATA: 15 * 60 * 1000, // 15 minutes
};

export const MAX_HISTORY_ITEMS = 20;

export const DEBOUNCE_DELAY = 300;

export const MAX_RETRIES = 3;
export const INITIAL_BACKOFF = 1000;

export const RESEARCH_TEMPLATES = [
  {
    id: "market-analysis",
    title: "Market Analysis",
    description: "Analyze market trends, size, and growth potential",
    query: "Perform a comprehensive market analysis for [Industry/Product]. Include market size, growth drivers, key trends, and future outlook for the next 5 years.",
    icon: "BarChart3",
  },
  {
    id: "competitor-research",
    title: "Competitor Research",
    description: "Deep dive into competitor strategies and positioning",
    query: "Conduct a detailed competitor research on [Company/Product]. Identify top 3-5 competitors, their market share, pricing strategies, key strengths, and weaknesses.",
    icon: "Users",
  },
  {
    id: "financial-health",
    title: "Financial Health",
    description: "Evaluate financial performance and stability",
    query: "Analyze the financial health of [Company Name]. Review their latest quarterly earnings, debt-to-equity ratio, cash flow, and profitability margins.",
    icon: "TrendingUp",
  },
  {
    id: "tech-stack-audit",
    title: "Tech Stack Audit",
    description: "Investigate technologies used by a company",
    query: "Research the technology stack used by [Company/Website]. Identify their frontend frameworks, backend infrastructure, cloud providers, and any notable third-party tools.",
    icon: "Cpu",
  },
];
