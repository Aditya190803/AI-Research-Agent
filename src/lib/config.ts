import { env } from "./env";

export const config = {
  app: {
    name: "AI Research Agent",
    description: "AI-powered research agent and financial analysis tool",
    url: env.NEXT_PUBLIC_APP_URL,
  },
  api: {
    openRouter: {
      key: env.OPENROUTER_API_KEY,
      baseUrl: "https://openrouter.ai/api/v1",
      model: "xiaomi/mimo-v2-flash:free",
    },
    langSearch: {
      key: env.LANGSEARCH_API_KEY,
      baseUrl: "https://api.langsearch.com/v1",
    },
    alphaVantage: {
      key: env.ALPHA_VANTAGE_API_KEY,
      baseUrl: "https://www.alphavantage.co",
    },
  },
  isDevelopment: process.env.NODE_ENV === "development",
  isProduction: process.env.NODE_ENV === "production",
};
