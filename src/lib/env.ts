import { z } from "zod";

const envSchema = z.object({
  OPENROUTER_API_KEY: z.string().min(1, "OPENROUTER_API_KEY is required"),
  LANGSEARCH_API_KEY: z.string().min(1, "LANGSEARCH_API_KEY is required"),
  ALPHA_VANTAGE_API_KEY: z.string().min(1, "ALPHA_VANTAGE_API_KEY is required"),
  NEXT_PUBLIC_APP_URL: z.string().url().optional().default("http://localhost:3000"),
});

export const env = envSchema.parse({
  OPENROUTER_API_KEY: process.env.OPENROUTER_API_KEY,
  LANGSEARCH_API_KEY: process.env.LANGSEARCH_API_KEY,
  ALPHA_VANTAGE_API_KEY: process.env.ALPHA_VANTAGE_API_KEY,
  NEXT_PUBLIC_APP_URL: process.env.NEXT_PUBLIC_APP_URL,
});
