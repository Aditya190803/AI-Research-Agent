import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { detectQueryType, extractStockSymbols } from "@/lib/utils";
import { 
  fetchFinanceData, 
  fetchSearchResults, 
  synthesizeResults, 
  callLLM 
} from "@/lib/api-helpers";
import { rateLimit } from "@/lib/rate-limit";
import { FinanceData, SearchResult } from "@/types";
import { logger } from "@/lib/logger";

const requestSchema = z.object({
  action: z.enum(["execute", "explore", "strategy", "synthesize"]),
  query: z.string().min(1).max(1000).optional(),
  subQueries: z.array(z.string()).optional(),
  context: z.string().max(2000).optional(),
  searchResults: z.array(z.custom<SearchResult>()).optional(),
  financeData: z.union([z.custom<FinanceData>(), z.array(z.custom<FinanceData>())]).optional(),
});

interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

export async function POST(req: NextRequest) {
  try {
    // Simple rate limiting
    const ip = req.headers.get("x-forwarded-for") || "anonymous";
    const rl = rateLimit(ip, 10, 60000); // 10 requests per minute

    if (!rl) {
      return NextResponse.json(
        { error: "Too many requests. Please try again in a minute." },
        { status: 429 }
      );
    }

    const body = await req.json();
    const validated = requestSchema.safeParse(body);

    if (!validated.success) {
      return NextResponse.json(
        { error: "Invalid request data", details: validated.error.format() },
        { status: 400 }
      );
    }

    const { action, query = "", subQueries = [], context, searchResults, financeData } = validated.data;

    switch (action) {
      case "execute": {
        if (!query) return NextResponse.json({ error: "Query is required" }, { status: 400 });
        
        const queryType = detectQueryType(query);
        const finance: FinanceData[] = [];
        let allSearchResults: SearchResult[] = [];

        // Run finance detection and search in parallel
        const tasks: Promise<void>[] = [];

        if (queryType === "finance") {
          const symbols = extractStockSymbols(query);
          if (symbols.length > 0) {
            symbols.forEach(symbol => {
              tasks.push(
                fetchFinanceData(symbol)
                  .then(data => {
                    finance.push(data);
                  })
                  .catch(e => logger.error(`Finance API error for ${symbol}`, e))
              );
            });
          }
        }

        // Use subQueries if provided, otherwise use the main query
        const queriesToRun = subQueries.length > 0 ? subQueries : [query];
        
        queriesToRun.forEach(q => {
          tasks.push(fetchSearchResults(q).then(res => {
            allSearchResults = [...allSearchResults, ...res.results];
          }).catch(e => logger.error(`Search API error for query "${q}"`, e)));
        });

        await Promise.all(tasks);

        // Deduplicate search results by URL
        const uniqueResults = Array.from(new Map(allSearchResults.map(item => [item.url, item])).values());

        const synthesis = await synthesizeResults(
          query,
          uniqueResults,
          finance
        );

        return NextResponse.json({
          ...synthesis,
          searchResults: uniqueResults,
          financeData: finance,
          type: queryType,
        });
      }

      case "explore": {
        const messages: Message[] = [
          {
            role: "system",
            content: `You are a research analyst based in India. Analyze the user's query and provide:
1. A brief initial assessment of the topic
2. Key areas that need investigation  
3. 2-4 clarifying questions to better understand what the user wants

IMPORTANT: Unless another country is explicitly mentioned in the query, assume an Indian context. This means:
- Use Indian Rupees (₹/INR) for prices and financial figures
- Reference Indian markets (NSE, BSE), regulations, and policies
- Consider Indian economic conditions, taxes, and laws
- Use Indian examples, companies, and data sources when relevant
- Reference Indian government schemes, RBI policies, SEBI regulations where applicable

You MUST respond with valid JSON in this exact format:
{
  "assessment": "Brief assessment of what the user is asking about",
  "keyAreas": ["area1", "area2", "area3"],
  "needsClarification": true,
  "clarificationQuestions": [
    {
      "id": "q1",
      "question": "What is your investment timeline?",
      "options": ["Short-term (under 1 year)", "Medium-term (1-5 years)", "Long-term (5+ years)"]
    },
    {
      "id": "q2", 
      "question": "What is your risk tolerance?",
      "options": ["Conservative (low risk)", "Moderate (balanced risk)", "Aggressive (high risk)"]
    }
  ]
}

CRITICAL RULES:
1. The "question" field must be a simple STRING containing the question text
2. The "options" field must be an ARRAY of 3-5 STRING options relevant to that question
3. Options must be meaningful answers to the specific question - NOT generic Yes/No
4. Each question should help clarify the user's intent, goals, preferences, or constraints
5. Always set needsClarification to true and provide questions`,
          },
          {
            role: "user",
            content: `Query: ${query}${context ? `\n\nContext from previous research: ${context}` : ""}`,
          },
        ];

        const response = await callLLM(messages);
        
        // Try to parse JSON, handle potential formatting issues
        try {
          const jsonMatch = response.match(/\{[\s\S]*\}/);
          let parsed: Record<string, unknown> | null = null;
          
          if (jsonMatch) {
            try {
              // Clean up common JSON issues from LLMs
              const jsonStr = jsonMatch[0]
                .replace(/,\s*([\]}])/g, '$1') // Remove trailing commas
                .replace(/(['"])?([a-zA-Z0-9_]+)(['"])?:/g, '"$2":'); // Ensure keys are quoted
              
              parsed = JSON.parse(jsonStr);
            } catch (e) {
              logger.error("Failed to parse JSON match, trying raw match", e);
              try {
                parsed = JSON.parse(jsonMatch[0]);
              } catch (e2) {
                logger.error("Failed to parse raw JSON match", e2);
              }
            }
          }

          if (parsed) {
            // Normalize clarification questions to ensure correct format
            if (parsed.clarificationQuestions && Array.isArray(parsed.clarificationQuestions)) {
              parsed.clarificationQuestions = parsed.clarificationQuestions.map((q: unknown, idx: number) => {
                if (typeof q === 'string') {
                  return { id: `q${idx + 1}`, question: q, options: [] };
                }
                if (q && typeof q === 'object') {
                  const qObj = q as Record<string, unknown>;
                  return {
                    id: typeof qObj.id === 'string' ? qObj.id : `q${idx + 1}`,
                    question: typeof qObj.question === 'string' ? qObj.question : String(qObj.question || ''),
                    options: Array.isArray(qObj.options) ? qObj.options.filter((o): o is string => typeof o === 'string') : [],
                  };
                }
                return { id: `q${idx + 1}`, question: '', options: [] };
              });
            }
            
            return NextResponse.json({
              assessment: parsed.assessment || "Assessment generated",
              keyAreas: Array.isArray(parsed.keyAreas) ? parsed.keyAreas : [],
              needsClarification: !!parsed.needsClarification,
              clarificationQuestions: parsed.clarificationQuestions || [],
            });
          }
        } catch (err) {
          logger.error("Clarification parsing error:", err);
        }
        
        // Final fallback
        return NextResponse.json({
          assessment: response.slice(0, 500),
          keyAreas: [],
          needsClarification: false,
          clarificationQuestions: [],
        });
      }

      case "strategy": {
        const messages: Message[] = [
          {
            role: "system",
            content: `You are a research strategist based in India. Based on the query and any provided context, create a research strategy.

IMPORTANT: Unless another country is explicitly mentioned, assume an Indian context:
- Focus on Indian markets, companies, regulations, and data
- Use INR (₹) for financial references
- Consider Indian-specific factors like RBI policies, SEBI regulations, GST, Indian tax laws
- Prioritize Indian sources and examples
- Reference NSE/BSE for stock markets, Indian banks, and local institutions

Respond in JSON format:
{
  "scope": "A clear, concise scope statement",
  "optimizedQuery": "The refined search query for deep research (add 'India' or 'Indian' to the query if no country is specified)",
  "subQueries": ["Query 1 for specific aspect", "Query 2 for another aspect", "Query 3 for financial/regulatory aspect"],
  "investigationPath": ["Step 1: ...", "Step 2: ...", "Step 3: ..."]
}

CRITICAL: Provide 3-5 distinct sub-queries that cover different aspects of the research topic to ensure comprehensive coverage. Each sub-query should be optimized for a search engine.`,
          },
          {
            role: "user",
            content: `Query: ${query}\n\nAdditional Context: ${context || "None provided"}`,
          },
        ];

        const response = await callLLM(messages);
        
        try {
          const jsonMatch = response.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            return NextResponse.json(JSON.parse(jsonMatch[0]));
          }
        } catch {
          return NextResponse.json({
            scope: query,
            optimizedQuery: query,
            subQueries: [query],
            investigationPath: ["Research the topic", "Analyze findings", "Synthesize results"],
          });
        }
        
        return NextResponse.json({
          scope: query,
          optimizedQuery: query,
          subQueries: [query],
          investigationPath: ["Research the topic", "Analyze findings", "Synthesize results"],
        });
      }

      case "synthesize": {
        const normalizedFinanceData = Array.isArray(financeData) 
          ? financeData 
          : financeData 
            ? [financeData] 
            : [];
            
        const synthesis = await synthesizeResults(
          query,
          searchResults || [],
          normalizedFinanceData
        );
        return NextResponse.json(synthesis);
      }

      default:
        return NextResponse.json({ error: "Invalid action" }, { status: 400 });
    }
  } catch (error) {
    logger.error("Research API error:", error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Internal server error" },
      { status: 500 }
    );
  }
}
