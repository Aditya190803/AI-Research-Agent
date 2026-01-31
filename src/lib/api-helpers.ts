import { SearchResult, FinanceData, LangSearchItem } from "@/types";
import { config } from "@/lib/config";
import { MAX_RETRIES, INITIAL_BACKOFF } from "@/lib/constants";
import { logger } from "@/lib/logger";
import { filterLowQualityResults } from "@/lib/utils";

const OPENROUTER_API_KEY = config.api.openRouter.key;
const MODEL = config.api.openRouter.model;

interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

async function fetchWithRetry(url: string, options: RequestInit, retries = MAX_RETRIES, backoff = INITIAL_BACKOFF): Promise<Response> {
  try {
    const response = await fetch(url, options);

    if (!response.ok && retries > 0) {
      if (response.status === 429) {
        // For rate limits, use a much longer backoff
        const retryAfter = response.headers.get("Retry-After");
        const waitTime = retryAfter ? parseInt(retryAfter) * 1000 : backoff * 4;
        await new Promise(resolve => setTimeout(resolve, waitTime));
        return fetchWithRetry(url, options, retries - 1, backoff * 2);
      }

      if (response.status >= 500) {
        await new Promise(resolve => setTimeout(resolve, backoff));
        return fetchWithRetry(url, options, retries - 1, backoff * 2);
      }
    }

    return response;
  } catch (error) {
    if (retries > 0) {
      await new Promise(resolve => setTimeout(resolve, backoff));
      return fetchWithRetry(url, options, retries - 1, backoff * 2);
    }
    throw error;
  }
}

/**
 * Helper to execute an array of tasks with a delay between them to avoid hitting rate limits
 */
export async function executeStaggered<T>(
  tasks: (() => Promise<T>)[],
  delayMs = 500
): Promise<T[]> {
  const results: T[] = [];
  for (const task of tasks) {
    results.push(await task());
    if (delayMs > 0 && tasks.indexOf(task) < tasks.length - 1) {
      await new Promise(resolve => setTimeout(resolve, delayMs));
    }
  }
  return results;
}

export async function callLLM(messages: Message[]): Promise<string> {
  const response = await fetchWithRetry(`${config.api.openRouter.baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      "HTTP-Referer": config.app.url,
    },
    body: JSON.stringify({
      model: MODEL,
      messages,
      temperature: 0.3,
      // Increased to allow longer research reports
      max_tokens: 4096,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenRouter API error: ${error}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || "";
}

export async function fetchFinanceData(symbol: string): Promise<FinanceData> {
  // Try Yahoo Finance first (better support for Indian stocks)
  try {
    const yahooData = await fetchYahooFinanceData(symbol);
    if (yahooData && yahooData.price > 0) {
      return yahooData;
    }
  } catch (e) {
    logger.error(`Yahoo Finance failed for ${symbol}, trying Alpha Vantage`, e);
  }

  // Fallback to Alpha Vantage
  return fetchAlphaVantageData(symbol);
}

async function fetchYahooFinanceData(symbol: string): Promise<FinanceData | null> {
  // Ensure proper symbol format for Yahoo Finance
  let yahooSymbol = symbol;

  // If no exchange suffix and appears to be an Indian stock, try .NS first
  const isLikelyIndianStock = !symbol.includes(".") && /^[A-Z]+$/.test(symbol);

  const symbolsToTry = isLikelyIndianStock
    ? [`${symbol}.NS`, `${symbol}.BO`, symbol]
    : [symbol];

  for (const sym of symbolsToTry) {
    try {
      const response = await fetchWithRetry(
        `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(sym)}?interval=1d&range=1mo&includePrePost=false`,
        {
          headers: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
          },
        }
      );

      if (!response.ok) continue;

      const data = await response.json();
      const result = data?.chart?.result?.[0];

      if (!result || !result.meta) continue;

      const meta = result.meta;
      const indicators = result.indicators?.quote?.[0];
      const timestamps = result.timestamp || [];

      // Build historical data
      const historicalData: { date: string; price: number }[] = [];
      if (indicators?.close && timestamps.length > 0) {
        for (let i = 0; i < timestamps.length; i++) {
          const closePrice = indicators.close[i];
          if (closePrice != null && !isNaN(closePrice)) {
            const date = new Date(timestamps[i] * 1000).toISOString().split('T')[0];
            historicalData.push({ date, price: closePrice });
          }
        }
      }

      // Get today's high/low from the last available data point
      const lastIdx = indicators?.high?.length ? indicators.high.length - 1 : -1;
      const todayHigh = lastIdx >= 0 ? indicators.high[lastIdx] : meta.regularMarketDayHigh;
      const todayLow = lastIdx >= 0 ? indicators.low[lastIdx] : meta.regularMarketDayLow;

      // Determine currency
      const currency = meta.currency || "INR";
      const currencySymbol = getCurrencySymbol(currency);

      // Calculate change and change percent
      const currentPrice = meta.regularMarketPrice || 0;
      const previousClose = meta.chartPreviousClose || meta.previousClose || currentPrice;
      const change = currentPrice - previousClose;
      const changePercent = previousClose > 0 ? ((change / previousClose) * 100).toFixed(2) + "%" : "0%";

      // Determine exchange from symbol suffix
      let exchange = "N/A";
      if (sym.endsWith(".NS")) exchange = "NSE";
      else if (sym.endsWith(".BO")) exchange = "BSE";
      else if (meta.exchangeName) exchange = meta.exchangeName;

      return {
        symbol: meta.symbol || sym,
        companyName: meta.longName || meta.shortName || sym.split(".")[0],
        price: currentPrice,
        change: change,
        changePercent: changePercent,
        high: todayHigh || 0,
        low: todayLow || 0,
        volume: meta.regularMarketVolume?.toString() || "0",
        marketCap: formatLargeNumber(meta.marketCap),
        peRatio: "N/A", // Yahoo chart endpoint doesn't provide P/E
        dividendYield: "N/A", // Yahoo chart endpoint doesn't provide dividend yield
        fiftyTwoWeekHigh: meta.fiftyTwoWeekHigh || 0,
        fiftyTwoWeekLow: meta.fiftyTwoWeekLow || 0,
        description: "",
        sector: "N/A",
        industry: "N/A",
        exchange: exchange,
        currency: currency,
        currencySymbol: currencySymbol,
        historicalData: historicalData,
      };
    } catch (e) {
      logger.error(`Yahoo Finance attempt failed for ${sym}`, e);
      continue;
    }
  }

  return null;
}

function getCurrencySymbol(currency: string): string {
  const symbols: Record<string, string> = {
    USD: "$",
    INR: "₹",
    EUR: "€",
    GBP: "£",
    JPY: "¥",
    CNY: "¥",
    AUD: "A$",
    CAD: "C$",
  };
  return symbols[currency] || currency;
}

function formatLargeNumber(num: number | undefined): string {
  if (!num || isNaN(num)) return "N/A";
  if (num >= 1e12) return `${(num / 1e12).toFixed(2)}T`;
  if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e7) return `${(num / 1e7).toFixed(2)}Cr`;
  if (num >= 1e5) return `${(num / 1e5).toFixed(2)}L`;
  if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
  return num.toString();
}

async function fetchAlphaVantageData(symbol: string): Promise<FinanceData> {
  const apiKey = config.api.alphaVantage.key;

  // Internal helper to fetch for a specific symbol
  async function attemptFetch(sym: string) {
    const [quoteResponse, overviewResponse, historyResponse] = await Promise.all([
      fetchWithRetry(
        `${config.api.alphaVantage.baseUrl}/query?function=GLOBAL_QUOTE&symbol=${sym}&apikey=${apiKey}`,
        {}
      ),
      fetchWithRetry(
        `${config.api.alphaVantage.baseUrl}/query?function=OVERVIEW&symbol=${sym}&apikey=${apiKey}`,
        {}
      ),
      fetchWithRetry(
        `${config.api.alphaVantage.baseUrl}/query?function=TIME_SERIES_DAILY&symbol=${sym}&apikey=${apiKey}`,
        {}
      ),
    ]);

    const quoteData = await quoteResponse.json();
    const overviewData = await overviewResponse.json();
    const historyData = await historyResponse.json();

    return { quoteData, overviewData, historyData };
  }

  let { quoteData, overviewData, historyData } = await attemptFetch(symbol);

  // If initial fetch fails for Indian stocks (common issue), try with .NS then .BSE
  if ((!quoteData["Global Quote"] || Object.keys(quoteData["Global Quote"]).length === 0) && !symbol.includes(".")) {
    try {
      const nsResult = await attemptFetch(`${symbol}.NS`);
      if (nsResult.quoteData["Global Quote"] && Object.keys(nsResult.quoteData["Global Quote"]).length > 0) {
        quoteData = nsResult.quoteData;
        overviewData = nsResult.overviewData;
        historyData = nsResult.historyData;
      } else {
        const bseResult = await attemptFetch(`${symbol}.BSE`);
        if (bseResult.quoteData["Global Quote"] && Object.keys(bseResult.quoteData["Global Quote"]).length > 0) {
          quoteData = bseResult.quoteData;
          overviewData = bseResult.overviewData;
          historyData = bseResult.historyData;
        }
      }
    } catch (e) {
      logger.error(`Retry fail for ${symbol}`, e);
    }
  }

  if (quoteData.Note || overviewData.Note || historyData.Note) {
    throw new Error("API rate limit reached. Please try again later.");
  }

  let quote = quoteData["Global Quote"];
  if (!quote || Object.keys(quote).length === 0) {
    // Attempt fallback by stripping exchange suffix (e.g., INFY.NS -> INFY)
    if (symbol.includes(".")) {
      try {
        const base = symbol.split(".")[0];
        const baseResult = await attemptFetch(base);
        if (baseResult.quoteData["Global Quote"] && Object.keys(baseResult.quoteData["Global Quote"]).length > 0) {
          quoteData = baseResult.quoteData;
          overviewData = baseResult.overviewData;
          historyData = baseResult.historyData;
          quote = quoteData["Global Quote"];
        }
      } catch (e) {
        logger.error(`Fallback fetch failed for ${symbol}`, e);
      }
    }

    if (!quote || Object.keys(quote).length === 0) {
      logger.warn(`No data found for symbol: ${symbol}, returning placeholder finance data.`);
      return {
        symbol,
        companyName: symbol,
        price: 0,
        change: 0,
        changePercent: "0%",
        high: 0,
        low: 0,
        volume: "0",
        marketCap: "N/A",
        peRatio: "N/A",
        dividendYield: "N/A",
        fiftyTwoWeekHigh: 0,
        fiftyTwoWeekLow: 0,
        description: "",
        sector: "N/A",
        industry: "N/A",
        exchange: "N/A",
        currency: "INR",
        currencySymbol: "₹",
        historicalData: [],
      };
    }
  }

  const timeSeries = historyData["Time Series (Daily)"];
  let historicalData: { date: string; price: number }[] = [];

  if (timeSeries) {
    historicalData = Object.entries(timeSeries)
      .slice(0, 30) // Last 30 days
      .map(([date, values]: [string, any]) => ({
        date,
        price: parseFloat(values["4. close"]),
      }))
      .reverse();
  }

  const exchange = overviewData.Exchange || "";
  let currency = "INR";
  let currencySymbol = "₹";

  if (exchange.includes("NYSE") || exchange.includes("NASDAQ") || exchange.includes("AMEX")) {
    currency = "USD";
    currencySymbol = "$";
  } else if (exchange.includes("LSE") || exchange.includes("London")) {
    currency = "GBP";
    currencySymbol = "£";
  } else if (exchange.includes("TSE") || exchange.includes("Tokyo")) {
    currency = "JPY";
    currencySymbol = "¥";
  } else if (exchange.includes("SSE") || exchange.includes("Shanghai") || exchange.includes("SZSE") || exchange.includes("Shenzhen")) {
    currency = "CNY";
    currencySymbol = "¥";
  } else if (exchange.includes("NSE") || exchange.includes("BSE") || exchange.includes("National Stock Exchange") || exchange.includes("Bombay")) {
    currency = "INR";
    currencySymbol = "₹";
  } else if (exchange.includes("FSX") || exchange.includes("Frankfurt") || exchange.includes("XETRA")) {
    currency = "EUR";
    currencySymbol = "€";
  }

  return {
    symbol: quote["01. symbol"] || symbol,
    companyName: overviewData.Name || symbol,
    price: parseFloat(quote["05. price"]) || 0,
    change: parseFloat(quote["09. change"]) || 0,
    changePercent: quote["10. change percent"] || "0%",
    high: parseFloat(quote["03. high"]) || 0,
    low: parseFloat(quote["04. low"]) || 0,
    volume: quote["06. volume"] || "0",
    marketCap: overviewData.MarketCapitalization || "N/A",
    peRatio: overviewData.PERatio || "N/A",
    dividendYield: overviewData.DividendYield || "N/A",
    fiftyTwoWeekHigh: parseFloat(overviewData["52WeekHigh"]) || 0,
    fiftyTwoWeekLow: parseFloat(overviewData["52WeekLow"]) || 0,
    description: overviewData.Description || "",
    sector: overviewData.Sector || "N/A",
    industry: overviewData.Industry || "N/A",
    exchange: exchange || "N/A",
    currency,
    currencySymbol,
    historicalData,
  };
}

export async function fetchSearchResults(query: string): Promise<{ results: SearchResult[]; summary: string }> {
  const apiKey = config.api.langSearch.key;

  const response = await fetchWithRetry(`${config.api.langSearch.baseUrl}/web-search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      query: query,
      freshness: "noLimit",
      summary: true,
      count: 20,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`LangSearch API error: ${errorText}`);
  }

  const data = await response.json();

  const rawResults =
    data.data?.webPages?.value?.map(
      (item: LangSearchItem) => ({
        title: item.name || "Untitled",
        url: item.url || "",
        content: item.summary || item.snippet || "",
      })
    ) || [];

  const results = filterLowQualityResults(rawResults);

  return {
    results,
    summary: data.data?.summary || "",
  };
}

export async function fetchNewsResults(query: string): Promise<SearchResult[]> {
  try {
    const apiKey = config.api.langSearch.key;
    const response = await fetchWithRetry(`${config.api.langSearch.baseUrl}/news-search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        query: query,
        count: 10,
      }),
    });

    if (response.ok) {
      const data = await response.json();
      return data.data?.value?.map((item: any) => ({
        title: item.name || item.title || "Untitled News",
        url: item.url || "",
        content: item.description || item.snippet || item.summary || "",
      })) || [];
    }
  } catch (error) {
    logger.error(`Failed to fetch news for ${query}`, error);
  }
  return [];
}

export async function fetchPageContent(url: string): Promise<string> {
  try {
    const apiKey = config.api.langSearch.key;
    const response = await fetchWithRetry(`${config.api.langSearch.baseUrl}/content?url=${encodeURIComponent(url)}`, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
    });

    if (response.ok) {
      const data = await response.json();
      return data.data?.content || "";
    }
  } catch (error) {
    logger.error(`Failed to fetch content for ${url}`, error);
  }
  return "";
}

export async function synthesizeResults(
  query: string,
  searchResults: SearchResult[],
  financeData?: FinanceData[]
): Promise<{ summary: string; keyFindings: string[]; detailedAnalysis: string }> {
  // Scrape top 2 results for deeper context - keep inputs smaller to avoid exceeding LLM context limits
  const topResults = searchResults.slice(0, 2);
  const scrapedContents = await executeStaggered(
    topResults.map(r => () => fetchPageContent(r.url)),
    800 // 800ms delay between scrapes
  );

  const MAX_SCRAPE_CHARS = 2000;
  const MAX_RESULT_CONTENT_CHARS = 1000;

  let sourcesContext = searchResults
    ?.map((r, i) => {
      const scrapedContent = i < topResults.length ? scrapedContents[i] : null;
      const content = scrapedContent && scrapedContent.length > 0
        ? scrapedContent.slice(0, MAX_SCRAPE_CHARS)
        : (r.content || "").slice(0, MAX_RESULT_CONTENT_CHARS);

      return `[${i + 1}] ${r.title}\nURL: ${r.url}\nContent: ${content}`;
    })
    .join("\n\n") || "";

  // Ensure total source context stays within a safe character budget
  const MAX_TOTAL_CONTEXT_CHARS = 120000;
  if (sourcesContext.length > MAX_TOTAL_CONTEXT_CHARS) {
    // truncate to fit the budget
    sourcesContext = sourcesContext.slice(0, MAX_TOTAL_CONTEXT_CHARS);
  }

  const financeContext = financeData && financeData.length > 0
    ? `\n\nFinancial Data:\n${financeData.slice(0, 5).map(fd => {
      const desc = (fd.description || "").slice(0, 500);
      return `Company: ${fd.companyName} (${fd.symbol})\nPrice: ${fd.currencySymbol || '₹'}${fd.price}\nChange: ${fd.change} (${fd.changePercent})\nMarket Cap: ${fd.marketCap}\nP/E Ratio: ${fd.peRatio}\n52-Week Range: ${fd.currencySymbol || '₹'}${fd.fiftyTwoWeekLow} - ${fd.currencySymbol || '₹'}${fd.fiftyTwoWeekHigh}\nDescription: ${desc}`;
    }).join('\n\n')}`
    : "";

  const messages: Message[] = [
    {
      role: "system",
      content: `You are an expert Senior Research Analyst. Your task is to provide a DEEP, COMPREHENSIVE research report based on the provided sources and financial data.

CRITICAL INSTRUCTIONS:
1. DO NOT just summarize. Analyze, compare, and provide deep insights.
2. If financial data is provided, incorporate it deeply into your analysis. Mention specifically that a chart and detailed financial metrics are available below.
3. Use a professional, authoritative tone.
4. Cite your sources accurately using [1], [2], etc.
5. Unless another country is explicitly mentioned, assume an Indian context (INR, NSE/BSE, RBI, SEBI).

Your report MUST have:
1. Executive Summary: A high-level overview of the most critical insights (2-4 sentences).
2. Key Findings: 4-6 distinct, high-impact takeaways.
3. Detailed Analysis: A multi-paragraph, in-depth evaluation covering background, current status, future outlook, and risks/opportunities.

Respond in JSON format:
{
  "summary": "...",
  "keyFindings": ["...", "...", ...],
  "detailedAnalysis": "..."
}`,
    },
    {
      role: "user",
      content: `Research Query: ${query}\n\nSources:\n${sourcesContext}${financeContext}`,
    },
  ];

  const response = await callLLM(messages);

  // Helper function to clean up JSON strings with common issues
  const cleanJsonString = (str: string): string => {
    return str
      .replace(/,\s*([\]}])/g, '$1') // Remove trailing commas
      .replace(/\n/g, '\\n') // Escape newlines in strings
      .replace(/\r/g, '\\r') // Escape carriage returns
      .replace(/\t/g, '\\t'); // Escape tabs
  };

  // Helper function to extract field value from JSON-like text
  const extractField = (text: string, fieldName: string): string | null => {
    // Try to find the field with various patterns
    const patterns = [
      new RegExp(`"${fieldName}"\\s*:\\s*"([^"]*(?:\\\\.[^"]*)*)"`, 's'),
      new RegExp(`"${fieldName}"\\s*:\\s*"([\\s\\S]*?)(?:"|$)`, 's'),
      new RegExp(`"${fieldName}"\\s*:\\s*\`([\\s\\S]*?)\``, 's'),
    ];

    for (const pattern of patterns) {
      const match = text.match(pattern);
      if (match && match[1]) {
        return match[1].replace(/\\n/g, '\n').replace(/\\"/g, '"');
      }
    }
    return null;
  };

  // Helper function to extract array field
  const extractArrayField = (text: string, fieldName: string): string[] => {
    const pattern = new RegExp(`"${fieldName}"\\s*:\\s*\\[([\\s\\S]*?)\\]`, 's');
    const match = text.match(pattern);
    if (match && match[1]) {
      // Try to parse individual items
      const items: string[] = [];
      const itemMatches = match[1].matchAll(/"([^"]*(?:\\.[^"]*)*)"/g);
      for (const itemMatch of itemMatches) {
        if (itemMatch[1]) {
          items.push(itemMatch[1].replace(/\\n/g, '\n').replace(/\\"/g, '"'));
        }
      }
      return items;
    }
    return [];
  };

  try {
    // Try to find and parse JSON
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      try {
        // First try direct parse
        const parsed = JSON.parse(jsonMatch[0]);
        return {
          summary: parsed.summary || "",
          keyFindings: Array.isArray(parsed.keyFindings) ? parsed.keyFindings : [],
          detailedAnalysis: parsed.detailedAnalysis || "",
        };
      } catch {
        // If direct parse fails, try with cleaned JSON
        try {
          const cleaned = cleanJsonString(jsonMatch[0]);
          const parsed = JSON.parse(cleaned);
          return {
            summary: parsed.summary || "",
            keyFindings: Array.isArray(parsed.keyFindings) ? parsed.keyFindings : [],
            detailedAnalysis: parsed.detailedAnalysis || "",
          };
        } catch {
          // Fall through to field extraction
        }
      }
    }
  } catch {
    // Fall through to field extraction
  }

  // Fallback: Try to extract fields individually (handles truncated JSON)
  logger.warn("JSON parsing failed, attempting field extraction");
  const summary = extractField(response, "summary");
  const keyFindings = extractArrayField(response, "keyFindings");
  const detailedAnalysis = extractField(response, "detailedAnalysis");

  if (summary || detailedAnalysis) {
    return {
      summary: summary || "Research completed successfully.",
      keyFindings: keyFindings.length > 0 ? keyFindings : ["See detailed analysis below"],
      detailedAnalysis: detailedAnalysis || summary || response,
    };
  }

  // Last resort: Use the raw response but clean it up
  logger.warn("Field extraction failed, using raw response");
  return {
    summary: "Research completed. See detailed analysis for findings.",
    keyFindings: ["Research data collected from multiple sources"],
    detailedAnalysis: response
      .replace(/^\s*\{?\s*"?summary"?\s*:\s*"?/i, '')
      .replace(/"?\s*,?\s*"?keyFindings"?\s*:\s*\[.*?\]\s*,?/is, '\n\n')
      .replace(/"?\s*,?\s*"?detailedAnalysis"?\s*:\s*"?/i, '')
      .replace(/"\s*\}?\s*$/i, '')
      .trim(),
  };
}
