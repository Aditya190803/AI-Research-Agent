import { SearchResult, FinanceData, LangSearchItem } from "@/types";
import { config } from "@/lib/config";
import { MAX_RETRIES, INITIAL_BACKOFF } from "@/lib/constants";

const OPENROUTER_API_KEY = config.api.openRouter.key;
const MODEL = config.api.openRouter.model;

interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

async function fetchWithRetry(url: string, options: RequestInit, retries = MAX_RETRIES, backoff = INITIAL_BACKOFF): Promise<Response> {
  try {
    const response = await fetch(url, options);
    if (!response.ok && retries > 0 && (response.status === 429 || response.status >= 500)) {
      await new Promise(resolve => setTimeout(resolve, backoff));
      return fetchWithRetry(url, options, retries - 1, backoff * 2);
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
      temperature: 0.7,
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
  const apiKey = config.api.alphaVantage.key;

  const [quoteResponse, overviewResponse] = await Promise.all([
    fetchWithRetry(
      `${config.api.alphaVantage.baseUrl}/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${apiKey}`,
      {}
    ),
    fetchWithRetry(
      `${config.api.alphaVantage.baseUrl}/query?function=OVERVIEW&symbol=${symbol}&apikey=${apiKey}`,
      {}
    ),
  ]);

  const quoteData = await quoteResponse.json();
  const overviewData = await overviewResponse.json();

  if (quoteData.Note || overviewData.Note) {
    throw new Error("API rate limit reached. Please try again later.");
  }

  if (quoteData["Error Message"] || overviewData["Error Message"]) {
    throw new Error("Invalid stock symbol");
  }

  const quote = quoteData["Global Quote"];
  if (!quote || Object.keys(quote).length === 0) {
    throw new Error("No data found for this symbol");
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
      count: 10,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`LangSearch API error: ${errorText}`);
  }

  const data = await response.json();

  const results =
    data.data?.webPages?.value?.map(
      (item: LangSearchItem) => ({
        title: item.name || "Untitled",
        url: item.url || "",
        content: item.summary || item.snippet || "",
      })
    ) || [];

  return {
    results,
    summary: data.data?.summary || "",
  };
}

export async function synthesizeResults(
  query: string,
  searchResults: SearchResult[],
  financeData?: FinanceData[]
): Promise<{ summary: string; keyFindings: string[]; detailedAnalysis: string }> {
  const sourcesContext = searchResults
    ?.map((r, i) => 
      `[${i + 1}] ${r.title}\nURL: ${r.url}\nContent: ${r.content}`
    )
    .join("\n\n") || "";

  const financeContext = financeData && financeData.length > 0
    ? `\n\nFinancial Data:\n${financeData.map(fd => 
        `Company: ${fd.companyName} (${fd.symbol})\nPrice: ${fd.currencySymbol || '₹'}${fd.price}\nChange: ${fd.change} (${fd.changePercent})\nMarket Cap: ${fd.marketCap}\nP/E Ratio: ${fd.peRatio}\n52-Week Range: ${fd.currencySymbol || '₹'}${fd.fiftyTwoWeekLow} - ${fd.currencySymbol || '₹'}${fd.fiftyTwoWeekHigh}\nDescription: ${fd.description}`
      ).join('\n\n')}`
    : "";

  const messages: Message[] = [
    {
      role: "system",
      content: `You are a professional research analyst based in India. Synthesize the provided information into a comprehensive research report.

IMPORTANT: Unless another country is explicitly mentioned in the query, provide an India-focused analysis:
- Use Indian Rupees (₹/INR) for all prices and financial figures
- Reference Indian context, regulations, markets (NSE, BSE), and policies
- Consider Indian economic conditions, tax implications, and regulatory environment
- Compare with Indian benchmarks (Nifty 50, Sensex) where relevant
- Mention Indian-specific factors like RBI policies, SEBI guidelines, GST impact

Your report should include:
1. An executive summary (2-3 sentences)
2. Key findings (3-5 bullet points)
3. Detailed analysis (comprehensive but concise)
4. Cite sources using [1], [2], etc.

Respond in JSON format:
{
  "summary": "Executive summary here",
  "keyFindings": ["Finding 1", "Finding 2", "Finding 3"],
  "detailedAnalysis": "Detailed analysis with citations [1], [2], etc."
}`,
    },
    {
      role: "user",
      content: `Research Query: ${query}\n\nSources:\n${sourcesContext}${financeContext}`,
    },
  ];

  const response = await callLLM(messages);
  
  try {
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
  } catch {
    return {
      summary: "Research completed but formatting failed.",
      keyFindings: ["See detailed analysis"],
      detailedAnalysis: response,
    };
  }
  
  return {
    summary: response,
    keyFindings: [],
    detailedAnalysis: response,
  };
}
