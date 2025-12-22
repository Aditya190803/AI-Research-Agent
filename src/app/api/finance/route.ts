import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { fetchFinanceData } from "@/lib/api-helpers";
import { rateLimit } from "@/lib/rate-limit";
import { logger } from "@/lib/logger";

const financeSchema = z.object({
  symbol: z.string().min(1).max(10).regex(/^[A-Z0-9.]+$/),
});

export async function POST(req: NextRequest) {
  try {
    const ip = req.headers.get("x-forwarded-for") || "anonymous";
    const rl = rateLimit(ip, 20, 60000); // 20 requests per minute for finance

    if (!rl) {
      return NextResponse.json(
        { error: "Too many requests. Please try again in a minute." },
        { status: 429 }
      );
    }

    const body = await req.json();
    const validated = financeSchema.safeParse(body);

    if (!validated.success) {
      return NextResponse.json(
        { error: "Invalid stock symbol format" },
        { status: 400 }
      );
    }

    const { symbol } = validated.data;

    try {
      const data = await fetchFinanceData(symbol);
      return NextResponse.json({ data });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to fetch finance data";
      const status = message.includes("rate limit") ? 429 : 400;
      return NextResponse.json({ error: message }, { status });
    }
  } catch (error) {
    logger.error("Finance API error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
