import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { fetchSearchResults } from "@/lib/api-helpers";
import { rateLimit } from "@/lib/rate-limit";
import { logger } from "@/lib/logger";

const searchSchema = z.object({
  query: z.string().min(1).max(500),
});

export async function POST(req: NextRequest) {
  try {
    const ip = req.headers.get("x-forwarded-for") || "anonymous";
    const rl = rateLimit(ip, 15, 60000); // 15 requests per minute for search

    if (!rl) {
      return NextResponse.json(
        { error: "Too many requests. Please try again in a minute." },
        { status: 429 }
      );
    }

    const body = await req.json();
    const validated = searchSchema.safeParse(body);

    if (!validated.success) {
      return NextResponse.json(
        { error: "Invalid search query" },
        { status: 400 }
      );
    }

    const { query } = validated.data;

    try {
      const data = await fetchSearchResults(query);
      return NextResponse.json(data);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to fetch search results";
      return NextResponse.json({ error: message }, { status: 500 });
    }
  } catch (error) {
    logger.error("Search error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
