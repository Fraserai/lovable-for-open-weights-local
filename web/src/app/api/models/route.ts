import type { NextRequest } from "next/server";
import { searchHub, trainableModels } from "@/lib/catalog";

export const runtime = "nodejs";

// GET /api/models            -> every model a connected trainer can fine-tune
// GET /api/models?q=qwen     -> plus matching open models on Hugging Face, labelled by trainer
export async function GET(req: NextRequest) {
  const q = req.nextUrl.searchParams.get("q")?.trim() ?? "";
  const trainable = await trainableModels();
  const hub = q.length >= 2 ? await searchHub(q).catch(() => []) : [];
  return Response.json({ trainable, hub });
}
