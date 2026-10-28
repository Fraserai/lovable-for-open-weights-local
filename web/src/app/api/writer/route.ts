import { rootEnv } from "@/lib/env";

export const runtime = "nodejs";

// GET -> which service writes training examples from documents: "openrouter" (zero data retention,
// the default) or "openai" (test or public documents only). Mirrors worker config.generator_provider.
export async function GET() {
  const env = await rootEnv();
  const provider = (process.env.GENERATOR_PROVIDER || env.GENERATOR_PROVIDER || "openrouter").toLowerCase();
  return Response.json({ provider: provider === "openai" ? "openai" : "openrouter" });
}
