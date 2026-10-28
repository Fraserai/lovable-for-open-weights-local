import { FRONTIERS } from "@/lib/models";
import { frontierEstimate, startScore } from "@/lib/runs";

export const runtime = "nodejs";

// GET ?frontier=<id> -> { estimate } (dollars for the frontier's answers; free to ask)
export async function GET(req: Request, ctx: RouteContext<"/api/runs/[id]/score">) {
  const { id } = await ctx.params;
  const model = new URL(req.url).searchParams.get("frontier") ?? FRONTIERS[0].id;
  if (!FRONTIERS.some((f) => f.id === model)) return Response.json({ error: "Unknown frontier model." }, { status: 400 });
  try {
    return Response.json({ estimate: await frontierEstimate(id, model) });
  } catch {
    return Response.json({ error: "Bad run id." }, { status: 400 });
  }
}

// POST { frontier: <id> | null } -> start scoring (paid: the trainer and OpenRouter answer the held-back questions)
export async function POST(req: Request, ctx: RouteContext<"/api/runs/[id]/score">) {
  const { id } = await ctx.params;
  const body = await req.json().catch(() => ({}));
  const frontier = body.frontier === null ? null : FRONTIERS.some((f) => f.id === body.frontier) ? body.frontier : FRONTIERS[0].id;
  try {
    await startScore(id, frontier);
    return Response.json({ started: true }, { status: 202 });
  } catch (e) {
    return Response.json({ error: e instanceof Error ? e.message : "Could not start scoring." }, { status: 400 });
  }
}
