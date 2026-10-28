import { startFetch } from "@/lib/runs";

export const runtime = "nodejs";

// POST { publish? } -> download the trained model to this machine, and (default) publish it to a private Hugging Face repo
export async function POST(req: Request, ctx: RouteContext<"/api/runs/[id]/fetch">) {
  const { id } = await ctx.params;
  const body = await req.json().catch(() => ({}));
  try {
    await startFetch(id, body.publish !== false);
    return Response.json({ started: true }, { status: 202 });
  } catch (e) {
    return Response.json({ error: e instanceof Error ? e.message : "Could not start." }, { status: 400 });
  }
}
