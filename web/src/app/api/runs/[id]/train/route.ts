import { getPlan, startTraining } from "@/lib/runs";

export const runtime = "nodejs";

// GET  -> what training would cost (free)
// POST -> start the paid training run
export async function GET(_req: Request, ctx: RouteContext<"/api/runs/[id]/train">) {
  const { id } = await ctx.params;
  try {
    return Response.json(await getPlan(id));
  } catch (e) {
    return Response.json({ error: e instanceof Error ? e.message : "Could not plan this run." }, { status: 400 });
  }
}

export async function POST(_req: Request, ctx: RouteContext<"/api/runs/[id]/train">) {
  const { id } = await ctx.params;
  try {
    const plan = await getPlan(id);
    if (!plan.ok) return Response.json({ error: "The training data has errors. Fix them first." }, { status: 400 });
    await startTraining(id);
    return Response.json({ started: true }, { status: 202 });
  } catch (e) {
    return Response.json({ error: e instanceof Error ? e.message : "Could not start training." }, { status: 400 });
  }
}
