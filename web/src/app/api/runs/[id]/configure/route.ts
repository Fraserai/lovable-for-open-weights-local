import { findTrainable } from "@/lib/catalog";
import { configure } from "@/lib/runs";

export const runtime = "nodejs";

// POST { model, settings } -> the plan with those settings (free). The worker checks every value.
export async function POST(req: Request, ctx: RouteContext<"/api/runs/[id]/configure">) {
  const { id } = await ctx.params;
  const body = await req.json().catch(() => ({}));
  const model = await findTrainable(String(body.model ?? ""));
  if (!model?.ready) return Response.json({ error: "No connected trainer can fine-tune that model yet." }, { status: 400 });
  const settings = body.settings && typeof body.settings === "object" ? body.settings : {};
  try {
    return Response.json(await configure(id, model.id, settings));
  } catch (e) {
    return Response.json({ error: e instanceof Error ? e.message : "Could not plan this run." }, { status: 400 });
  }
}
