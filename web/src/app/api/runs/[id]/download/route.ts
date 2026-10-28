import { getStatus, modelArchive } from "@/lib/runs";

export const runtime = "nodejs";

// GET -> the trained LoRA adapter (runs/<id>/model) as <name>-lora.tar.gz
export async function GET(_req: Request, ctx: RouteContext<"/api/runs/[id]/download">) {
  const { id } = await ctx.params;
  const run = await getStatus(id).catch(() => null);
  if (!run?.train.localModel) return Response.json({ error: "This model isn't on this machine yet. Get the weights first." }, { status: 404 });
  const name = (run.settings.project_name || id).replace(/[^\w.-]+/g, "-");
  return new Response(modelArchive(id), {
    headers: { "Content-Type": "application/gzip", "Content-Disposition": `attachment; filename="${name}-lora.tar.gz"` },
  });
}
