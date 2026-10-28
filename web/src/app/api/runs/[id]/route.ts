import { deleteDraft, getStatus } from "@/lib/runs";

export const runtime = "nodejs";

export async function GET(_req: Request, ctx: RouteContext<"/api/runs/[id]">) {
  const { id } = await ctx.params;
  let status;
  try {
    status = await getStatus(id);
  } catch {
    return Response.json({ error: "Bad run id." }, { status: 400 });
  }
  if (!status) return Response.json({ error: "No such run." }, { status: 404 });
  return Response.json(status);
}

// DELETE -> remove a run that never started training (the Train page replaces its dataset draft)
export async function DELETE(_req: Request, ctx: RouteContext<"/api/runs/[id]">) {
  const { id } = await ctx.params;
  try {
    return (await deleteDraft(id)) ? new Response(null, { status: 204 })
      : Response.json({ error: "That run has started training, so it's kept." }, { status: 409 });
  } catch {
    return Response.json({ error: "Bad run id." }, { status: 400 });
  }
}
