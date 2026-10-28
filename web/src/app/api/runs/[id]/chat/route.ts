import { chat } from "@/lib/runs";

export const runtime = "nodejs";

// POST { message } -> { original, trained }: the same question to both models
export async function POST(req: Request, ctx: RouteContext<"/api/runs/[id]/chat">) {
  const { id } = await ctx.params;
  const { message } = await req.json().catch(() => ({}));
  if (typeof message !== "string" || !message.trim()) {
    return Response.json({ error: "Type a question first." }, { status: 400 });
  }
  try {
    return Response.json(await chat(id, message.trim().slice(0, 4000)));
  } catch (e) {
    return Response.json({ error: e instanceof Error ? e.message : "The models didn't answer." }, { status: 400 });
  }
}
