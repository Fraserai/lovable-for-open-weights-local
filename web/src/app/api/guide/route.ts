import { askGuide, type GuideMessage } from "@/lib/guide-agent";

export const runtime = "nodejs";

// POST { messages, page } -> { reply, actions }: the guide's answer, plus form changes for the page to make
export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  const messages: GuideMessage[] = Array.isArray(body.messages)
    ? body.messages.filter((m: GuideMessage) => (m?.role === "user" || m?.role === "assistant") && typeof m.content === "string")
        .map((m: GuideMessage) => ({ role: m.role, content: m.content.slice(0, 4000) }))
    : [];
  if (!messages.length || messages.at(-1)!.role !== "user") return Response.json({ error: "Ask something first." }, { status: 400 });
  try {
    return Response.json(await askGuide(messages, body.page ?? {}));
  } catch (e) {
    return Response.json({ error: e instanceof Error ? e.message : "The guide can't answer right now." }, { status: 502 });
  }
}
