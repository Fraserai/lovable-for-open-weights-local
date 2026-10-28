import { findTrainable } from "@/lib/catalog";
import { listRuns, newRunId, startPrepare, type HubSource } from "@/lib/runs";

export const runtime = "nodejs";

const MAX_TOTAL_BYTES = 200 * 1024 * 1024;
const HF_ID = /^[\w.-]+\/[\w.-]+$/;

// GET -> every run, newest first (History and the sidebar)
export async function GET() {
  return Response.json(await listRuns());
}

// POST -> start a run: save uploads (or name a Hugging Face dataset), then have the worker
// write or import the training examples and check every row.
export async function POST(req: Request) {
  const form = await req.formData();
  const goal = String(form.get("goal") ?? "").trim();
  const modelId = String(form.get("model") ?? "");
  const files = form.getAll("files").filter((f): f is File => f instanceof File && f.size > 0);
  const hf = String(form.get("hf") ?? "").trim();
  const field = (k: string) => String(form.get(k) ?? "").trim() || undefined;

  const model = await findTrainable(modelId);
  // Documents are turned into examples for a goal; a ready-made dataset only uses it as an optional system prompt
  const documents = !hf && files.some((f) => !f.name.toLowerCase().endsWith(".jsonl"));
  if (documents && !goal) return Response.json({ error: "Say what the model should learn from these documents." }, { status: 400 });
  if (!model) return Response.json({ error: "No connected trainer can fine-tune that model yet." }, { status: 400 });
  if (!model.ready) return Response.json({ error: `${model.name} on ${model.trainer} isn't connected yet.` }, { status: 400 });

  let hub: HubSource | undefined;
  if (hf) {
    if (!HF_ID.test(hf)) return Response.json({ error: "That isn't a Hugging Face dataset id (owner/name)." }, { status: 400 });
    const limit = Math.min(Math.max(Number(form.get("limit")) || 1000, 10), 20000);
    hub = { id: hf, config: field("config"), split: field("split"), limit, userField: field("userField"), assistantField: field("assistantField") };
  } else if (files.length === 0) {
    return Response.json({ error: "Add documents or a dataset to learn from." }, { status: 400 });
  } else if (files.reduce((n, f) => n + f.size, 0) > MAX_TOTAL_BYTES) {
    return Response.json({ error: "Uploads are over 200 MB in total. Start with fewer documents." }, { status: 413 });
  }

  const id = newRunId(goal || hf.split("/").pop() || files[0]?.name.replace(/\.[^.]+$/, "") || "run");
  await startPrepare(id, goal, model.id, hub ? [] : files, hub);
  return Response.json({ id }, { status: 202 });
}
