import "server-only";

import { BRAND } from "./brand";
import { trainableModels } from "./catalog";
import { rootEnv } from "./env";
import { TRAINERS } from "./models";

// The in-app guide: answers questions about the Train screen and fills in the form on request.
// It runs through OpenRouter pinned to zero-data-retention providers (same rule as the worker's
// example writer), because people type their own plans and data details into it.

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";
const ZDR_PROVIDER = { zdr: true, data_collection: "deny", require_parameters: true };
// Cheap and good at tool calls; the free model keeps the guide talking when the account has no credit.
const DEFAULT_MODEL = "z-ai/glm-5.3-flash";
const FREE_FALLBACK = "qwen/qwen3.8-27b:free";
const MAX_TURNS = 12;

export type GuideMessage = { role: "user" | "assistant"; content: string };
export type GuideReply = { reply: string; actions: { name: string; args: Record<string, unknown> }[] };

const SETTING_KEYS = ["project_name", "method", "max_steps", "epochs", "context_length", "learning_rate", "lora_rank",
  "lora_alpha", "lora_dropout", "target_modules", "lora_variant", "optimizer", "lr_scheduler", "batch_size",
  "grad_accum", "weight_decay", "warmup_steps", "seed", "gradient_checkpointing"];

const TOOLS = [
  fn("set_model", "Choose the base model to train. Use a model name from the catalog, or any Hugging Face id (owner/name).",
    { query: { type: "string", description: "Model name or Hugging Face id, e.g. 'Qwen3.5-4B' or 'moonshotai/Kimi-K2.6'" } }, ["query"]),
  fn("set_trainer", "Switch which trainer runs the chosen model, when it's offered on more than one.",
    { trainer: { type: "string", enum: ["Tinker", "Fireworks", "Hugging Face"] } }, ["trainer"]),
  fn("set_dataset", "Use a Hugging Face dataset as the training data. It's checked automatically (free).",
    { id: { type: "string", description: "Hugging Face dataset id, owner/name" } }, ["id"]),
  fn("set_settings", "Change training parameters. Omit a key to leave it; use null to put it back to automatic.",
    { settings: { type: "object", description: `Any of: ${SETTING_KEYS.join(", ")}`, additionalProperties: true } }, ["settings"]),
  fn("open_tab", "Open a tab of the Train page.", { tab: { type: "string", enum: ["configure", "run", "history"] } }, ["tab"]),
  fn("spotlight", "Scroll to and highlight part of the page while you explain it.",
    { target: { type: "string", enum: ["model", "dataset", "parameters", "configuration", "preview", "tabs", "scorecard"] } }, ["target"]),
];

function fn(name: string, description: string, properties: Record<string, unknown>, required: string[]) {
  return { type: "function", function: { name, description, parameters: { type: "object", properties, required } } };
}

async function systemPrompt(page: unknown) {
  const models = await trainableModels().catch(() => []);
  const catalog = [...new Map(models.map((m) => [m.name, m])).values()]
    .map((m) => `${m.name} (${m.trainer ? TRAINERS[m.trainer].label : "?"}${m.size ? `, ${m.size}` : ""})`).join("; ");
  return `You are the guide inside ${BRAND.name}. ${BRAND.mission} Tagline: ${BRAND.tagline}

The app is one page, Train, with tabs Configure, Current Run and History. Configure has cards:
- Model: pick any open model; Method (LoRA, or QLoRA on Hugging Face only); Trainer (Tinker, Fireworks, or a GPU rented on the user's Hugging Face account running Unsloth).
- Dataset: any Hugging Face dataset with chat messages or question/answer columns (checked free as soon as it's picked), or upload .jsonl examples, or upload documents (PDF, Word, etc.) plus a sentence on what the model should learn, and we write training examples from them (a few cents, zero data retention).
- Parameters: Simple (project name, epochs or max steps, context length, learning rate) and Advanced (LoRA rank/alpha/dropout, target modules, RS-LoRA, optimizer, scheduler, batch size, grad accumulation, weight decay, warmup, seed, gradient checkpointing). Anything left on Auto is sized for the data. Controls a trainer can't honour are greyed out.
- Configuration: save or load all settings as YAML.
- Run preview (right-hand rail): steps, batch, context, learning rate, which trainer, where it runs, and the cost, updated live. Start Training is the only step that costs money.
Every dataset keeps ~10% of its examples back from training. Current Run shows progress, loss and learning-rate charts, the Scorecard (the original model, the trained one and a frontier model such as GPT-6 Astra, scored on those held-back questions, which is how you prove your model beats the frontier at one thing; scoring costs a little, and the user presses Score it), side-by-side chat for finished Tinker runs, and Publish to Hugging Face (a private repo with the LoRA adapter). History lists past runs.

Trainable models right now: ${catalog || "unavailable"}.

How to help:
- Be brief and friendly, in plain English. Explain ML terms simply when asked. Short paragraphs, no headings.
- When the user says what they want, set it up with the tools, then say what you set and what they still need to do.
- Use spotlight when pointing at a part of the page.
- You cannot start training or spend money. When everything is ready, tell them to check the cost in the Run preview and press Start Training themselves.
- Never invent prices; the Run preview shows the real cost once a dataset is checked.
- Nobody needs ML knowledge to use this. The weights they train are theirs to keep.

What's on the page right now (JSON): ${JSON.stringify(page).slice(0, 4000)}`;
}

async function call(model: string, key: string, messages: unknown[]) {
  const res = await fetch(OPENROUTER_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", "X-Title": BRAND.name },
    body: JSON.stringify({ model, messages, tools: TOOLS, provider: ZDR_PROVIDER, temperature: 0.3, max_tokens: 700 }),
    signal: AbortSignal.timeout(45_000),
  });
  const body = await res.json().catch(() => ({}));
  return { status: res.status, body };
}

export async function askGuide(history: GuideMessage[], page: unknown): Promise<GuideReply> {
  const env = await rootEnv();
  const key = process.env.OPENROUTER_API_KEY || env.OPENROUTER_API_KEY;
  if (!key) throw new Error("The guide needs OPENROUTER_API_KEY in .env. The tour still works.");
  const model = process.env.AGENT_MODEL || env.AGENT_MODEL || DEFAULT_MODEL;
  const messages = [{ role: "system", content: await systemPrompt(page) }, ...history.slice(-MAX_TURNS)];

  let { status, body } = await call(model, key, messages);
  if (status === 402 && model !== FREE_FALLBACK) ({ status, body } = await call(FREE_FALLBACK, key, messages));
  if (status !== 200) {
    const reason = status === 402 ? "the OpenRouter account has no credit"
      : status === 429 ? "the free model is busy right now; try again in a minute"
      : body?.error?.message ?? `OpenRouter answered ${status}`;
    throw new Error(`The guide can't answer: ${reason}. The tour still works.`);
  }

  const msg = body.choices?.[0]?.message ?? {};
  const actions = (msg.tool_calls ?? []).flatMap((c: { function?: { name?: string; arguments?: string } }) => {
    try {
      return c.function?.name ? [{ name: c.function.name, args: JSON.parse(c.function.arguments || "{}") }] : [];
    } catch {
      return [];
    }
  });
  return { reply: typeof msg.content === "string" ? msg.content.trim() : "", actions };
}
