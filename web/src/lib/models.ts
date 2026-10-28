// The live list comes from lib/catalog.ts (trainer support lists + Hugging Face search).
export type Trainer = "Fireworks" | "Tinker" | "Unsloth";

export type ModelOption = {
  id: string;
  name: string;
  trainer: Trainer | null; // null = on Hugging Face, but no connected trainer can fine-tune it yet
  hfRepo: string | null;
  note: string;
  ready: boolean; // the worker can train it end to end today
  size?: string | null; // "4B", "1T"
};

/** What the Train page starts on: small, cheap, and proven end to end (Banking77: 67.5% -> 82.5% for ~41c on Tinker). */
export const DEFAULT_MODEL: ModelOption = {
  id: "Qwen/Qwen3.5-4B",
  name: "Qwen3.5-4B",
  trainer: "Tinker",
  hfRepo: "Qwen/Qwen3.5-4B",
  note: "4B · 64K context · LoRA",
  ready: true,
  size: "4B",
};

/** What each trainer is, in the Run preview's words. Where training runs is what regulated customers ask first. */
export const TRAINERS: Record<Trainer, { label: string; runsOn: string; billing: string; env: string }> = {
  Tinker: {
    label: "Tinker",
    runsOn: "Thinking Machines cloud (US)",
    billing: "per training token",
    env: "TINKER_API_KEY",
  },
  Fireworks: {
    label: "Fireworks",
    runsOn: "Fireworks cloud (training region not selectable)",
    billing: "per training token",
    env: "FIREWORKS_API_KEY",
  },
  Unsloth: {
    label: "Hugging Face",
    runsOn: "A GPU on your Hugging Face account",
    billing: "per second of GPU time",
    env: "HF_TOKEN",
  },
};

/** The same open model can be tunable on several trainers; the picker shows it once with each as an option. */
export const modelKey = (m: ModelOption) => (m.hfRepo ?? m.id).toLowerCase();

// Matches what the worker can ingest today (worker/src/trainloop/ingest.py), plus ready-made datasets.
export const ACCEPTED_FILES = ".pdf,.docx,.pptx,.html,.md,.txt,.jsonl";

/** Frontier models the Scorecard can compare against (OpenRouter ids). */
export const FRONTIERS = [
  { id: "openai/gpt-6-astra", name: "GPT-6 Astra" },
  { id: "anthropic/claude-opus-5.5", name: "Claude Opus 5.5" },
  { id: "google/gemini-3.8-flash", name: "Gemini 3.8 Flash" },
  { id: "openai/gpt-6-sol", name: "GPT-6 Sol" },
];
