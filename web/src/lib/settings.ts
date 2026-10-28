import type { Trainer } from "./models";

// The Train page's settings, named exactly as the worker names them (worker/src/trainloop/settings.py).
// Keep SUPPORTED, METHODS and MAX_LORA_RANK in step with that file. Unset = sized for the data.
export type Settings = {
  project_name?: string;
  method?: "lora" | "qlora";
  max_steps?: number;
  epochs?: number;
  context_length?: number;
  learning_rate?: number;
  lora_rank?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  target_modules?: string[];
  lora_variant?: "lora" | "rslora";
  optimizer?: "adamw_8bit" | "adamw_torch";
  lr_scheduler?: "linear" | "cosine" | "constant";
  batch_size?: number;
  grad_accum?: number;
  weight_decay?: number;
  warmup_steps?: number;
  seed?: number;
  gradient_checkpointing?: "unsloth" | "true" | "false";
};
export type Key = keyof Settings;

export const TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

const ALL: Key[] = ["project_name", "method", "max_steps", "epochs", "context_length", "learning_rate", "lora_rank",
  "lora_alpha", "lora_dropout", "target_modules", "lora_variant", "optimizer", "lr_scheduler", "batch_size",
  "grad_accum", "weight_decay", "warmup_steps", "seed", "gradient_checkpointing"];

const SUPPORTED: Record<Trainer, Set<Key>> = {
  Tinker: new Set(["project_name", "method", "max_steps", "epochs", "context_length", "learning_rate", "lora_rank",
    "lr_scheduler", "batch_size"]),
  Fireworks: new Set(["project_name", "method", "epochs", "context_length", "learning_rate", "lora_rank"]),
  Unsloth: new Set(ALL),
};
export const METHODS: Record<Trainer, Set<string>> = {
  Tinker: new Set(["lora"]), Fireworks: new Set(["lora"]), Unsloth: new Set(["lora", "qlora"]),
};
export const MAX_LORA_RANK: Record<Trainer, number> = { Fireworks: 32, Tinker: 32, Unsloth: 256 };

/** What a control falls back to when the chosen trainer runs its own recipe (shown greyed). */
export const FIXED_BY: Partial<Record<Key, Partial<Record<Trainer, string>>>> = {
  optimizer: { Tinker: "Adam", Fireworks: "Set by Fireworks" },
  lr_scheduler: { Fireworks: "Set by Fireworks" },
  batch_size: { Fireworks: "Set by Fireworks" },
};

export const supports = (t: Trainer | null, k: Key) => !!t && SUPPORTED[t].has(k);

/** Only what's set, ready for `trainloop configure --json`. */
export function toWorker(s: Settings): Settings {
  return Object.fromEntries(Object.entries(s).filter(([, v]) => v !== undefined && v !== "" && v !== null)) as Settings;
}

export const LABELS: Record<Key, string> = {
  project_name: "project name", method: "method", max_steps: "max steps", epochs: "epochs",
  context_length: "context length", learning_rate: "learning rate", lora_rank: "LoRA rank", lora_alpha: "LoRA alpha",
  lora_dropout: "dropout", target_modules: "target modules", lora_variant: "LoRA variant", optimizer: "optimizer",
  lr_scheduler: "LR scheduler", batch_size: "batch size", grad_accum: "gradient accumulation",
  weight_decay: "weight decay", warmup_steps: "warmup steps", seed: "seed", gradient_checkpointing: "gradient checkpointing",
};
