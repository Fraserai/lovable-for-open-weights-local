"use client";

import { useState } from "react";
import { TRAINERS, type Trainer } from "@/lib/models";
import type { TrainPlan } from "@/lib/runs";
import { FIXED_BY, MAX_LORA_RANK, supports, TARGET_MODULES, type Key, type Settings } from "@/lib/settings";
import { Chevron, Info, Sliders } from "../icons";
import { Card, input, Label, Select, SliderRow, Tabs, sci } from "../ui";

const CONTEXTS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];

/** Parameters, after Unsloth's: Simple shows the four that matter; Advanced opens LoRA and hyperparameters.
 * Controls the chosen trainer can't honour are greyed and say so, like Unsloth's "Not supported on Apple Silicon". */
export default function ParamsCard({ s, set, trainer, plan }: {
  s: Settings; set: (patch: Partial<Settings>) => void; trainer: Trainer | null; plan: TrainPlan | null;
}) {
  const [mode, setMode] = useState<"simple" | "advanced">("simple");
  const [loraOpen, setLoraOpen] = useState(true);
  const [hyperOpen, setHyperOpen] = useState(true);
  const [hyperTab, setHyperTab] = useState<"optimization" | "schedule" | "memory">("optimization");
  const [useSteps, setUseSteps] = useState(s.max_steps !== undefined);

  const off = (k: Key) => !supports(trainer, k);
  const why = (k: Key) => (off(k) ? FIXED_BY[k]?.[trainer!] ?? `Not supported on ${trainer ? TRAINERS[trainer].label : "this trainer"}` : undefined);
  const maxRank = trainer ? MAX_LORA_RANK[trainer] : 32;
  const rank = s.lora_rank ?? plan?.lora_rank ?? 32;
  const steps = plan?.steps ?? 250;

  return (
    <Card guide="parameters" icon={<Sliders className="size-6" />} tone="#ff9500" title="Parameters" subtitle="Configure training parameters"
          action={<Tabs label="Detail" value={mode} onChange={setMode} options={[{ value: "simple", label: "Simple" }, { value: "advanced", label: "Advanced" }]} />}>
      <div className="grid gap-x-10 gap-y-7 @2xl:grid-cols-2">
        <div>
          <Label note="Optional" hint="Names the trained model's Hugging Face repo. Defaults to the run's name." htmlFor="project">Project Name</Label>
          <input id="project" value={s.project_name ?? ""} onChange={(e) => set({ project_name: e.target.value || undefined })}
                 placeholder="customer-support-lora" className={input} />
        </div>

        <div>
          {useSteps ? (
            <StepsOrEpochs label="Max Steps" hint="Stop after this many training steps, whatever the epochs." value={s.max_steps} auto={steps}
                           min={1} max={Math.max(2000, steps * 2)} disabled={off("max_steps")} note={why("max_steps")}
                           onChange={(v) => set({ max_steps: v })}
                           link={<button type="button" onClick={() => { setUseSteps(false); set({ max_steps: undefined }); }} className="text-[15px] text-accent underline underline-offset-4">Use Epochs</button>} />
          ) : (
            <StepsOrEpochs label="Epochs" hint="Passes over your data. Auto aims for about 250 steps, up to 4 passes." value={s.epochs} auto={plan?.epochs ?? 1}
                           min={1} max={10} disabled={off("epochs")} note={why("epochs")} onChange={(v) => set({ epochs: v })}
                           link={<button type="button" onClick={() => { setUseSteps(true); set({ epochs: undefined }); }} disabled={off("max_steps")}
                                         title={why("max_steps")} className="text-[15px] text-accent underline underline-offset-4 disabled:text-faint disabled:no-underline">Use Max Steps</button>} />
          )}
        </div>

        <div className={off("context_length") ? "opacity-45" : ""}>
          <Label hint="Longest example, in tokens. Longer examples are refused, never cut." htmlFor="context">Context Length</Label>
          <Select id="context" value={String(s.context_length ?? "auto")} disabled={off("context_length")}
                  onChange={(v) => set({ context_length: v === "auto" ? undefined : Number(v) })}
                  options={[{ value: "auto", label: plan?.max_length ? `Auto · ${plan.max_length.toLocaleString()}` : "Auto (model maximum)" },
                    ...CONTEXTS.map((c) => ({ value: String(c), label: c.toLocaleString() }))]} />
        </div>

        <div>
          <Label hint="Leave blank for the rate tuned for this model family." htmlFor="lr">Learning Rate</Label>
          <LrInput value={s.learning_rate} auto={plan?.learning_rate ?? null} onChange={(v) => set({ learning_rate: v })} />
        </div>
      </div>

      {mode === "advanced" && (
        <div className="mt-8 space-y-8">
          <Section title="LoRA Settings" open={loraOpen} onToggle={() => setLoraOpen(!loraOpen)}>
            <SliderRow label="Rank" hint="Adapter size. Higher learns more and costs more memory." value={s.lora_rank} auto={rank}
                       min={4} max={maxRank} step={4} onChange={(v) => set({ lora_rank: v })} disabled={off("lora_rank")} note={why("lora_rank")} />
            <SliderRow label="Alpha" hint="Scales the adapter's updates. Usually equal to the rank." value={s.lora_alpha} auto={s.lora_rank ?? rank}
                       min={4} max={256} step={4} onChange={(v) => set({ lora_alpha: v })} disabled={off("lora_alpha")} note={why("lora_alpha")} />
            <SliderRow label="Dropout" hint="Randomly drops adapter inputs while training to reduce overfitting." value={s.lora_dropout} auto={0}
                       min={0} max={0.5} step={0.01} onChange={(v) => set({ lora_dropout: v })} disabled={off("lora_dropout")} note={why("lora_dropout")}
                       format={(v) => v.toFixed(2)} />

            <div className={`py-3 ${off("target_modules") ? "opacity-45" : ""}`}>
              <Label hint="Which layers get an adapter. All of them is the usual choice." note={why("target_modules")}>Target Modules</Label>
              <div className="flex flex-wrap gap-2">
                {TARGET_MODULES.map((m) => {
                  const on = (s.target_modules ?? TARGET_MODULES).includes(m);
                  return (
                    <button key={m} type="button" aria-pressed={on} disabled={off("target_modules")}
                            onClick={() => {
                              const cur = s.target_modules ?? TARGET_MODULES;
                              const next = on ? cur.filter((x) => x !== m) : [...cur, m];
                              if (next.length) set({ target_modules: next.length === TARGET_MODULES.length ? undefined : next });
                            }}
                            className="rounded-full px-3.5 py-1.5 font-mono text-sm transition disabled:cursor-not-allowed aria-pressed:bg-white/80 aria-pressed:text-ink aria-pressed:shadow-sm not-aria-pressed:text-faint not-aria-pressed:line-through">
                      {m}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="grid gap-3 pt-3 @xl:grid-cols-2" role="radiogroup" aria-label="LoRA variant">
              <Variant title="Enable LoRA" sub="Train with LoRA" on={(s.lora_variant ?? "lora") === "lora"} onPick={() => set({ lora_variant: undefined })} />
              <Variant title="RS-LoRA" sub={why("lora_variant") ?? "Stable Rank"} on={s.lora_variant === "rslora"} disabled={off("lora_variant")}
                       onPick={() => set({ lora_variant: "rslora" })} />
              <Variant title="LoftQ" sub="Needs a quantised base; not offered" on={false} disabled onPick={() => {}} />
              <Variant title="DoRA" sub="Weight-Decomposed; not offered" on={false} disabled onPick={() => {}} />
            </div>
          </Section>

          <Section title="Training Hyperparameters" open={hyperOpen} onToggle={() => setHyperOpen(!hyperOpen)}>
            <div className="flex justify-center py-2">
              <Tabs label="Hyperparameters" wide value={hyperTab} onChange={setHyperTab}
                    options={[{ value: "optimization", label: "Optimization" }, { value: "schedule", label: "Schedule" }, { value: "memory", label: "Memory" }]} />
            </div>
            <div className="mt-4">
              {hyperTab === "optimization" && (
                <>
                  <Row label="Optimizer" hint="AdamW 8-bit saves memory with the same results." note={why("optimizer")} disabled={off("optimizer")}>
                    <Select value={s.optimizer ?? "adamw_8bit"} disabled={off("optimizer")} onChange={(v) => set({ optimizer: v === "adamw_8bit" ? undefined : v })}
                            options={[{ value: "adamw_8bit", label: "AdamW 8-bit" }, { value: "adamw_torch", label: "AdamW" }]} />
                  </Row>
                  <Row label="LR scheduler" hint="How the learning rate changes over the run." note={why("lr_scheduler")} disabled={off("lr_scheduler")}>
                    <Select value={s.lr_scheduler ?? "linear"} disabled={off("lr_scheduler")} onChange={(v) => set({ lr_scheduler: v === "linear" ? undefined : v })}
                            options={[{ value: "linear", label: "Linear" }, { value: "cosine", label: "Cosine" }, { value: "constant", label: "Constant" }]} />
                  </Row>
                  <SliderRow label="Batch Size" hint="Examples per step." value={s.batch_size} auto={plan?.batch_size ?? 4} min={1} max={64}
                             onChange={(v) => set({ batch_size: v })} disabled={off("batch_size")} note={why("batch_size")} />
                  <SliderRow label="Grad Accum" hint="Steps to add up before each update: a bigger effective batch without more memory." value={s.grad_accum}
                             auto={plan?.grad_accum ?? 1} min={1} max={32} onChange={(v) => set({ grad_accum: v })} disabled={off("grad_accum")} note={why("grad_accum")} />
                  <Row label="Weight Decay" hint="Pulls weights toward zero to reduce overfitting." note={why("weight_decay")} disabled={off("weight_decay")}>
                    <NumberInput value={s.weight_decay} auto="0" disabled={off("weight_decay")} onChange={(v) => set({ weight_decay: v })} />
                  </Row>
                </>
              )}
              {hyperTab === "schedule" && (
                <>
                  <SliderRow label="Warmup Steps" hint="Steps spent ramping the learning rate up at the start." value={s.warmup_steps}
                             auto={Math.max(1, Math.min(10, Math.floor(steps / 20)))} min={0} max={200}
                             onChange={(v) => set({ warmup_steps: v })} disabled={off("warmup_steps")} note={why("warmup_steps")} />
                  <Row label="Random Seed" hint="Same seed and data give the same run." note={why("seed")} disabled={off("seed")}>
                    <NumberInput value={s.seed} auto="3407" disabled={off("seed")} onChange={(v) => set({ seed: v === undefined ? undefined : Math.round(v) })} />
                  </Row>
                  <Row label="Train on Completions" hint="Learn only the assistant's replies, not the prompts. Always on: every trainer here trains on the replies.">
                    <p className="text-right text-[15px] text-muted">Always on</p>
                  </Row>
                </>
              )}
              {hyperTab === "memory" && (
                <>
                  <Row label="Gradient Checkpointing" hint="Recomputes activations to save memory. The default is the fastest." note={why("gradient_checkpointing")} disabled={off("gradient_checkpointing")}>
                    <Select value={s.gradient_checkpointing ?? "unsloth"} disabled={off("gradient_checkpointing")}
                            onChange={(v) => set({ gradient_checkpointing: v === "unsloth" ? undefined : v })}
                            options={[{ value: "unsloth", label: "Default" }, { value: "true", label: "On" }, { value: "false", label: "Off" }]} />
                  </Row>
                  <Row label="4-bit base (QLoRA)" hint="Set with Method in the Model card.">
                    <p className="text-right text-[15px] text-muted">{s.method === "qlora" ? "On" : "Off"}</p>
                  </Row>
                </>
              )}
            </div>
          </Section>
        </div>
      )}
    </Card>
  );
}

function StepsOrEpochs({ label, hint, value, auto, min, max, onChange, disabled, note, link }: {
  label: string; hint: string; value: number | undefined; auto: number; min: number; max: number;
  onChange: (v: number | undefined) => void; disabled: boolean; note?: string; link: React.ReactNode;
}) {
  const shown = value ?? auto;
  const pct = `${Math.max(0, Math.min(100, ((shown - min) / (max - min)) * 100))}%`;
  return (
    <div className={disabled ? "opacity-45" : ""}>
      <div className="mb-2.5 flex items-center gap-2">
        <label htmlFor="steps" className="text-[15px] text-muted">{label}</label>
        <span title={hint} aria-label={hint} className="cursor-help text-faint"><Info className="size-4" /></span>
        {value === undefined && !disabled && <span className="rounded-full bg-white/60 px-2 py-0.5 text-[11px] text-faint">Auto</span>}
        <span className="ml-auto flex items-center gap-4">
          {link}
          <input id="steps" inputMode="numeric" value={value ?? ""} placeholder={String(auto)} disabled={disabled}
                 onChange={(e) => { const n = Number(e.target.value); onChange(e.target.value.trim() && Number.isFinite(n) ? Math.max(min, Math.min(max, Math.round(n))) : undefined); }}
                 className="h-9 w-20 rounded-full border border-white/70 bg-field px-3 text-right font-mono text-sm outline-none placeholder:text-muted focus:ring-4 focus:ring-accent/15" />
        </span>
      </div>
      {note && <p className="-mt-1 mb-2 text-xs text-faint">{note}</p>}
      <input type="range" min={min} max={max} value={shown} disabled={disabled} aria-label={label}
             onChange={(e) => onChange(Number(e.target.value))}
             className={`slider mt-2 w-full ${value === undefined ? "auto" : ""}`} style={{ "--pct": pct } as React.CSSProperties} />
    </div>
  );
}

function LrInput({ value, auto, onChange }: { value: number | undefined; auto: number | null; onChange: (v: number | undefined) => void }) {
  const [text, setText] = useState(value !== undefined ? String(value) : "");
  const n = Number(text);
  const bad = text.trim() !== "" && !(Number.isFinite(n) && n > 0 && n < 0.1);
  return (
    <>
      <input id="lr" inputMode="decimal" value={text} placeholder={auto ? `Auto · ${sci(auto)}` : "Auto"}
             onChange={(e) => {
               setText(e.target.value);
               const v = Number(e.target.value);
               if (!e.target.value.trim()) onChange(undefined);
               else if (Number.isFinite(v) && v > 0 && v < 0.1) onChange(v);
             }}
             className={`${input} font-mono ${bad ? "ring-4 ring-red-400/30" : ""}`} />
      {bad && <p className="mt-1.5 text-xs text-red-600">Use a number like 0.0002 or 2e-4.</p>}
    </>
  );
}

function NumberInput({ value, auto, onChange, disabled }: { value: number | undefined; auto: string; onChange: (v: number | undefined) => void; disabled?: boolean }) {
  return (
    <input inputMode="decimal" value={value ?? ""} placeholder={auto} disabled={disabled}
           onChange={(e) => { const n = Number(e.target.value); onChange(e.target.value.trim() && Number.isFinite(n) ? n : undefined); }}
           className="ml-auto block h-12 w-full max-w-[200px] rounded-2xl border border-white/70 bg-field px-4 font-mono text-[15px] outline-none placeholder:text-muted focus:ring-4 focus:ring-accent/15 disabled:cursor-not-allowed" />
  );
}

function Row({ label, hint, note, disabled, children }: { label: string; hint: string; note?: string; disabled?: boolean; children: React.ReactNode }) {
  return (
    <div className={`grid grid-cols-[minmax(0,1fr)_minmax(0,320px)] items-center gap-6 py-3 ${disabled ? "opacity-45" : ""}`}>
      <div>
        <div className="flex items-center gap-2">
          <span className="text-[15px] text-muted">{label}</span>
          <span title={hint} aria-label={hint} className="cursor-help text-faint"><Info className="size-4" /></span>
        </div>
        {note && <p className="mt-0.5 text-xs text-faint">{note}</p>}
      </div>
      <div>{children}</div>
    </div>
  );
}

function Section({ title, open, onToggle, children }: { title: string; open: boolean; onToggle: () => void; children: React.ReactNode }) {
  return (
    <div>
      <button type="button" onClick={onToggle} aria-expanded={open} className="flex items-center gap-2 text-[17px] text-muted hover:text-ink">
        <Chevron className={`size-4 transition ${open ? "rotate-180" : ""}`} /> {title}
      </button>
      {open && <div className="mt-3">{children}</div>}
    </div>
  );
}

function Variant({ title, sub, on, disabled, onPick }: { title: string; sub: string; on: boolean; disabled?: boolean; onPick: () => void }) {
  return (
    <button type="button" role="radio" aria-checked={on} disabled={disabled} onClick={onPick}
            className="rounded-2xl border border-white/70 bg-field px-5 py-3.5 text-left transition enabled:hover:bg-white/80 disabled:cursor-not-allowed disabled:opacity-45 aria-checked:bg-white aria-checked:shadow-[0_4px_16px_-6px_rgb(0_0_0/0.2)] aria-checked:ring-2 aria-checked:ring-accent/60">
      <span className="block text-[15px]">{title}</span>
      <span className="block text-[13px] text-muted">{sub}</span>
    </button>
  );
}
