"use client";

import { TRAINERS, type ModelOption } from "@/lib/models";
import type { TrainPlan } from "@/lib/runs";
import { LABELS, type Key } from "@/lib/settings";
import { Info, Rocket } from "../icons";
import { btn, Dot, ErrorNote, money, Pill, sci } from "../ui";

export type PreviewState = "empty" | "checking" | "planning" | "ready" | "blocked";

const owner = (id: string | null | undefined) => (id?.includes("/") ? id.split("/")[0] : null);

/** The sticky summary on the right: what will train, on what, with what, for how much. */
export default function RunPreview({ model, method, dataset, split, rows, plan, state, note, error, starting, onStart, action }: {
  model: ModelOption; method: "lora" | "qlora"; dataset: string | null; split: string | null; rows: number | null; plan: TrainPlan | null;
  state: PreviewState; note: string | null; error: string | null; starting: boolean; onStart: () => void;
  /** Replaces Start Training when the data needs a step first (documents: write examples) */
  action?: React.ReactNode;
}) {
  const trainer = model.trainer ? TRAINERS[model.trainer] : null;
  const pill = state === "ready" ? <Pill tone="accent">Ready</Pill>
    : state === "checking" || state === "planning" ? <Pill><Dot className="size-1.5" /> Checking</Pill>
    : <Pill>Not ready</Pill>;
  const cost = !plan ? "—"
    : plan.max_cost != null ? `at most ${money(plan.max_cost)}`
    : plan.est_cost != null ? `≈ ${money(plan.est_cost)}`
    : "billed by the trainer";

  const rowsList: [string, React.ReactNode][] = [
    ["Method", method === "qlora" && !plan?.ignored.includes("method")
      ? <>QLoRA <span className="text-faint">· 4-bit</span></> : <>LoRA <span className="text-faint">· 16-bit</span></>],
    ["Length", plan ? `${plan.steps.toLocaleString()} steps` : "—"],
    ["Batch", plan ? (plan.grad_accum > 1
      ? <>{plan.batch_size} <span className="text-faint">×</span> {plan.grad_accum} <span className="text-faint">=</span> {plan.batch_size * plan.grad_accum}</>
      : plan.batch_size) : "—"],
    ["Context", plan?.max_length ? plan.max_length.toLocaleString() : "—"],
    ["LR", plan?.learning_rate ? sci(plan.learning_rate) : "—"],
  ];

  return (
    <aside data-guide="preview" className="glass rounded-[32px] lg:sticky lg:top-6" aria-label="Run preview">
      <div className="p-7 sm:p-8">
        <div className="flex items-center justify-between">
          <p className="text-[15px] text-muted">Run preview</p>
          {pill}
        </div>

        <p className="mt-8 font-mono text-[13px] tracking-[0.08em] text-muted uppercase">{owner(model.hfRepo) ?? model.trainer ?? "Model"}</p>
        <p className="mt-1 text-[26px] leading-tight break-words">{model.name}</p>

        <p className="mt-6 font-mono text-[13px] tracking-[0.08em] text-muted uppercase">{owner(dataset) ?? (dataset ? "Local file" : "Dataset")}</p>
        <p className="mt-1 font-mono text-[15px] break-words">
          {dataset ? dataset.split("/").pop() : <span className="text-faint">None yet</span>}
          {split && <span className="text-faint"> · {split}</span>}
        </p>

        <dl className="mt-8 space-y-3.5 text-[15px]">
          {rowsList.map(([k, v]) => (
            <div key={k} className="flex items-baseline justify-between gap-4">
              <dt className="text-muted">{k}</dt>
              <dd className="text-right tabular-nums">{v}</dd>
            </div>
          ))}
        </dl>

        <dl className="mt-8 space-y-3.5 text-[15px]">
          <div className="flex items-baseline justify-between gap-4">
            <dt className="text-muted">Trainer</dt>
            <dd className="text-right">{trainer?.label ?? "—"}{plan?.gpu ? <span className="text-muted"> · {plan.gpu}</span> : null}</dd>
          </div>
          <div className="flex items-baseline justify-between gap-4">
            <dt className="whitespace-nowrap text-muted">Runs on</dt>
            <dd className="text-right">{trainer?.runsOn ?? "—"}</dd>
          </div>
          <div className="flex items-baseline justify-between gap-4">
            <dt className="text-muted">Cost</dt>
            <dd className="text-right tabular-nums">{cost}</dd>
          </div>
        </dl>

        <p className="mt-8 font-mono text-[13px] tracking-[0.08em] text-muted uppercase">Files</p>
        <dl className="mt-3 space-y-3 text-[15px]">
          <div className="flex items-center justify-between gap-4">
            <dt className="flex items-center gap-2 text-muted"><Info className="size-4 text-faint" /> Model</dt>
            <dd className="text-right">{trainer ? `Loaded by ${trainer.label}` : "—"}</dd>
          </div>
          <div className="flex items-center justify-between gap-4">
            <dt className="flex items-center gap-2 text-muted"><Info className="size-4 text-faint" /> Dataset</dt>
            <dd className="text-right">{rows != null ? `${rows.toLocaleString()} rows checked` : state === "checking" ? "Checking…" : "—"}</dd>
          </div>
        </dl>

        {plan && plan.ignored.length > 0 && (
          <p className="mt-6 text-sm text-amber-800">
            {trainer?.label} ignores: {plan.ignored.map((k) => LABELS[k as Key] ?? k).join(", ")}.
          </p>
        )}
        {plan && !plan.ok && <div className="mt-6"><ErrorNote>Some training rows failed the checks. Use different data.</ErrorNote></div>}
        {error && <div className="mt-6"><ErrorNote>{error}</ErrorNote></div>}
      </div>

      <div className="border-t border-line p-7 sm:p-8">
        {action ?? (
          <button type="button" onClick={onStart} disabled={state !== "ready" || starting} className={`${btn.primary} h-14 w-full text-[17px]`}>
            {state === "checking" ? <><Dot className="bg-white" /> Checking dataset…</>
              : state === "planning" ? <><Dot className="bg-white" /> Working out the cost…</>
              : <><Rocket className="size-5" /> {starting ? "Starting…" : "Start Training"}</>}
          </button>
        )}
        <p className="mt-3 text-center text-xs text-faint">{note ?? "Only training costs money. Nothing is spent until you press Start."}</p>
      </div>
    </aside>
  );
}
