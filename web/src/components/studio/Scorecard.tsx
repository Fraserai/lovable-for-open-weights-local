"use client";

import { useEffect, useState } from "react";
import { BRAND } from "@/lib/brand";
import { shortModel, trainerOf } from "@/lib/client";
import { FRONTIERS, TRAINERS, type Trainer } from "@/lib/models";
import type { RunStatus } from "@/lib/runs";
import { Target } from "../icons";
import { btn, Card, Dot, ErrorNote, money, Select } from "../ui";

const NONE = "none";

/** Original vs yours vs a frontier model, on the questions held back from training. The proof. */
export default function Scorecard({ run, bump }: { run: RunStatus; bump: () => void }) {
  const card = run.scorecard;
  const [frontier, setFrontier] = useState(card?.frontierModel ?? FRONTIERS[0].id);
  const [estimate, setEstimate] = useState<number | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scoring = run.scoring.state === "running" || starting;
  const trainer = trainerOf(run.baseModel) as Trainer | null;
  const before = run.train.state === "waiting" || run.train.state === "failed"; // scoring the gap before training
  const name = (id: string | null) => FRONTIERS.find((f) => f.id === id)?.name ?? id ?? "";

  useEffect(() => {
    if (frontier === NONE || !run.testRows) return;
    let alive = true;
    fetch(`/api/runs/${run.id}/score?frontier=${encodeURIComponent(frontier)}`).then((r) => r.json())
      .then((b) => alive && setEstimate(typeof b.estimate === "number" ? b.estimate : null)).catch(() => {});
    return () => { alive = false; };
  }, [frontier, run.id, run.testRows]);

  const start = async () => {
    setStarting(true);
    setError(null);
    const res = await fetch(`/api/runs/${run.id}/score`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frontier: frontier === NONE ? null : frontier }),
    });
    if (!res.ok) setError((await res.json().catch(() => ({}))).error ?? "Could not start scoring.");
    bump();
    setStarting(false);
  };

  const c = card?.contenders ?? {};
  const rows = ([
    ["original", `Original · ${shortModel(run.baseModel)}`, "bg-black/25"],
    ["trained", "Yours", "bg-accent"],
    ["frontier", `Frontier · ${name(card?.frontierModel ?? null)}`, "bg-[#af52de]"],
  ] as const).filter(([k]) => c[k]);
  const pts = (a: number, b: number) => Math.round((a - b) * 1000) / 10;
  const verdict = c.trained && c.frontier
    ? c.trained.score >= c.frontier.score
      ? { line: BRAND.lines[2], text: `Yours beats ${name(card!.frontierModel)} by ${pts(c.trained.score, c.frontier.score)} points at this.` }
      : { line: null, text: `${name(card!.frontierModel)} leads by ${pts(c.frontier.score, c.trained.score)} points. More or better examples, more epochs or a bigger base model can close the gap.` }
    : c.trained && c.original
      ? { line: null, text: `${pts(c.trained.score, c.original.score) >= 0 ? "+" : ""}${pts(c.trained.score, c.original.score)} points over the original model.` }
      : c.original && c.frontier
        ? { line: null, text: `That's the gap to close before training: ${pts(c.frontier.score, c.original.score)} points.` }
        : null;

  return (
    <Card guide="scorecard" icon={<Target className="size-6" />} tone="#af52de" title="Scorecard"
          subtitle={run.testRows ? `${run.testRows.toLocaleString()} held-back questions your model never trained on` : "Your model against the frontier"}>
      {!run.testRows ? (
        <p className="text-[15px] text-muted">
          This run has no held-back test set (it was made before the Scorecard). Import the data again and every run
          keeps 10% back to score on.
        </p>
      ) : (
        <div className="space-y-6">
          {rows.length > 0 && (
            <div className="space-y-3">
              {rows.map(([k, label, tone]) => (
                <div key={k} className="grid grid-cols-[minmax(0,10rem)_1fr_4rem] items-center gap-4 sm:grid-cols-[minmax(0,14rem)_1fr_4rem]">
                  <span className={`truncate text-[15px] ${k === "trained" ? "font-medium" : "text-muted"}`}>{label}</span>
                  <span className="h-3 overflow-hidden rounded-full bg-black/[0.06]">
                    <span className={`block h-full rounded-full ${tone} transition-[width] duration-700`} style={{ width: `${c[k]!.score * 100}%` }} />
                  </span>
                  <span className="text-right font-mono text-[15px] tabular-nums">{(c[k]!.score * 100).toFixed(1)}%</span>
                </div>
              ))}
              {verdict && (
                <div className="pt-2">
                  {verdict.line && <p className="text-xl font-medium">{verdict.line}</p>}
                  <p className="text-[15px] text-muted">{verdict.text}</p>
                </div>
              )}
              <p className="text-xs text-faint">
                {card!.metric === "exact" ? "Marked by exact match." : "Marked by a judge model (GLM 5.3, zero data retention) against the reference answers."}
                {card!.notes.map((n) => ` ${n}`)}
              </p>
            </div>
          )}

          <div className={`flex flex-wrap items-end gap-3 ${rows.length ? "border-t border-black/5 pt-5" : ""}`}>
            <div className="min-w-[14rem] flex-1">
              <label htmlFor="frontier" className="mb-2 block text-[15px] text-muted">Compare against</label>
              <Select id="frontier" value={frontier} onChange={setFrontier} disabled={scoring}
                      options={[...FRONTIERS.map((f) => ({ value: f.id, label: f.name })), { value: NONE, label: "No frontier model" }]} />
            </div>
            <button type="button" onClick={start} disabled={scoring} className={`${btn.primary} h-12`}>
              {scoring ? <><Dot className="bg-white" /> Scoring…</> : card ? "Score again" : "Score it"}
            </button>
          </div>
          <p className="-mt-3 text-xs text-faint">
            {trainer === "Tinker" ? (before ? "Tinker answers as the original model" : "Tinker answers as the original and your model")
              : trainer === "Unsloth" ? (before ? "The original is scored on the GPU during training" : "Your Hugging Face GPU run already scored the original and yours")
              : `${trainer ? TRAINERS[trainer].label : "This trainer"} models can't answer here yet`}
            {frontier !== NONE && `; ${name(frontier)} costs ${estimate != null ? `up to ${money(estimate)}` : "a little"}`}.
            {before && " See the gap before you pay to train."}
            {!run.sourceInfo?.hf && frontier !== NONE && " Your data goes to zero-data-retention hosts only."}
          </p>
          {(error || run.scoring.error) && <ErrorNote>{error ?? run.scoring.error}</ErrorNote>}
        </div>
      )}
    </Card>
  );
}
