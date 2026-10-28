"use client";

import { useState } from "react";
import { formatBytes, shortModel, trainerOf, useRun } from "@/lib/client";
import { TRAINERS, type Trainer } from "@/lib/models";
import type { RunStatus } from "@/lib/runs";
import { BRAND } from "@/lib/brand";
import Chart from "../Chart";
import Scorecard from "./Scorecard";
import { External } from "../icons";
import { btn, Dot, duration, ErrorNote, input, Pill, sci } from "../ui";

/** Current Run, laid out like Unsloth's: status and trainer panels, then live charts, then take the model home. */
export default function CurrentRun({ id }: { id: string }) {
  const { run, missing, bump } = useRun(id, true);
  if (missing) return <p className="text-muted">There&apos;s no run called {id}.</p>;
  if (!run) return <p className="flex items-center gap-2 text-muted"><Dot /> Loading…</p>;
  return <Body run={run} bump={bump} />;
}

function Body({ run, bump }: { run: RunStatus; bump: (launched?: boolean) => void }) {
  const trainer = trainerOf(run.baseModel) as Trainer | null;
  const t = run.train;
  const pct = t.totalSteps ? Math.min(100, (t.steps / t.totalSteps) * 100) : 0;
  const perEpoch = t.totalSteps && t.epochs ? t.totalSteps / t.epochs : null;
  const last = t.metrics.at(-1);
  const eta = t.elapsedSec && t.steps && t.totalSteps && t.state === "running" ? (t.elapsedSec / t.steps) * (t.totalSteps - t.steps) : null;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const act = async (path: "train" | "fetch", body?: object) => {
    setBusy(true);
    setError(null);
    const res = await fetch(`/api/runs/${run.id}/${path}`, {
      method: "POST", ...(body ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}),
    });
    if (!res.ok) setError((await res.json().catch(() => ({}))).error ?? "Could not start.");
    else bump(path === "train");
    setBusy(false);
  };

  const status = t.state === "done" ? <Pill tone="accent">Trained</Pill>
    : t.state === "failed" ? <Pill tone="bad">Stopped</Pill>
    : t.state === "running" ? <Pill tone="accent"><Dot className="size-1.5" /> Training</Pill>
    : <Pill>Not started</Pill>;

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_420px]">
        <section className="glass rounded-[32px] p-6 sm:p-8" aria-label="Status">
          <div className="flex flex-wrap items-start gap-3">
            <div className="min-w-0 flex-1">
              <p className="text-[13px] tracking-[0.08em] text-muted uppercase">{trainer ? TRAINERS[trainer].label : "Run"}</p>
              <h2 className="mt-1 text-2xl">{shortModel(run.baseModel)}</h2>
              <p className="mt-1 truncate font-mono text-sm text-muted">{run.sourceLabel ?? run.files.map((f) => f.name).join(", ")} · {run.rows?.toLocaleString()} examples</p>
            </div>
            {status}
          </div>

          <p className="mt-8 text-3xl font-light tabular-nums">Epoch {perEpoch ? (t.steps / perEpoch).toFixed(2) : "—"}</p>
          <div className="mt-4 h-2 overflow-hidden rounded-full bg-black/[0.08]" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100} aria-label="Training progress">
            <div className="h-full rounded-full bg-accent transition-[width] duration-700" style={{ width: `${t.state === "done" ? 100 : pct}%` }} />
          </div>
          <p className="mt-2 flex justify-between text-sm text-muted">
            <span>Step {t.steps.toLocaleString()}{t.totalSteps ? ` of ${t.totalSteps.toLocaleString()}` : ""}</span>
            <span>{t.state === "done" ? "100%" : `${pct.toFixed(0)}%`}</span>
          </p>

          <dl className="mt-8 grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-4">
            {([
              ["Loss", last ? last.loss.toFixed(4) : "—"],
              ["LR", last?.lr ? sci(last.lr) : "—"],
              ["Rank", t.loraRank ?? "—"],
              ["Method", "LoRA"],
              ["Elapsed", duration(t.elapsedSec)],
              ["ETA", t.state === "done" ? "—" : duration(eta)],
              ["Steps / sec", t.elapsedSec && t.steps ? (t.steps / t.elapsedSec).toFixed(2) : "—"],
              ["Epochs", t.epochs ?? "—"],
            ] as [string, React.ReactNode][]).map(([k, v]) => (
              <div key={k}>
                <dt className="text-sm text-muted">{k}</dt>
                <dd className="mt-1 font-mono text-lg tabular-nums">{v}</dd>
              </div>
            ))}
          </dl>

          {t.state === "running" && t.steps === 0 && (
            <p className="mt-6 text-sm text-muted">{t.jobUrl ? "Starting a cloud GPU and loading the model (a few minutes)…" : `Starting on ${trainer ? TRAINERS[trainer].label : "the trainer"}…`}</p>
          )}
          {t.state === "failed" && (
            <div className="mt-6 space-y-3">
              <ErrorNote>{t.error ?? "Training stopped. See train.log in the run folder."}</ErrorNote>
              <button type="button" onClick={() => act("train")} disabled={busy} className={btn.ghost}>{busy ? "Starting…" : "Try again"}</button>
            </div>
          )}
        </section>

        <section className="glass rounded-[32px] p-6 sm:p-8" aria-label="Trainer">
          <p className="text-[13px] tracking-[0.08em] text-muted uppercase">Trainer</p>
          <dl className="mt-5 space-y-4 text-[15px]">
            {([
              ["Trainer", trainer ? TRAINERS[trainer].label : "—"],
              ["Runs on", trainer ? TRAINERS[trainer].runsOn : "—"],
              ["Billing", trainer ? TRAINERS[trainer].billing : "—"],
              ["Run", <span key="r" className="font-mono text-sm">{run.id}</span>],
            ] as [string, React.ReactNode][]).map(([k, v]) => (
              <div key={k} className="flex items-baseline justify-between gap-4">
                <dt className="text-muted">{k}</dt>
                <dd className="text-right">{v}</dd>
              </div>
            ))}
          </dl>
          {t.jobUrl && (
            <a href={t.jobUrl} target="_blank" rel="noreferrer" className={`${btn.ghost} mt-6 w-full`}>Job on Hugging Face <External className="size-3.5" /></a>
          )}
          <p className="mt-6 text-sm text-faint">Runs in the cloud: closing this page doesn&apos;t stop it.</p>

          {t.state === "done" && (
            <div className="mt-8 border-t border-line pt-6">
              <p className="text-[13px] tracking-[0.08em] text-muted uppercase">Your model</p>
              <p className="mt-1 text-[15px]">{BRAND.lines[3]}</p>
              {t.localModel ? (
                <a href={`/api/runs/${run.id}/download`} download className={`${btn.primary} mt-4 w-full`}>
                  Download weights <span className="font-normal opacity-80">· {formatBytes(t.localModel.bytes)}</span>
                </a>
              ) : (
                <button type="button" onClick={() => act("fetch", { publish: false })} disabled={t.handback === "running" || busy}
                        className={`${btn.primary} mt-4 w-full`}>
                  {t.handback === "running" ? "Getting the weights…" : "Get the weights"}
                </button>
              )}
              <p className="mt-2 text-xs text-faint">
                A LoRA adapter for {run.baseModel?.replace(/^unsloth:/, "")}: load it on top of that model with PEFT, vLLM or Unsloth.
              </p>
              {t.hfRepo ? (
                <a href={t.hfRepo} target="_blank" rel="noreferrer" className={`${btn.ghost} mt-4 w-full`}>
                  {t.hfRepo.replace("https://huggingface.co/", "")} <External className="size-3.5" />
                </a>
              ) : (
                <>
                  <button type="button" onClick={() => act("fetch")} disabled={t.handback === "running" || busy} className={`${btn.ghost} mt-4 w-full`}>
                    {t.handback === "running" ? "Publishing…" : "Publish to Hugging Face"}
                  </button>
                  <p className="mt-2 text-xs text-faint">{t.handback === "failed" ? "That didn't finish. Try again; nothing gets retrained." : "The LoRA adapter, in a private repo on your account."}</p>
                </>
              )}
            </div>
          )}
          {error && <div className="mt-4"><ErrorNote>{error}</ErrorNote></div>}
        </section>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Chart title="Training Loss" smooth points={t.metrics.map((m) => ({ x: m.step, y: m.loss }))}
               empty={trainer === "Fireworks" ? "Fireworks reports metrics when the job ends." : undefined} />
        <Chart title="Learning Rate" format={sci} points={t.metrics.filter((m) => m.lr != null).map((m) => ({ x: m.step, y: m.lr! }))} />
      </div>

      <Scorecard run={run} bump={() => bump()} />

      {t.canChat && <TalkToModel id={run.id} model={shortModel(run.baseModel)} />}
    </div>
  );
}

type Turn = { q: string; original?: string; trained?: string; error?: string };

/** Ask the original model and the trained one the same question, answers side by side (Tinker runs). */
function TalkToModel({ id, model }: { id: string; model: string }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);

  const ask = async () => {
    const q = question.trim();
    if (!q || asking) return;
    setAsking(true);
    setQuestion("");
    setTurns((all) => [...all, { q }]);
    const settle = (patch: Partial<Turn>) => setTurns((all) => all.map((turn, i) => (i === all.length - 1 ? { ...turn, ...patch } : turn)));
    try {
      const res = await fetch(`/api/runs/${id}/chat`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: q }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error ?? "The models didn't answer.");
      settle({ original: body.original, trained: body.trained });
    } catch (e) {
      settle({ error: e instanceof Error ? e.message : "The models didn't answer." });
    } finally {
      setAsking(false);
    }
  };

  return (
    <section className="glass space-y-5 rounded-[32px] p-6 sm:p-8" aria-label="Talk to your model">
      <div>
        <p className="text-[13px] tracking-[0.08em] text-muted uppercase">Talk to your model</p>
        <p className="mt-1 text-[15px] text-muted">Ask anything. The original {model} and yours get the same question.</p>
      </div>

      {turns.map((turn, i) => (
        <div key={i} className="space-y-3">
          <p className="ml-auto w-fit max-w-[560px] rounded-3xl rounded-br-lg bg-accent px-5 py-3 text-white">{turn.q}</p>
          {turn.error ? <ErrorNote>{turn.error}</ErrorNote> : (
            <div className="grid gap-3 md:grid-cols-2">
              {([["Original", turn.original, false], ["Yours", turn.trained, true]] as const).map(([label, text, mine]) => (
                <figure key={label} className={`rounded-2xl border p-4 text-[15px] ${mine ? "border-accent/40 bg-white/70" : "border-white/70 bg-white/40"}`}>
                  <figcaption className={`mb-2 text-xs font-medium tracking-[0.08em] uppercase ${mine ? "text-accent" : "text-faint"}`}>{label}</figcaption>
                  {text === undefined
                    ? <p className="flex items-center gap-2 text-faint"><Dot /> Thinking…</p>
                    : <p className="leading-relaxed whitespace-pre-wrap">{text || "(no answer)"}</p>}
                </figure>
              ))}
            </div>
          )}
        </div>
      ))}

      <form onSubmit={(e) => { e.preventDefault(); ask(); }} className="flex gap-3">
        <label htmlFor="ask" className="sr-only">Ask your model</label>
        <input id="ask" value={question} onChange={(e) => setQuestion(e.target.value)} disabled={asking}
               placeholder={asking ? "Both models are answering (about 30 seconds)…" : "Ask your model something…"}
               className={`${input} min-w-0 flex-1 rounded-full`} />
        <button type="submit" disabled={asking || !question.trim()} className={btn.primary}>Ask</button>
      </form>
    </section>
  );
}
