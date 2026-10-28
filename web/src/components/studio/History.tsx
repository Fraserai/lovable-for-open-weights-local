"use client";

import { shortModel, trainerOf } from "@/lib/client";
import type { RunSummary } from "@/lib/runs";
import { ago, Pill } from "../ui";

const STATE: Record<RunSummary["state"], { label: string; tone: "muted" | "accent" | "warn" | "bad" }> = {
  preparing: { label: "Preparing", tone: "warn" },
  ready: { label: "Not started", tone: "muted" },
  training: { label: "Training", tone: "accent" },
  trained: { label: "Trained", tone: "accent" },
  failed: { label: "Stopped", tone: "bad" },
};

/** Every run that was started, newest first. */
export default function History({ runs, onOpen }: { runs: RunSummary[] | null; onOpen: (id: string) => void }) {
  const started = runs?.filter((r) => r.started);
  if (!started) return <p className="text-faint">Loading…</p>;
  if (started.length === 0) return <p className="glass rounded-[32px] p-10 text-center text-muted">No training runs yet. Configure one and press Start Training.</p>;
  return (
    <div className="glass overflow-hidden rounded-[32px]">
      <table className="w-full table-fixed text-left text-[15px]">
        <thead className="text-[13px] tracking-[0.08em] text-muted uppercase">
          <tr>
            <th className="px-6 py-4 font-normal">Run</th>
            <th className="hidden w-48 px-6 py-4 font-normal md:table-cell">Model</th>
            <th className="hidden w-28 px-6 py-4 font-normal sm:table-cell">Examples</th>
            <th className="w-36 px-6 py-4 font-normal">Status</th>
            <th className="hidden w-32 px-6 py-4 text-right font-normal sm:table-cell">Updated</th>
          </tr>
        </thead>
        <tbody>
          {started.map((r) => (
            <tr key={r.id} className="relative border-t border-line transition hover:bg-white/40">
              <td className="px-6 py-4">
                <button type="button" onClick={() => onOpen(r.id)} className="block w-full truncate text-left after:absolute after:inset-0">{r.goal || r.id}</button>
                <span className="block truncate font-mono text-xs text-faint">{r.id}</span>
              </td>
              <td className="hidden px-6 py-4 md:table-cell">
                <span className="block truncate">{shortModel(r.baseModel)}</span>
                <span className="text-xs text-faint">{trainerOf(r.baseModel)}</span>
              </td>
              <td className="hidden px-6 py-4 font-mono tabular-nums sm:table-cell">{r.rows?.toLocaleString() ?? "—"}</td>
              <td className="px-6 py-4"><Pill tone={STATE[r.state].tone}>{STATE[r.state].label}</Pill></td>
              <td className="hidden px-6 py-4 text-right text-muted sm:table-cell">{ago(r.updated)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
