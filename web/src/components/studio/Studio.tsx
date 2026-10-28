"use client";

import { useEffect, useState } from "react";
import { BRAND } from "@/lib/brand";
import { useRuns } from "@/lib/client";
import { onGuideAction, setPageState } from "@/lib/guide";
import Guide from "../Guide";
import { Logo } from "../icons";
import Configure from "./Configure";
import CurrentRun from "./CurrentRun";
import History from "./History";

export type Tab = "configure" | "run" | "history";

/** The Train page: Configure and start training, watch the current run, look back at History. */
export default function Studio({ initialTab, initialRun, initialDraft }: { initialTab: Tab; initialRun: string | null; initialDraft: string | null }) {
  const [tab, setTab] = useState<Tab>(initialTab);
  const [run, setRun] = useState<string | null>(initialRun);
  const [draft, setDraft] = useState<string | null>(initialDraft);
  const runs = useRuns(tab === "configure" ? 0 : 10_000);
  // "Current Run": the one opened, else whatever is training, else the latest started
  const current = run ?? runs?.find((r) => r.state === "training")?.id ?? runs?.find((r) => r.started)?.id ?? null;

  // Keep the address shareable and reload-safe
  useEffect(() => {
    const q = new URLSearchParams();
    if (tab !== "configure") q.set("tab", tab);
    if (tab === "run" && current) q.set("run", current);
    if (tab === "configure" && draft) q.set("draft", draft);
    const url = q.size ? `/?${q}` : "/";
    if (url !== location.pathname + location.search) window.history.replaceState(null, "", url);
  }, [tab, current, draft]);

  const open = (id: string) => { setRun(id); setTab("run"); };

  // The guide can switch tabs, and needs to know which one is showing
  useEffect(() => { setPageState({ tab, currentRun: tab === "run" ? current : null, runsStarted: runs?.filter((r) => r.started).length ?? 0 }); },
    [tab, current, runs]);
  useEffect(() => onGuideAction((a) => {
    if (a.type !== "open_tab") return null;
    if (a.tab === "run" && !current) return "There's no run to show yet.";
    setTab(a.tab);
    return `Opened ${a.tab === "run" ? "Current Run" : a.tab === "history" ? "History" : "Configure"}.`;
  }), [current]);
  const tabs: { key: Tab; label: string; disabled?: boolean }[] = [
    { key: "configure", label: "Configure" },
    { key: "run", label: "Current Run", disabled: !current },
    { key: "history", label: "History" },
  ];

  return (
    <main className="mx-auto w-full max-w-[1400px] px-4 pt-6 pb-16 sm:px-10">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted">
        <Logo className="size-5" /> <span className="text-ink">{BRAND.name}</span>
        <span className="text-faint">·</span> <span title={BRAND.mission}>{BRAND.tagline}</span>
      </div>
      <h1 className="mt-8 text-[44px] leading-none font-normal tracking-[-0.02em]">Train</h1>
      <p className="mt-3 text-[17px] text-muted">Configure and start training</p>

      <nav data-guide="tabs" className="mt-10 flex gap-10 border-b border-line" role="tablist" aria-label="Train">
        {tabs.map((t) => (
          <button key={t.key} type="button" role="tab" aria-selected={tab === t.key} disabled={t.disabled}
                  onClick={() => setTab(t.key)}
                  className="-mb-px border-b-2 border-transparent pb-4 text-[17px] text-muted transition hover:text-ink disabled:cursor-default disabled:text-faint disabled:hover:text-faint aria-selected:border-ink aria-selected:text-ink">
            {t.label}
          </button>
        ))}
      </nav>

      <div className="mt-12">
        <div hidden={tab !== "configure"}>
          <Configure active={tab === "configure"} initialDraft={initialDraft} onDraft={setDraft} onStarted={open} />
        </div>
        {tab === "run" && current && <CurrentRun key={current} id={current} />}
        {tab === "history" && <History runs={runs} onOpen={open} />}
      </div>
      <Guide />
    </main>
  );
}
