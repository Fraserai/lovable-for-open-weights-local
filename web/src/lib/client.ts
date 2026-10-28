"use client";

import { useEffect, useRef, useState } from "react";
import type { ModelOption } from "./models";
import type { RunStatus, RunSummary } from "./runs";

// Fetched once per page load; the server caches the trainer lists too.
let modelsCache: Promise<ModelOption[]> | null = null;
export const loadModels = () =>
  (modelsCache ??= fetch("/api/models").then((r) => r.json()).then((d: { trainable: ModelOption[] }) => d.trainable).catch(() => {
    modelsCache = null;
    return [];
  }));

/** A run, re-read every 2s while anything is still moving. `bump` restarts polling after an action;
 * `justLaunched` shows training as started while the trainer reports in. */
export function useRun(id: string | null, justLaunched = false) {
  const [run, setRun] = useState<RunStatus | null>(null);
  const [missing, setMissing] = useState(false);
  const [key, setKey] = useState(0);
  const graceUntil = useRef(0);

  // Declared before the poller so the first tick already sees it
  useEffect(() => {
    if (justLaunched) graceUntil.current = Date.now() + 120_000;
  }, [justLaunched]);

  useEffect(() => {
    if (!id) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      const res = await fetch(`/api/runs/${id}`, { cache: "no-store" }).catch(() => null);
      if (!alive) return;
      if (res?.status === 404) return setMissing(true);
      if (res?.ok) {
        let next: RunStatus = await res.json();
        // The trainer takes a few seconds to report in after Start: keep showing it as started
        const launching = next.train.state === "waiting" && Date.now() < graceUntil.current;
        if (launching) next = { ...next, train: { ...next.train, state: "running" } };
        setRun(next);
        const moving = launching || next.status === "running" || next.train.state === "running" || next.train.handback === "running"
          || next.scoring.state === "running";
        if (!moving) return;
      }
      timer = setTimeout(tick, 2000);
    };
    tick();
    return () => { alive = false; clearTimeout(timer); };
  }, [id, key]);

  return {
    run, missing,
    /** Poll again; `launched` marks training as just started for up to two minutes */
    bump: (launched = false) => {
      if (launched) graceUntil.current = Date.now() + 120_000;
      setKey((k) => k + 1);
    },
  };
}

/** Every run, for History and the sidebar. Refreshes when the page regains focus. */
export function useRuns(refreshMs = 0) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () => fetch("/api/runs", { cache: "no-store" }).then((r) => r.json())
      .then((d: RunSummary[]) => alive && setRuns(d)).catch(() => alive && setRuns((r) => r ?? []));
    load();
    window.addEventListener("focus", load);
    const t = refreshMs ? setInterval(load, refreshMs) : undefined;
    return () => { alive = false; window.removeEventListener("focus", load); clearInterval(t); };
  }, [refreshMs]);
  return runs;
}

/** "Qwen/Qwen3.5-4B" -> "Qwen3.5-4B"; "accounts/fireworks/models/glm-5p3-flash" -> "glm-5p3-flash" */
export const shortModel = (id: string | null) => (id ?? "").replace(/^unsloth:/, "").split("/").pop() || "—";

export const trainerOf = (id: string | null) =>
  !id ? null : id.startsWith("accounts/") ? "Fireworks" : id.startsWith("unsloth:") ? "Unsloth" : "Tinker";

export function formatBytes(n: number) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
}
