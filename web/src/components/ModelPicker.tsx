"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { loadModels } from "@/lib/client";
import { modelKey, TRAINERS, type ModelOption, type Trainer } from "@/lib/models";
import { Check, Chevron, Chip, Search } from "./icons";
import { Tabs } from "./ui";

/** One open model and every trainer that can tune it. */
type ModelGroup = { key: string; name: string; size: string | null; options: ModelOption[] };

// When a model is on several trainers, start with the one proven end to end.
const PREFERENCE: Trainer[] = ["Tinker", "Fireworks", "Unsloth"];
const rank = (m: ModelOption) => (m.ready ? 0 : 10) + PREFERENCE.indexOf(m.trainer ?? "Unsloth");

function groupModels(models: ModelOption[]): ModelGroup[] {
  const groups = new Map<string, ModelGroup>();
  for (const m of models) {
    const k = modelKey(m);
    const g = groups.get(k);
    if (!g) groups.set(k, { key: k, name: m.name, size: m.size ?? null, options: [m] });
    else if (!g.options.some((o) => o.id === m.id)) g.options.push(m);
  }
  for (const g of groups.values()) g.options.sort((a, b) => rank(a) - rank(b));
  return [...groups.values()];
}

/** Every way to train the chosen model: its trainer listings, plus a rented Hugging Face GPU for any Hub model. */
export function trainerChoices(value: ModelOption, all: ModelOption[]): ModelOption[] {
  const k = modelKey(value);
  const same = all.filter((m) => modelKey(m) === k);
  const withValue = same.some((m) => m.id === value.id) ? same : [value, ...same];
  const unsloth = value.hfRepo && !withValue.some((m) => m.trainer === "Unsloth")
    ? [{ id: `unsloth:${value.hfRepo}`, name: value.name, trainer: "Unsloth" as const, hfRepo: value.hfRepo,
         note: "cloud GPU · LoRA", ready: true, size: value.size }]
    : [];
  return [...withValue, ...unsloth].sort((a, b) => rank(a) - rank(b));
}

function TrainerBadge({ m }: { m: ModelOption }) {
  if (!m.trainer) return null;
  return (
    <span className={`rounded-md px-1.5 py-0.5 text-[11px] font-medium ${m.ready ? "bg-accent/12 text-accent" : "bg-black/5 text-faint"}`}
          title={m.ready ? `Trains on ${TRAINERS[m.trainer].label}` : `${TRAINERS[m.trainer].label} isn't connected yet`}>
      {TRAINERS[m.trainer].label}
    </span>
  );
}

export default function ModelPicker({ value, onChange, id }: { value: ModelOption; onChange: (m: ModelOption) => void; id?: string }) {
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"trainers" | "hub">("trainers");
  const [query, setQuery] = useState("");
  const [trainable, setTrainable] = useState<ModelOption[] | null>(null);
  const [hub, setHub] = useState<ModelOption[] | null>(null);
  const root = useRef<HTMLDivElement>(null);
  const search = useRef<HTMLInputElement>(null);

  const toggle = () => {
    const next = !open;
    setOpen(next);
    if (next) loadModels().then(setTrainable);
  };

  useEffect(() => {
    if (open) search.current?.focus();
  }, [open, tab]);

  // Close on outside click / Escape
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // Hugging Face tab: search the whole Hub (most-downloaded text models when the box is empty)
  useEffect(() => {
    if (!open || tab !== "hub") return;
    const ctrl = new AbortController();
    const t = setTimeout(() => {
      fetch(`/api/models?q=${encodeURIComponent(query.trim() || "instruct")}`, { signal: ctrl.signal })
        .then((r) => r.json()).then((d: { hub: ModelOption[] }) => setHub(d.hub)).catch(() => {});
    }, 250);
    return () => { clearTimeout(t); ctrl.abort(); };
  }, [query, open, tab]);

  const q = query.trim().toLowerCase();
  const groups = useMemo(() => groupModels(trainable ?? []), [trainable]);
  const shown = tab === "trainers"
    ? groups.filter((g) => !q || `${g.name} ${g.key} ${g.options.map((o) => o.trainer).join(" ")}`.toLowerCase().includes(q))
    : groupModels(hub ?? []);
  const selectedKey = modelKey(value);

  const pick = (g: ModelGroup) => {
    // Keep the current trainer if this model is on it too
    onChange(g.options.find((o) => o.ready && o.trainer === value.trainer) ?? g.options.find((o) => o.ready) ?? g.options[0]);
    setOpen(false);
    setQuery("");
  };

  return (
    <div ref={root} className="relative">
      <button id={id} type="button" onClick={toggle} aria-haspopup="listbox" aria-expanded={open}
              className="flex h-12 w-full items-center gap-3 rounded-2xl border border-white/70 bg-field px-4 text-left transition hover:bg-white/80">
        <Chip className="size-5 shrink-0 text-muted" />
        <span className="min-w-0 flex-1 truncate text-[15px]">{value.name}</span>
        <Chevron className="size-5 shrink-0 text-muted" />
      </button>

      {open && (
        <div className="glass-pop absolute top-full left-0 z-30 mt-2 flex max-h-[min(520px,70vh)] w-[min(520px,calc(100vw-32px))] flex-col overflow-hidden rounded-3xl p-3">
          <Tabs label="Model source" value={tab} onChange={(t) => { setTab(t); setQuery(""); }} wide
                options={[{ value: "trainers", label: "Trainers" }, { value: "hub", label: "Hugging Face" }]} />
          <div className="mt-3 flex items-center gap-2 rounded-2xl bg-black/[0.05] px-4">
            <Search className="size-4 text-faint" />
            <label htmlFor="model-search" className="sr-only">Search models</label>
            <input id="model-search" ref={search} value={query} onChange={(e) => setQuery(e.target.value)}
                   placeholder={tab === "hub" ? "Search or paste a Hugging Face id…" : "Search models…"}
                   className="h-11 w-full bg-transparent text-[15px] outline-none placeholder:text-faint" />
          </div>

          <div role="listbox" aria-label="Base model" className="mt-2 overflow-y-auto">
            {(tab === "trainers" ? trainable : hub) === null && <p className="px-3 py-3 text-sm text-faint">Loading models…</p>}
            {shown.length === 0 && (tab === "trainers" ? trainable : hub) !== null && (
              <p className="px-3 py-3 text-sm text-faint">
                No match{tab === "trainers" ? ". Try the Hugging Face tab: any open model there trains on a Hugging Face GPU." : "."}
              </p>
            )}
            {shown.map((g) => {
              const ready = g.options.some((o) => o.ready);
              return (
                <button key={g.key} type="button" role="option" aria-selected={g.key === selectedKey} disabled={!ready} onClick={() => pick(g)}
                        className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition enabled:hover:bg-black/[0.05] disabled:cursor-default aria-selected:bg-accent/10">
                  <span className="min-w-0 flex-1">
                    <span className={`block truncate text-[15px] ${ready ? "" : "text-muted"}`}>{g.name}</span>
                    <span className="block truncate font-mono text-xs text-faint">{g.options[0].hfRepo ?? g.options[0].id}</span>
                  </span>
                  <span className="flex shrink-0 items-center gap-1">
                    {g.options.filter((o, i, all) => all.findIndex((x) => x.trainer === o.trainer) === i).map((o) => <TrainerBadge key={o.id} m={o} />)}
                  </span>
                  {g.size && <span className="w-12 shrink-0 text-right font-mono text-xs text-muted">{g.size}</span>}
                  {g.key === selectedKey && <Check className="size-4 shrink-0 text-accent" />}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
