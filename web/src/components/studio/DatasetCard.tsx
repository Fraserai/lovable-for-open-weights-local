"use client";

import { useEffect, useRef, useState } from "react";
import { formatBytes } from "@/lib/client";
import type { DatasetHit, DatasetInfo, DatasetPage, DatasetSort } from "@/lib/hub";
import { ACCEPTED_FILES } from "@/lib/models";
import type { RunStatus } from "@/lib/runs";
import { Chevron, Close, CloudUp, Database, Doc, Eye, Hug, Search } from "../icons";
import { btn, Card, Dot, ErrorNote, input, Label, Pill, Select, Tabs } from "../ui";

export type Source =
  | { kind: "hub"; id: string; config?: string; split?: string; limit: number; userField?: string; assistantField?: string }
  | { kind: "files"; files: File[] }
  | { kind: "saved"; label: string }; // reopened from the address: the checked copy lives on the server

export const isDocs = (s: Source | null) =>
  s?.kind === "files" && s.files.some((f) => !f.name.toLowerCase().endsWith(".jsonl"));

const SEP = "\u0000";

/** Dataset card, after Unsloth's: pick a Hugging Face dataset or upload, choose subset and split, preview it. */
export default function DatasetCard({ source, onSource, info, onInfo, systemPrompt, onSystemPrompt, docsGoal, onDocsGoal, draft }: {
  source: Source | null; onSource: (s: Source | null) => void; info: DatasetInfo | null; onInfo: (i: DatasetInfo | null) => void;
  systemPrompt: string; onSystemPrompt: (v: string) => void; docsGoal: string; onDocsGoal: (v: string) => void;
  draft: RunStatus | null;
}) {
  const [infoError, setInfoError] = useState<string | null>(null);
  const [writer, setWriter] = useState<"openrouter" | "openai" | null>(null);
  const docs = isDocs(source);
  useEffect(() => {
    if (!docs || writer) return;
    fetch("/api/writer").then((r) => r.json()).then((b) => setWriter(b.provider)).catch(() => {});
  }, [docs, writer]);
  const [advanced, setAdvanced] = useState(false);
  const [viewing, setViewing] = useState(false);

  // Read subsets, splits, columns and sample rows through the dataset viewer (nothing downloaded)
  const hubKey = source?.kind === "hub" ? `${source.id}${SEP}${source.config ?? ""}${SEP}${source.split ?? ""}` : null;
  useEffect(() => {
    if (!hubKey) return;
    const [id, config, split] = hubKey.split(SEP);
    let alive = true;
    const qs = new URLSearchParams({ id, ...(config ? { config } : {}), ...(split ? { split } : {}) });
    fetch(`/api/datasets?${qs}`).then(async (r) => {
      const body = await r.json();
      if (!alive) return;
      if (!r.ok) throw new Error(body.error);
      onInfo(body);
      setInfoError(null);
    }).catch((e) => { if (alive) { onInfo(null); setInfoError(e instanceof Error ? e.message : "Couldn't read that dataset."); } });
    return () => { alive = false; };
  }, [hubKey, onInfo]);

  const hub = source?.kind === "hub" ? source : null;
  const configs = info ? [...new Set(info.splits.map((s) => s.config))] : [];
  const splits = info ? info.splits.filter((s) => s.config === info.picked.config).map((s) => s.split) : [];
  const mapping = hub?.userField && hub.assistantField ? `${hub.userField}${SEP}${hub.assistantField}` : "auto";
  const autoWorks = !!info && (info.chat || !!info.guess);

  return (
    <Card guide="dataset" icon={<Database className="size-6" />} tone="#ff2d55" title="Dataset" subtitle="Select or upload training data">
      <div className="grid gap-6 @2xl:grid-cols-2">
        <div>
          <Label upper hint="Any dataset on Hugging Face with chat messages or question/answer columns. Imported as-is." htmlFor="dataset">Dataset</Label>
          <HubSearch value={hub?.id ?? null} onPick={(id) => { onInfo(null); setInfoError(null); onSource({ kind: "hub", id, limit: 1000 }); }} />
        </div>
        <div>
          <Label upper hint="Your own examples (.jsonl), or documents we turn into training examples.">Or upload a local file</Label>
          <Upload onFiles={(files) => onSource({ kind: "files", files })} />
          <p className="mt-2.5 text-[13px] text-muted">JSONL, PDF, DOCX, PPTX, HTML, TXT, MD</p>
        </div>
      </div>

      {source && (
        <div className="mt-6 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-3xl border border-white/70 bg-white/35 px-5 py-4">
          <span className="grid size-10 shrink-0 place-items-center rounded-full bg-[#af52de]/15 text-[#8e3fc0]">
            {source.kind === "hub" ? <Hug className="size-5" /> : <Doc className="size-5" />}
          </span>
          <div className="min-w-[12rem] flex-1">
            <p className="truncate font-mono text-[17px]">
              {source.kind === "hub" ? source.id : source.kind === "saved" ? source.label
                : source.files.length === 1 ? source.files[0].name : `${source.files.length} files`}
            </p>
            <p className="text-[13px] text-muted">
              {source.kind === "hub" ? `Hugging Face Dataset${info ? ` / ${info.picked.config} / ${info.picked.split}` : ""}`
                : source.kind === "saved" ? "Checked copy on the server"
                : `Local ${isDocs(source) ? "documents" : "dataset"} · ${formatBytes(source.files.reduce((n, f) => n + f.size, 0))}`}
            </p>
          </div>
          {draft?.status === "done" && (draft.stages.check === "done"
            ? <Pill tone="accent">{draft.rows?.toLocaleString()} rows pass{draft.testRows ? ` · ${draft.testRows} held back to score` : ""}</Pill>
            : <Pill tone="bad">Rows failed checks</Pill>)}
          <button type="button" onClick={() => setViewing(true)} disabled={!info && !draft?.sample.length} className={`${btn.quiet} disabled:opacity-40`}>
            <Eye className="size-4" /> View dataset
          </button>
          <button type="button" onClick={() => { onSource(null); onInfo(null); setInfoError(null); }} className={btn.quiet}>Clear</button>
        </div>
      )}

      {infoError && <div className="mt-4"><ErrorNote>{infoError}</ErrorNote></div>}
      {draft?.status === "failed" && <div className="mt-4"><ErrorNote>{draft.error}</ErrorNote></div>}

      {isDocs(source) && (
        <div className="mt-6">
          <Label hint="We read your documents and write question/answer pairs that teach this. Uses an open model with zero data retention." htmlFor="docs-goal">
            What should the model learn from these?
          </Label>
          <textarea id="docs-goal" rows={2} value={docsGoal} onChange={(e) => onDocsGoal(e.target.value)}
                    placeholder="e.g. Answer member questions about our PDS and insurance terms, and stay inside general advice"
                    className={`${input} h-auto resize-y py-3 leading-relaxed`} />
          {writer === "openai" && (
            <p className="mt-3 rounded-2xl bg-amber-400/15 px-4 py-3 text-sm text-amber-900">
              Examples are written by OpenAI on this setup, which can keep data for up to 30 days. Use test or public
              documents only, or set <code className="font-mono">GENERATOR_PROVIDER=openrouter</code> in .env for zero data retention.
            </p>
          )}
        </div>
      )}

      {hub && info && (
        <div className="mt-6 grid gap-5 @2xl:grid-cols-3">
          <div>
            <Label hint="The dataset's configuration (Hugging Face calls it a subset)." htmlFor="subset">Subset</Label>
            <Select id="subset" value={info.picked.config} options={configs.map((c) => ({ value: c, label: c }))}
                    onChange={(c) => onSource({ ...hub, config: c, split: undefined, userField: undefined, assistantField: undefined })} />
          </div>
          <div>
            <Label hint="The rows to train on." htmlFor="train-split">Train Split</Label>
            <Select id="train-split" value={info.picked.split} options={splits.map((s) => ({ value: s, label: s }))}
                    onChange={(s) => onSource({ ...hub, config: info.picked.config, split: s })} />
          </div>
          <div>
            <Label hint="Held-out rows scored during training. Not wired to the trainers yet." htmlFor="eval-split">Evaluation Split</Label>
            <Select id="eval-split" value="none" disabled onChange={() => {}} options={[{ value: "none", label: "None" }]} />
          </div>
        </div>
      )}

      {source && (
        <div className="mt-6">
          <button type="button" onClick={() => setAdvanced(!advanced)} aria-expanded={advanced} className="flex items-center gap-2 text-[15px] text-muted hover:text-ink">
            <Chevron className={`size-4 transition ${advanced ? "" : "-rotate-90"}`} /> Advanced
          </button>
          {advanced && (
            <div className="mt-5 grid gap-5 @2xl:grid-cols-2">
              {hub && info && (
                <>
                  <div>
                    <Label hint="Which column is what the user says, and which is the reply the model should learn." htmlFor="mapping">Question → Answer</Label>
                    <Select id="mapping" value={mapping}
                            onChange={(v) => {
                              if (v === "auto") return onSource({ ...hub, userField: undefined, assistantField: undefined });
                              const [u, a] = v.split(SEP);
                              onSource({ ...hub, userField: u, assistantField: a });
                            }}
                            options={[
                              { value: "auto", label: info.chat ? "Chat messages (auto)" : info.guess ? `${info.guess.user} → ${info.guess.assistant} (auto)` : "Choose columns…", disabled: !autoWorks },
                              ...info.columns.flatMap((u) => info.columns.filter((a) => a !== u).map((a) => ({ value: `${u}${SEP}${a}`, label: `${u} → ${a}` }))),
                            ]} />
                  </div>
                  <div>
                    <Label hint="How many rows to import. More rows cost more to train." htmlFor="limit">Rows to use</Label>
                    <input id="limit" type="number" min={10} max={20000} step={100} value={hub.limit}
                           onChange={(e) => onSource({ ...hub, limit: Math.min(20000, Math.max(10, Number(e.target.value) || 1000)) })}
                           className={`${input} font-mono`} />
                  </div>
                </>
              )}
              {!isDocs(source) && (
                <div className="@2xl:col-span-2">
                  <Label note="Optional" hint="Added as the system message to every row that has none. Use the same one when you run the model." htmlFor="system">System prompt</Label>
                  <input id="system" value={systemPrompt} onChange={(e) => onSystemPrompt(e.target.value)}
                         placeholder="e.g. You are a member services assistant for an Australian super fund."
                         className={input} />
                </div>
              )}
            </div>
          )}
          {hub && info && !autoWorks && mapping === "auto" && (
            <p className="mt-3 text-sm text-amber-800">Open Advanced and pick which column is the question and which is the answer.</p>
          )}
        </div>
      )}

      {viewing && <Preview info={source?.kind === "hub" ? info : null} draft={draft} onClose={() => setViewing(false)} />}
    </Card>
  );
}

type Readiness = "ready" | "mapping" | "unreadable";
// One check per dataset per page load (the server caches the viewer's answers for an hour too)
const readiness = new Map<string, Promise<Readiness>>();
function checkReady(id: string): Promise<Readiness> {
  let p = readiness.get(id);
  if (!p) {
    p = fetch(`/api/datasets?id=${encodeURIComponent(id)}`)
      .then(async (r) => (r.ok ? ((await r.json()) as DatasetInfo) : null))
      .then((info): Readiness => (!info ? "unreadable" : info.chat || info.guess ? "ready" : "mapping"))
      .catch((): Readiness => "unreadable");
    readiness.set(id, p);
  }
  return p;
}

const READY_BADGE: Record<Readiness, { label: string; tone: string; title: string }> = {
  ready: { label: "Ready", tone: "bg-[#34c759]/15 text-[#1f8a3a]", title: "Has chat messages or question/answer columns: imports as-is." },
  mapping: { label: "Needs mapping", tone: "bg-amber-400/20 text-amber-800", title: "Pick which column is the question and which the answer, under Advanced." },
  unreadable: { label: "Can't preview", tone: "bg-black/5 text-faint", title: "Hugging Face's viewer can't read it (gated, private or not a table)." },
};

const SORTS: { value: DatasetSort; label: string }[] = [
  { value: "trending", label: "Trending" }, { value: "downloads", label: "Downloads" },
  { value: "likes", label: "Likes" }, { value: "updated", label: "Recent" },
];

const compact = (n: number) => Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(n);

/** The DATASET box: every dataset on Hugging Face, searchable and sortable, loading more as you scroll.
 * Rows are checked as they come into view and labelled Ready, Needs mapping or Can't preview. */
function HubSearch({ value, onPick }: { value: string | null; onPick: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<DatasetSort>("trending");
  const [hits, setHits] = useState<DatasetHit[] | null>(null);
  const [next, setNext] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const [ready, setReady] = useState<Record<string, Readiness>>({});
  const root = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLUListElement>(null);

  // First page whenever the search or sort changes
  useEffect(() => {
    if (!open) return;
    const ctrl = new AbortController();
    const t = setTimeout(() => {
      fetch(`/api/datasets?q=${encodeURIComponent(query.trim())}&sort=${sort}`, { signal: ctrl.signal })
        .then((r) => r.json()).then((page: DatasetPage) => { setHits(page.hits); setNext(page.next); list.current?.scrollTo({ top: 0 }); })
        .catch(() => {});
    }, 250);
    return () => { clearTimeout(t); ctrl.abort(); };
  }, [query, sort, open]);

  const loadMore = async () => {
    if (!next || loadingMore) return;
    setLoadingMore(true);
    const page: DatasetPage | null = await fetch(`/api/datasets?q=${encodeURIComponent(query.trim())}&sort=${sort}&cursor=${encodeURIComponent(next)}`)
      .then((r) => r.json()).catch(() => null);
    if (page) {
      setHits((h) => [...(h ?? []), ...page.hits.filter((x) => !(h ?? []).some((y) => y.id === x.id))]);
      setNext(page.next);
    }
    setLoadingMore(false);
  };

  // Check each row once it's on screen (every row would be hundreds of requests)
  useEffect(() => {
    const el = list.current;
    if (!open || !el) return;
    const seen = new IntersectionObserver((entries) => {
      for (const e of entries) {
        const id = (e.target as HTMLElement).dataset.id;
        if (!e.isIntersecting || !id) continue;
        seen.unobserve(e.target);
        checkReady(id).then((r) => setReady((all) => (all[id] === r ? all : { ...all, [id]: r })));
      }
    }, { root: el });
    el.querySelectorAll("[data-id]").forEach((row) => seen.observe(row));
    return () => seen.disconnect();
  }, [hits, open]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => root.current && !root.current.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDown); document.removeEventListener("keydown", onKey); };
  }, [open]);

  const pick = (id: string) => { onPick(id); setOpen(false); setQuery(""); };

  return (
    <div ref={root} className="relative">
      <button id="dataset" type="button" onClick={() => setOpen(!open)} aria-haspopup="listbox" aria-expanded={open}
              className="flex h-12 w-full items-center gap-3 rounded-2xl border border-white/70 bg-field px-4 text-left transition hover:bg-white/80">
        <Database className="size-5 shrink-0 text-muted" />
        <span className={`min-w-0 flex-1 truncate text-[15px] ${value ? "" : "text-faint"}`}>{value ? value.split("/").pop() : "Choose a Hugging Face dataset"}</span>
        <Chevron className="size-5 shrink-0 text-muted" />
      </button>
      {open && (
        <div className="glass-pop absolute top-full left-0 z-30 mt-2 w-[min(640px,calc(100vw-2rem))] rounded-3xl p-3">
          <div className="flex items-center gap-2 rounded-2xl bg-black/[0.05] px-4">
            <Search className="size-4 text-faint" />
            <label htmlFor="dataset-search" className="sr-only">Search Hugging Face datasets</label>
            <input id="dataset-search" autoFocus value={query} onChange={(e) => setQuery(e.target.value)}
                   onKeyDown={(e) => { if (e.key === "Enter" && /^[\w.-]+\/[\w.-]+$/.test(query.trim())) pick(query.trim()); }}
                   placeholder="Search all of Hugging Face, or paste owner/name…" className="h-11 w-full bg-transparent text-[15px] outline-none placeholder:text-faint" />
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-2 px-1">
            <Tabs label="Sort datasets" value={sort} onChange={setSort} options={SORTS} />
            <span className="ml-auto text-xs text-faint">
              {hits === null ? "" : `${hits.length.toLocaleString()} shown${next ? ", more as you scroll" : ""}`}
            </span>
          </div>
          <ul ref={list} role="listbox" aria-label="Datasets" className="mt-2 max-h-[min(24rem,50vh)] overflow-y-auto"
              onScroll={(e) => { const el = e.currentTarget; if (el.scrollHeight - el.scrollTop - el.clientHeight < 200) loadMore(); }}>
            {hits === null && <li className="px-3 py-2 text-sm text-faint">Loading…</li>}
            {hits?.length === 0 && <li className="px-3 py-2 text-sm text-faint">No match. Paste the full owner/name and press Enter.</li>}
            {hits?.map((h) => (
              <li key={h.id} data-id={h.id}>
                <button type="button" role="option" aria-selected={h.id === value} onClick={() => pick(h.id)}
                        disabled={ready[h.id] === "unreadable"} title={ready[h.id] && READY_BADGE[ready[h.id]].title}
                        className="flex w-full items-center gap-3 rounded-xl px-3 py-2 text-left enabled:hover:bg-black/[0.05] disabled:cursor-default disabled:opacity-50 aria-selected:bg-accent/10">
                  <span className="min-w-0 flex-1">
                    <span className="block truncate font-mono text-sm">{h.id}</span>
                    <span className="block truncate text-[11px] text-faint">
                      {[h.size && `${h.size} rows`, h.gated && "gated", `${compact(h.likes)} likes`].filter(Boolean).join(" · ")}
                    </span>
                  </span>
                  {ready[h.id]
                    ? <span className={`shrink-0 rounded-md px-1.5 py-0.5 text-[11px] font-medium ${READY_BADGE[ready[h.id]].tone}`}>{READY_BADGE[ready[h.id]].label}</span>
                    : <span className="shrink-0 text-[11px] text-faint">checking…</span>}
                  <span className="w-14 shrink-0 text-right text-xs text-faint">{compact(h.downloads)} ↓</span>
                </button>
              </li>
            ))}
            {next && (
              <li className="px-3 py-2">
                <button type="button" onClick={loadMore} disabled={loadingMore} className={`${btn.quiet} w-full justify-center`}>
                  {loadingMore ? <><Dot /> Loading more…</> : "Load more"}
                </button>
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}

function Upload({ onFiles }: { onFiles: (f: File[]) => void }) {
  const [dragging, setDragging] = useState(false);
  const picker = useRef<HTMLInputElement>(null);
  const take = (list: FileList | null) => {
    const allowed = ACCEPTED_FILES.split(",");
    const ok = Array.from(list ?? []).filter((f) => allowed.some((ext) => f.name.toLowerCase().endsWith(ext)));
    if (ok.length) onFiles(ok);
  };
  return (
    <>
      <input ref={picker} type="file" multiple accept={ACCEPTED_FILES} className="hidden" aria-label="Upload files"
             onChange={(e) => { take(e.target.files); e.target.value = ""; }} />
      <button type="button" onClick={() => picker.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); take(e.dataTransfer.files); }}
              className={`flex h-12 w-full items-center justify-center gap-2.5 rounded-2xl border border-dashed text-[15px] transition ${
                dragging ? "border-accent bg-accent/10" : "border-black/15 bg-white/30 hover:bg-white/60"}`}>
        <CloudUp className="size-5 text-muted" /> Drop file or click to upload
      </button>
    </>
  );
}

function Preview({ info, draft, onClose }: { info: DatasetInfo | null; draft: RunStatus | null; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/15 p-4 backdrop-blur-sm" onClick={onClose}>
      <div role="dialog" aria-modal="true" aria-label="Dataset preview" onClick={(e) => e.stopPropagation()}
           className="glass-pop flex max-h-[85vh] w-full max-w-4xl flex-col overflow-hidden rounded-[28px]">
        <div className="flex items-center justify-between border-b border-line px-6 py-4">
          <p className="text-[17px]">{draft?.sample.length ? "Training examples" : "First rows"}</p>
          <button type="button" onClick={onClose} aria-label="Close" className={btn.quiet}><Close className="size-5" /></button>
        </div>
        <div className="overflow-auto p-6">
          {draft?.sample.length ? (
            <div className="space-y-3">
              {draft.systemPrompt && <p className="text-sm text-muted">System: {draft.systemPrompt}</p>}
              {draft.sample.map((ex, i) => (
                <figure key={i} className="rounded-2xl border border-white/70 bg-white/60 p-4 text-[15px]">
                  <p className="text-xs tracking-[0.08em] text-faint uppercase">User</p>
                  <p className="mt-1 whitespace-pre-wrap">{ex.user}</p>
                  <p className="mt-4 text-xs tracking-[0.08em] text-faint uppercase">Assistant</p>
                  <p className="mt-1 whitespace-pre-wrap text-muted">{ex.assistant}</p>
                </figure>
              ))}
            </div>
          ) : info ? (
            <table className="w-full text-left text-sm">
              <thead className="text-faint"><tr>{info.columns.slice(0, 6).map((c) => <th key={c} className="px-3 py-2 font-normal">{c}</th>)}</tr></thead>
              <tbody>
                {info.rows.map((r, i) => (
                  <tr key={i} className="border-t border-line align-top">
                    {info.columns.slice(0, 6).map((c) => <td key={c} className="max-w-[280px] px-3 py-2"><p className="line-clamp-5">{r[c]}</p></td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p className="flex items-center gap-2 text-muted"><Dot /> Loading…</p>}
        </div>
      </div>
    </div>
  );
}
