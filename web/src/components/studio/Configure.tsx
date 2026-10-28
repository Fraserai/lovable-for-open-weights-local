"use client";

import { useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
import { loadModels, useRun } from "@/lib/client";
import { onGuideAction, setPageState, type GuideAction } from "@/lib/guide";
import type { DatasetInfo } from "@/lib/hub";
import { DEFAULT_MODEL, TRAINERS, type ModelOption, type Trainer } from "@/lib/models";
import type { RunStatus, TrainPlan } from "@/lib/runs";
import { LABELS, MAX_LORA_RANK, METHODS, toWorker, type Key, type Settings } from "@/lib/settings";
import { Brain, FileCog, Server } from "../icons";
import ModelPicker, { trainerChoices } from "../ModelPicker";
import { btn, Card, Dot, Label, Select } from "../ui";
import DatasetCard, { isDocs, type Source } from "./DatasetCard";
import ParamsCard from "./ParamsCard";
import Scorecard from "./Scorecard";
import RunPreview, { type PreviewState } from "./RunPreview";

/** Identifies what a draft was built from, so a change re-imports and nothing else does. */
function sourceKey(s: Source | null, text: string) {
  if (!s) return null;
  if (s.kind === "saved") return "saved";
  if (s.kind === "hub") return JSON.stringify([s.id, s.config, s.split, s.limit, s.userField, s.assistantField, text]);
  return JSON.stringify([s.files.map((f) => [f.name, f.size, f.lastModified]), text]);
}

const METHOD_DOT: Record<string, string> = { qlora: "#34c759", lora: "#007aff", full: "#ff9500", cpt: "#af52de" };

export default function Configure({ active, initialDraft, onDraft, onStarted }: {
  active: boolean; initialDraft: string | null; onDraft: (id: string | null) => void; onStarted: (id: string) => void;
}) {
  const [model, setModel] = useState<ModelOption>(DEFAULT_MODEL);
  const [allModels, setAllModels] = useState<ModelOption[]>([]);
  const [connected, setConnected] = useState<Partial<Record<Trainer, boolean>>>({});
  const [source, setSource] = useState<Source | null>(null);
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [systemPrompt, setSystemPrompt] = useState("");
  const [docsGoal, setDocsGoal] = useState("");
  const [settings, setSettings] = useState<Settings>({});
  const [draftId, setDraftId] = useState<string | null>(initialDraft);
  const [importedKey, setImportedKey] = useState<string | null>(initialDraft ? "saved" : null);
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  const [plan, setPlan] = useState<TrainPlan | null>(null);
  const [planFor, setPlanFor] = useState<string | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const { run: draft, bump: bumpDraft } = useRun(draftId);

  useEffect(() => {
    loadModels().then(setAllModels);
    fetch("/api/trainers").then((r) => r.json())
      .then((list: { trainer: Trainer; connected: boolean }[]) => setConnected(Object.fromEntries(list.map((t) => [t.trainer, t.connected]))))
      .catch(() => {});
  }, []);

  // Reopened from the address (?draft=): take the model, settings and dataset from the checked run
  useEffect(() => {
    if (!initialDraft) return;
    let alive = true;
    Promise.all([fetch(`/api/runs/${initialDraft}`, { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)), loadModels()])
      .then(([run, models]: [RunStatus | null, ModelOption[]]) => {
        if (!alive || !run) return;
        const base = run.baseModel ?? "";
        const m = models.find((x) => x.id === base) ?? (base.startsWith("unsloth:")
          ? { id: base, name: base.slice(8).split("/").pop()!, trainer: "Unsloth" as const, hfRepo: base.slice(8), note: "cloud GPU · LoRA", ready: true }
          : null);
        if (m) setModel(m);
        setSettings(run.settings);
        const prompt = run.source === "dataset" ? run.systemPrompt ?? "" : "";
        setSystemPrompt(prompt);
        const hf = run.sourceInfo?.hf;
        const restoredSource: Source = hf
          ? { kind: "hub", id: hf, config: run.sourceInfo?.config ?? undefined, split: run.sourceInfo?.split ?? undefined, limit: run.sourceInfo?.limit ?? 1000 }
          : { kind: "saved", label: run.sourceLabel ?? run.files.map((f) => f.name).join(", ") };
        setSource(restoredSource);
        setImportedKey(sourceKey(restoredSource, prompt));
      });
    return () => { alive = false; };
  }, [initialDraft]);

  const docs = isDocs(source);
  const key = sourceKey(source, docs ? docsGoal.trim() : systemPrompt.trim());
  const hubReady = source?.kind !== "hub" || (!!info && (info.chat || !!info.guess || !!source.userField));

  const importData = async () => {
    if (!source || source.kind === "saved") return;
    setImporting(true);
    setImportError(null);
    const form = new FormData();
    form.set("goal", (docs ? docsGoal : systemPrompt).trim());
    form.set("model", model.id);
    if (source.kind === "hub") {
      form.set("hf", source.id);
      form.set("limit", String(source.limit));
      for (const k of ["config", "split", "userField", "assistantField"] as const) if (source[k]) form.set(k, source[k]!);
      if (!source.config && info) { form.set("config", info.picked.config); form.set("split", info.picked.split); }
    } else {
      source.files.forEach((f) => form.append("files", f));
    }
    try {
      const res = await fetch("/api/runs", { method: "POST", body: form });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error ?? "Could not read that data.");
      // The old draft was only ever checked, never trained: remove it
      if (draftId && draft?.train.state === "waiting") fetch(`/api/runs/${draftId}`, { method: "DELETE" }).catch(() => {});
      setDraftId(body.id);
      setImportedKey(key);
      onDraft(body.id);
    } catch (e) {
      setImportError(e instanceof Error ? e.message : "Could not read that data.");
    } finally {
      setImporting(false);
    }
  };

  // Datasets are checked as soon as they're picked (free); documents wait for "Write training examples"
  const autoImport = useEffectEvent(importData);
  useEffect(() => {
    if (!active || !source || docs || !hubReady || !model.ready || key === importedKey || importing) return;
    const t = setTimeout(autoImport, 700);
    return () => clearTimeout(t);
  }, [active, source, docs, hubReady, model.ready, key, importedKey, importing]);

  // Clearing the dataset forgets the draft
  const changeSource = (next: Source | null) => {
    setSource(next);
    if (next) return;
    setDraftId(null);
    setImportedKey(null);
    setPlan(null);
    onDraft(null);
  };

  const prepared = !!draftId && key === importedKey && draft?.status === "done"
    && (draft.train.state === "waiting" || draft.train.state === "failed");
  const planKey = JSON.stringify([draftId, model.id, toWorker(settings)]);

  // Re-plan (free) whenever the model or a setting changes on a checked dataset
  const planSeq = useRef(0);
  const checked = draft?.stages.check === "done";
  useEffect(() => {
    if (!active || !prepared || !draftId || !checked) return;
    const seq = ++planSeq.current;
    const t = setTimeout(async () => {
      try {
        const res = await fetch(`/api/runs/${draftId}/configure`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: model.id, settings: toWorker(settings) }),
        });
        const body = await res.json();
        if (seq !== planSeq.current) return;
        if (!res.ok) throw new Error(body.error ?? "Could not work out the cost.");
        setPlan(body);
        setPlanError(null);
      } catch (e) {
        if (seq !== planSeq.current) return;
        setPlan(null);
        setPlanError(e instanceof Error ? e.message : "Could not work out the cost.");
      }
      setPlanFor(planKey);
    }, 500);
    return () => clearTimeout(t);
  }, [active, prepared, draftId, checked, model.id, settings, planKey]);

  const start = async () => {
    if (!draftId) return;
    setStarting(true);
    setStartError(null);
    try {
      const res = await fetch(`/api/runs/${draftId}/train`, { method: "POST" });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error ?? "Could not start training.");
      onStarted(draftId);
      // The next run gets a fresh copy of the same dataset when you come back to Configure
      setDraftId(null);
      setImportedKey(null);
      setPlan(null);
      onDraft(null);
    } catch (e) {
      setStartError(e instanceof Error ? e.message : "Could not start training.");
    } finally {
      setStarting(false);
    }
  };

  const pickModel = (m: ModelOption) => {
    setModel(m);
    const max = m.trainer ? MAX_LORA_RANK[m.trainer] : 32;
    if ((settings.lora_rank ?? 0) > max) setSettings((s) => ({ ...s, lora_rank: max }));
    if (settings.method === "qlora" && m.trainer && !METHODS[m.trainer].has("qlora")) setSettings((s) => ({ ...s, method: undefined }));
  };

  const choices = useMemo(() => trainerChoices(model, allModels), [model, allModels]);
  const currentPlan = planFor === planKey ? plan : null;
  const state: PreviewState = !source ? "empty"
    : importing || draft?.status === "running" ? "checking"
    : docs && !prepared ? "blocked"
    : !prepared ? (draft?.status === "failed" || importError ? "blocked" : "checking")
    : draft?.stages.check !== "done" ? "blocked"
    : !currentPlan && !planError ? "planning"
    : currentPlan?.ok ? "ready" : "blocked";
  const trainer = model.trainer;
  const method = settings.method ?? "lora";

  // The guide (the logo button) sees what's set and can fill the form in. It can't start training.
  // Uploaded files are shared as a count only: their names and contents stay out of it.
  const datasetLabel = source?.kind === "hub" ? source.id : source?.kind === "files" ? `${source.files.length} uploaded file(s)` : source ? "a checked dataset" : null;
  useEffect(() => {
    setPageState({
      configure: {
        model: { name: model.name, id: model.id, trainer: trainer ? TRAINERS[trainer].label : null },
        alsoTrainableOn: choices.filter((c) => c.id !== model.id && c.ready && c.trainer).map((c) => TRAINERS[c.trainer!].label),
        dataset: datasetLabel, datasetKind: source?.kind ?? null, documentsNeedGoal: docs && !docsGoal.trim(),
        previewState: state, rows: prepared ? draft?.rows ?? null : null,
        settings: toWorker(settings),
        plan: currentPlan && { steps: currentPlan.steps, epochs: currentPlan.epochs, batch: currentPlan.batch_size,
          estCost: currentPlan.est_cost, maxCost: currentPlan.max_cost, ignoredSettings: currentPlan.ignored },
        problem: importError ?? planError ?? startError ?? null,
        trainerKeys: Object.fromEntries(Object.entries(connected).map(([t, ok]) => [TRAINERS[t as Trainer].label, ok])),
      },
    });
  }, [model, trainer, choices, datasetLabel, source?.kind, docs, docsGoal, state, prepared, draft?.rows, settings, currentPlan,
      importError, planError, startError, connected]);

  const handleGuide = useEffectEvent(async (a: GuideAction): Promise<string | null> => {
    if (a.type === "set_model") {
      const q = a.query.trim().toLowerCase();
      const all = allModels.length ? allModels : await loadModels();
      const ready = all.filter((m) => m.ready);
      let hit = ready.find((m) => [m.name, m.hfRepo ?? "", m.id].some((x) => x.toLowerCase() === q))
        ?? ready.find((m) => `${m.name} ${m.hfRepo ?? ""}`.toLowerCase().includes(q));
      if (!hit && /^[\w.-]+\/[\w.-]+$/.test(a.query.trim())) {
        const hub: ModelOption[] = await fetch(`/api/models?q=${encodeURIComponent(a.query.trim())}`).then((r) => r.json()).then((d) => d.hub).catch(() => []);
        hit = hub.find((m) => m.ready && m.hfRepo?.toLowerCase() === q);
      }
      if (!hit) return `Couldn't find a model called "${a.query}".`;
      const pick = trainerChoices(hit, all).find((c) => c.ready && c.trainer === model.trainer) ?? trainerChoices(hit, all).find((c) => c.ready) ?? hit;
      pickModel(pick);
      return `Model set to ${pick.name} on ${pick.trainer ? TRAINERS[pick.trainer].label : "a trainer"}.`;
    }
    if (a.type === "set_trainer") {
      const want = a.trainer.trim().toLowerCase();
      const c = choices.find((x) => x.ready && x.trainer && [TRAINERS[x.trainer].label, x.trainer].some((n) => n.toLowerCase() === want));
      if (!c) return `${model.name} can't train on ${a.trainer}.`;
      pickModel(c);
      return `Trainer set to ${TRAINERS[c.trainer!].label}.`;
    }
    if (a.type === "set_dataset") {
      const id = a.id.trim();
      if (!/^[\w.-]+\/[\w.-]+$/.test(id)) return `"${a.id}" isn't a Hugging Face dataset id.`;
      setInfo(null);
      changeSource({ kind: "hub", id, limit: 1000 });
      return `Dataset set to ${id}; checking it now.`;
    }
    if (a.type === "set_settings") {
      const patch: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(a.settings ?? {})) if (k in LABELS) patch[k] = v === null || v === "auto" ? undefined : v;
      if (!Object.keys(patch).length) return "No settings I recognise.";
      setSettings((cur) => ({ ...cur, ...patch }));
      return `Changed ${Object.keys(patch).map((k) => LABELS[k as Key]).join(", ")}.`;
    }
    return null;
  });
  useEffect(() => onGuideAction((a) => handleGuide(a)), []);

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_440px]">
      <div className="min-w-0 space-y-6">
        <Card guide="model" icon={<Brain className="size-6" />} title="Model" subtitle="Select model and training method">
          <div className="grid gap-5 @3xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)_minmax(0,1.1fr)] @xl:grid-cols-2">
            <div className="@xl:col-span-2 @3xl:col-span-1">
              <Label upper hint="Any open model. The Trainers tab lists models our trainers can tune; the Hugging Face tab finds any other, trained on a GPU rented from Hugging Face." htmlFor="model">Model</Label>
              <ModelPicker id="model" value={model} onChange={pickModel} />
            </div>
            <div>
              <Label upper hint="LoRA trains a small adapter on top of the model. QLoRA does the same over a 4-bit copy, to fit bigger models on one GPU." htmlFor="method">Method</Label>
              <Select id="method" value={method} lead={<span className="block size-2.5 rounded-full" style={{ background: METHOD_DOT[method] }} />}
                      onChange={(v) => setSettings((s) => ({ ...s, method: v === "lora" ? undefined : (v as "qlora") }))}
                      options={[
                        { value: "qlora", label: `QLoRA${trainer && METHODS[trainer].has("qlora") ? "" : " (Hugging Face only)"}`, disabled: !trainer || !METHODS[trainer].has("qlora") },
                        { value: "lora", label: "LoRA" },
                        { value: "full", label: "Full fine-tune (with Together, next)", disabled: true },
                        { value: "cpt", label: "Continued pretraining (not yet)", disabled: true },
                      ]} />
            </div>
            <div>
              <Label upper hint="Where the job runs. The same model can often train on more than one; we start with the one proven end to end." htmlFor="trainer">Trainer</Label>
              <Select id="trainer" value={model.id} lead={<Server className="size-4 text-accent" />}
                      onChange={(id) => { const m = choices.find((c) => c.id === id); if (m) pickModel(m); }}
                      options={choices.map((c) => ({
                        value: c.id,
                        label: `${c.trainer ? TRAINERS[c.trainer].label : "None"}${c.trainer && connected[c.trainer] === false ? " · no key" : ""}${c.ready ? "" : " · not connected"}`,
                        disabled: !c.ready,
                      }))} />
            </div>
          </div>
        </Card>

        <DatasetCard source={source} onSource={changeSource} info={info} onInfo={setInfo}
                     systemPrompt={systemPrompt} onSystemPrompt={setSystemPrompt} docsGoal={docsGoal} onDocsGoal={setDocsGoal}
                     draft={key === importedKey ? draft : null} />

        <ParamsCard s={settings} set={(patch) => setSettings((s) => ({ ...s, ...patch }))} trainer={trainer} plan={currentPlan} />

        {/* Benchmark first: the original model against the frontier on the held-back questions, before paying to train */}
        {prepared && draft?.testRows ? <Scorecard run={draft} bump={() => bumpDraft()} /> : null}

        <ConfigCard model={model} settings={settings} source={source} allModels={allModels}
                    onLoad={(m, s, hub) => { if (m) pickModel(m); setSettings(s); if (hub) changeSource(hub); }}
                    onReset={() => setSettings({})} />
      </div>

      <div>
        <RunPreview model={model} method={method}
                    dataset={source?.kind === "hub" ? source.id : source?.kind === "files" ? (source.files.length === 1 ? source.files[0].name : `${source.files.length} files`) : source?.label ?? null}
                    split={source?.kind === "hub" ? info?.picked.split ?? source.split ?? null : null}
                    rows={prepared ? draft?.rows ?? null : null} plan={currentPlan} state={state}
                    error={importError ?? (prepared ? planError : null) ?? startError}
                    note={trainer && connected[trainer] === false ? `Add ${TRAINERS[trainer].env} to .env to train on ${TRAINERS[trainer].label}.` : null}
                    starting={starting} onStart={start}
                    action={docs && !prepared ? (
                      <button type="button" onClick={importData} disabled={!docsGoal.trim() || importing || draft?.status === "running"} className={`${btn.primary} h-14 w-full text-[17px]`}>
                        {importing || draft?.status === "running" ? <><Dot className="bg-white" /> {draft?.stages.read === "done" ? "Writing examples…" : "Reading documents…"}</> : "Write training examples"}
                      </button>
                    ) : undefined} />
      </div>
    </div>
  );
}

// ---- Save / load the whole configuration as YAML, like Unsloth's configs ----

function toYaml(model: ModelOption, s: Settings, source: Source | null) {
  const lines = ["# Weightroom training config", `model: ${model.id}`, `trainer: ${model.trainer ?? "none"}`];
  for (const [k, v] of Object.entries(toWorker(s))) lines.push(`${k}: ${Array.isArray(v) ? `[${v.join(", ")}]` : v}`);
  if (source?.kind === "hub") {
    lines.push("dataset:", `  hf: ${source.id}`, `  limit: ${source.limit}`);
    for (const [k, v] of [["config", source.config], ["split", source.split], ["user_field", source.userField], ["assistant_field", source.assistantField]]) {
      if (v) lines.push(`  ${k}: ${v}`);
    }
  }
  return lines.join("\n") + "\n";
}

function fromYaml(text: string) {
  const top: Record<string, string> = {}, ds: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const m = line.match(/^(\s*)([a-z_]+):\s*(.*?)\s*$/);
    if (!m || line.trimStart().startsWith("#")) continue;
    (m[1] ? ds : top)[m[2]] = m[3];
  }
  const { model: modelId, ...rest } = top;
  delete rest.trainer;
  delete rest.dataset;
  const settings: Settings = {};
  for (const [k, raw] of Object.entries(rest)) {
    const v = raw.startsWith("[") ? raw.slice(1, -1).split(",").map((x) => x.trim()).filter(Boolean)
      : raw !== "" && Number.isFinite(Number(raw)) ? Number(raw) : raw;
    (settings as Record<string, unknown>)[k] = v;
  }
  const hub: Source | null = ds.hf ? { kind: "hub", id: ds.hf, limit: Number(ds.limit) || 1000, config: ds.config, split: ds.split, userField: ds.user_field, assistantField: ds.assistant_field } : null;
  return { modelId, settings, hub };
}

function ConfigCard({ model, settings, source, allModels, onLoad, onReset }: {
  model: ModelOption; settings: Settings; source: Source | null; allModels: ModelOption[];
  onLoad: (m: ModelOption | null, s: Settings, hub: Source | null) => void; onReset: () => void;
}) {
  const picker = useRef<HTMLInputElement>(null);
  const [note, setNote] = useState<string | null>(null);
  const save = () => {
    const url = URL.createObjectURL(new Blob([toYaml(model, settings, source)], { type: "text/yaml" }));
    const slug = (settings.project_name || model.name).replace(/\W+/g, "-").toLowerCase();
    Object.assign(document.createElement("a"), { href: url, download: `${slug}_${settings.method ?? "lora"}_${new Date().toISOString().slice(0, 10)}.yaml` }).click();
    URL.revokeObjectURL(url);
  };
  const load = async (f: File | undefined) => {
    if (!f) return;
    const { modelId, settings: s, hub } = fromYaml(await f.text());
    const m = modelId ? allModels.find((x) => x.id === modelId) ?? (modelId.startsWith("unsloth:")
      ? { id: modelId, name: modelId.slice(8).split("/").pop()!, trainer: "Unsloth" as const, hfRepo: modelId.slice(8), note: "cloud GPU · LoRA", ready: true } : null) : null;
    onLoad(m, s, hub);
    setNote(modelId && !m ? `Loaded the settings; ${modelId} isn't on a connected trainer, so the model is unchanged.` : `Loaded ${f.name}.`);
  };
  return (
    <Card guide="configuration" icon={<FileCog className="size-6" />} tone="#30b0c7" title="Configuration" subtitle="Save and load configurations">
      <div className="flex flex-wrap gap-3">
        <input ref={picker} type="file" accept=".yaml,.yml" className="hidden" aria-label="Load configuration" onChange={(e) => { load(e.target.files?.[0]); e.target.value = ""; }} />
        <button type="button" onClick={() => picker.current?.click()} className={btn.ghost}>Load YAML</button>
        <button type="button" onClick={save} className={btn.ghost}>Save YAML</button>
        <button type="button" onClick={() => { onReset(); setNote("Back to defaults: everything sized for your data."); }} className={btn.ghost}>Reset to defaults</button>
      </div>
      {note && <p className="mt-4 text-sm text-muted" aria-live="polite">{note}</p>}
    </Card>
  );
}
