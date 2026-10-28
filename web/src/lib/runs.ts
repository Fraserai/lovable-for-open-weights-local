import "server-only";

import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { closeSync, fstatSync, openSync } from "node:fs";
import { mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { toWorker, type Settings } from "./settings";

// The dev server runs from web/; the worker and runs/ folder sit beside it.
const ROOT = path.resolve(process.cwd(), "..");
const WORKER = path.join(ROOT, "worker");
const RUNS = path.join(ROOT, "runs");

// Plain, unbuffered output: the page reads it as it arrives (progress, and trainers saying they're out of credit).
const WORKER_ENV = { NO_COLOR: "1", PYTHONUNBUFFERED: "1" };

const MAX_POINTS = 400;

// First pass keeps generator spend small; raise once the loop is proven.
const FIRST_PASS_CHUNKS = "20";

export type Stage = "waiting" | "running" | "done" | "failed";

export type TrainPlan = {
  trainer: "fireworks" | "tinker" | "unsloth";
  base_model: string;
  examples: number;
  epochs: number;
  steps: number;
  tokens: number;
  lora_rank: number;
  learning_rate: number | null;
  batch_size: number;
  grad_accum: number;
  lr_scheduler: string | null;
  max_length?: number | null;
  /** Settings the chosen trainer can't honour (worker settings.split) */
  ignored: string[];
  est_cost: number | null;
  rate_per_million: number | null;
  // Unsloth: a rented GPU billed by the second; the job's time limit caps the bill at max_cost
  gpu?: string;
  per_hour?: number;
  max_cost?: number;
  ok: boolean;
  report: string[];
};

export type TrainProgress = {
  state: "waiting" | "running" | "done" | "failed";
  steps: number;
  totalSteps: number | null;
  error: string | null;
  hfRepo: string | null;
  handback: "idle" | "running" | "failed";
  canChat: boolean;
  epochs: number | null;
  loraRank: number | null;
  jobUrl: string | null;
  /** The trained adapter saved on this machine (runs/<id>/model), ready to download */
  localModel: { bytes: number } | null;
  /** One point per logged step, thinned to at most MAX_POINTS for the charts. */
  metrics: { step: number; loss: number; lr: number | null }[];
  elapsedSec: number | null;
};


/** A dataset on the Hugging Face Hub, imported as-is (no generator). */
export type HubSource = {
  id: string; config?: string; split?: string; limit: number; userField?: string; assistantField?: string;
};

/** trainloop score's result (runs/<id>/scorecard.json), without the answers themselves. */
export type Scorecard = {
  metric: "exact" | "judge"; examples: number; updated: string; baseModel: string; frontierModel: string | null;
  public: boolean; notes: string[];
  contenders: Partial<Record<"original" | "trained" | "frontier", { score: number; correct: number; reused?: boolean; from?: string }>>;
};

export type RunSummary = {
  id: string; goal: string | null; baseModel: string | null; rows: number | null;
  state: "preparing" | "ready" | "training" | "trained" | "failed"; updated: number;
  /** Training was started (drafts that were only checked are not) */
  started: boolean;
};

export type RunStatus = {
  id: string;
  goal: string | null;
  baseModel: string | null;
  source: "documents" | "dataset";
  /** The Hub dataset or file the examples came from, when imported as-is */
  sourceLabel: string | null;
  /** How the examples were imported (dataset runs): Hub id, subset, split, row limit, column mapping */
  sourceInfo: { hf: string | null; config: string | null; split: string | null; limit: number | null } | null;
  files: { name: string; size: number }[];
  status: "running" | "done" | "failed";
  stages: { read: Stage; write: Stage; check: Stage };
  train: TrainProgress;
  error: string | null;
  rows: number | null;
  estTokens: number | null;
  systemPrompt: string | null;
  sample: { user: string; assistant: string }[];
  report: string[];
  settings: Settings;
  /** Examples held back from training to score on (none for runs made before the Scorecard) */
  testRows: number | null;
  scorecard: Scorecard | null;
  scoring: { state: "idle" | "running" | "failed"; error: string | null };
};

// goal/baseModel: known before the worker writes run.json, so a preparing run can be shown
// pid: the worker process, so a restarted server can tell a step that's still going from one that died
type WebState = { status: RunStatus["status"]; exitCode: number | null; startedAt: string; goal?: string; baseModel?: string; pid?: number };

/** Starts a long worker step (writing examples, training, publishing) that outlives this server: its own
 * process group, output straight into a log file. A pipe back to us would break when the dev server stops,
 * and take a paid training run down with it. `onText` sees the step's new log output every 2s while we're up. */
function launch(args: string[], logFile: string, handlers: {
  onText?: (text: string, stop: () => void) => void;
  onExit?: (code: number | null, text: string) => void;
}) {
  const fd = openSync(logFile, "a");
  const from = fstatSync(fd).size; // only this step's output, not earlier attempts in the same log
  // Args array, no shell: user text can't be interpreted as a command.
  const child = spawn("uv", ["run", "--no-sync", "--directory", WORKER, "trainloop", ...args],
    { env: { ...process.env, ...WORKER_ENV }, detached: true, stdio: ["ignore", fd, fd] });
  closeSync(fd);
  child.unref();
  const read = () => readFile(logFile).then((b) => stripAnsi(b.subarray(from).toString()), () => "");
  const stop = () => { try { process.kill(-child.pid!, "SIGTERM"); } catch { /* already gone */ } };
  const timer = setInterval(() => { if (handlers.onText) read().then((t) => handlers.onText!(t, stop)); }, 2000);
  child.on("error", (e) => { clearInterval(timer); handlers.onExit?.(-1, `Could not start the worker: ${e.message}`); });
  child.on("exit", (code) => { clearInterval(timer); read().then((t) => { handlers.onText?.(t, stop); handlers.onExit?.(code, t); }); });
  return child.pid;
}

/** Is this worker process still running? (Signal 0 checks without touching it.) */
function alive(pid: number | undefined) {
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

const ID = /^[a-z0-9][a-z0-9-]{1,40}$/;

export function runDir(id: string) {
  if (!ID.test(id)) throw new Error("bad run id");
  return path.join(RUNS, id);
}

export function newRunId(goal: string) {
  const slug = goal.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 30).replace(/-+$/, "");
  return `${slug || "run"}-${randomBytes(3).toString("hex")}`;
}

export function safeFileName(name: string) {
  const base = path.basename(name).replace(/[^\w.\- ]+/g, "_").trim();
  return base && !base.startsWith(".") ? base : `upload-${randomBytes(3).toString("hex")}`;
}

export async function startPrepare(id: string, goal: string, baseModel: string, files: File[], hub?: HubSource) {
  const dir = runDir(id);
  const uploads = path.join(dir, "uploads");
  await mkdir(uploads, { recursive: true });
  for (const f of files) {
    await writeFile(path.join(uploads, safeFileName(f.name)), Buffer.from(await f.arrayBuffer()));
  }

  const state: WebState = { status: "running", exitCode: null, startedAt: new Date().toISOString(), goal, baseModel };
  await writeFile(path.join(dir, "web.json"), JSON.stringify(state));

  // A ready-made dataset trains as-is; documents have to be read and turned into examples first.
  const datasets = files.filter((f) => f.name.toLowerCase().endsWith(".jsonl"));
  const common = ["--run", id, "--goal", goal, "--base-model", baseModel];
  const args = hub
    ? ["dataset", ...common, "--hf", hub.id, "--limit", String(hub.limit),
       ...(hub.config ? ["--config", hub.config] : []), ...(hub.split ? ["--split", hub.split] : []),
       ...(hub.userField && hub.assistantField ? ["--user-field", hub.userField, "--assistant-field", hub.assistantField] : [])]
    : datasets.length === files.length
    ? ["dataset", ...common, "--file", path.join(uploads, safeFileName(datasets[0].name))]
    : ["prepare", uploads, ...common, "--max-chunks", FIRST_PASS_CHUNKS];

  const finish = (code: number | null) => {
    state.status = code === 0 ? "done" : "failed";
    state.exitCode = code;
    writeFile(path.join(dir, "web.json"), JSON.stringify(state)).catch(() => {});
  };
  state.pid = launch(args, path.join(dir, "prepare.log"), { onExit: finish });
  await writeFile(path.join(dir, "web.json"), JSON.stringify(state));
}

/** A trainer account with no credit. The trainers' SDKs retry on that indefinitely, so spot it in the
 * worker's output and stop, with a message the page can show. */
function outOfCredit(text: string): string | null {
  if (!/billing status|Error code: 402|Insufficient credits|no prepaid credit|payment required/i.test(text)) return null;
  if (/thinkingmachines|tinker/i.test(text)) return "Tinker is out of credit. Add some at https://tinker.thinkingmachines.ai/billing/balance, then try again.";
  if (/fireworks/i.test(text)) return "Fireworks is out of credit. Add some in your Fireworks billing settings, then try again.";
  if (/huggingface/i.test(text)) return "Your Hugging Face account has no prepaid credit for GPU jobs. Add some at https://huggingface.co/settings/billing, then try again.";
  return "The trainer's account is out of credit. Add some, then try again.";
}

/** Runs a worker command, returning its stdout. Used for the free planning step. */
function worker(args: string[], timeoutMs = 120_000): Promise<string> {
  return new Promise((resolve, reject) => {
    const child = spawn("uv", ["run", "--no-sync", "--directory", WORKER, "trainloop", ...args],
      { env: { ...process.env, ...WORKER_ENV } });
    let out = "", err = "";
    const timer = setTimeout(() => { child.kill(); reject(new Error("worker timed out")); }, timeoutMs);
    const watch = () => {
      const broke = outOfCredit(err + out);
      if (!broke) return;
      clearTimeout(timer);
      child.kill();
      reject(new Error(broke));
    };
    child.stdout.on("data", (d) => { out += d; watch(); });
    child.stderr.on("data", (d) => { err += d; watch(); });
    child.on("error", reject);
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) resolve(out);
      else reject(new Error(stripAnsi(err || out).trim().split("\n").pop() || `worker exited ${code}`));
    });
  });
}

/** What training would cost and how it would run. Spends nothing. */
export async function getPlan(id: string): Promise<TrainPlan> {
  runDir(id);
  return JSON.parse(await worker(["plan", "--run", id, "--json"]));
}

/** Sets the model and the Parameters card's settings on a prepared run, then returns the new plan. Spends nothing. */
export async function configure(id: string, baseModel: string, settings: Settings): Promise<TrainPlan> {
  runDir(id);
  await worker(["configure", "--run", id, "--base-model", baseModel, "--json", JSON.stringify(toWorker(settings))]);
  return getPlan(id);
}

/** Removes a run that never started training (a dataset picked, then replaced). */
export async function deleteDraft(id: string): Promise<boolean> {
  const dir = runDir(id);
  const run = await readOr<Record<string, unknown> | null>(path.join(dir, "run.json"), JSON.parse, null);
  if (run?.job_state || run?.job) return false;
  await rm(dir, { recursive: true, force: true });
  return true;
}

/** Starts the paid training run in the background; progress is read back by getStatus. */
export async function startTraining(id: string): Promise<void> {
  const dir = runDir(id);
  const launchFile = path.join(dir, "launch.json");
  await rm(launchFile, { force: true });
  let stopped: string | null = null;
  // If the worker stops before the trainer takes the job (a missing key, say), say why instead of waiting forever
  const failed = (error: string) => writeFile(launchFile, JSON.stringify({ error })).catch(() => {});
  launch(["train", "--run", id, "--yes"], path.join(dir, "train.log"), {
    onText: (text, stop) => {
      const broke = !stopped && outOfCredit(text.slice(-8000));
      if (!broke) return;
      stopped = broke;
      stop();
      failed(broke);
      markFailed(dir, broke);
    },
    onExit: (code, text) => { if (code !== 0 && !stopped) failed(lastError(text.split("\n"))); },
  });
}

/** A run the worker had marked as training, but that we stopped: record why, so it doesn't show "Training" forever. */
async function markFailed(dir: string, error: string) {
  const file = path.join(dir, "run.json");
  const run = await readOr<Record<string, unknown> | null>(file, JSON.parse, null);
  if (run?.job_state !== "JOB_STATE_RUNNING") return;
  await writeFile(file, JSON.stringify({ ...run, job_state: "JOB_STATE_FAILED", error }, null, 2)).catch(() => {});
}

async function trainProgress(dir: string, run: Record<string, unknown> | null): Promise<TrainProgress> {
  const jobState = typeof run?.job_state === "string" ? run.job_state : null;
  const unsloth = run?.job === "unsloth";
  const plan = ((unsloth ? run?.unsloth_plan : run?.tinker_plan) ?? {}) as { steps?: number; epochs?: number; lora_rank?: number };
  const metricsFile = unsloth ? path.join(dir, "unsloth", "metrics.jsonl") : path.join(dir, "tinker", "logs", "metrics.jsonl");
  const lines = (await readOr(metricsFile, (s) => s, "")).split("\n").filter(Boolean);
  const steps = lines.length;
  const points = lines.flatMap((l) => {
    try {
      const m = JSON.parse(l);
      // Tinker logs mean per-token loss as train_mean_nll; Unsloth logs the trainer's loss
      const loss = m.train_mean_nll ?? m.loss;
      return typeof loss === "number" ? [{ step: m.step as number, loss, lr: m.learning_rate ?? null, sec: m["time/step"] ?? null }] : [];
    } catch {
      return [];
    }
  });
  const stride = Math.ceil(points.length / MAX_POINTS);
  const secs = points.map((p) => p.sec).filter((x): x is number => typeof x === "number");
  // Set when the worker stopped before any job existed (see startTraining)
  const launch = jobState ? null : await readOr<{ error?: string } | null>(path.join(dir, "launch.json"), JSON.parse, null);
  const state: TrainProgress["state"] =
    jobState === "JOB_STATE_COMPLETED" ? "done"
    : jobState === "JOB_STATE_FAILED" ? "failed"
    : jobState ? "running"
    : launch ? "failed"
    : "waiting";
  const hfRepo = typeof run?.hf_repo === "string" ? run.hf_repo : null;
  const fetch = await readOr<{ status: "running" | "done" | "failed"; pid?: number } | null>(path.join(dir, "fetch.json"), JSON.parse, null);
  // Publishing that died (the server stopped, or it crashed) without saying so: done if the repo exists, else failed
  if (fetch?.status === "running" && fetch.pid && !alive(fetch.pid)) fetch.status = hfRepo ? "done" : "failed";
  return {
    state,
    steps,
    totalSteps: plan.steps ?? null,
    error: typeof run?.error === "string" ? run.error : launch?.error ?? null,
    hfRepo,
    handback: hfRepo ? "idle" : fetch?.status === "running" ? "running" : fetch?.status === "failed" ? "failed" : "idle",
    canChat: state === "done" && run?.job === "tinker",
    epochs: typeof run?.epochs === "number" ? run.epochs : (plan as { epochs?: number }).epochs ?? null,
    loraRank: typeof run?.lora_rank === "number" ? run.lora_rank : (plan as { lora_rank?: number }).lora_rank ?? null,
    jobUrl: typeof run?.job_url === "string" ? run.job_url : null,
    localModel: await localModel(dir),
    metrics: points.filter((_, i) => i % stride === 0 || i === points.length - 1).map(({ step, loss, lr }) => ({ step, loss, lr })),
    elapsedSec: secs.length ? Math.round(secs.reduce((a, b) => a + b, 0)) : null,
  };
}

/** The downloaded adapter's total size, if the run has one on disk. */
async function localModel(dir: string): Promise<{ bytes: number } | null> {
  const names = await readdir(path.join(dir, "model")).catch(() => [] as string[]);
  if (!names.some((n) => n.endsWith(".safetensors"))) return null;
  const sizes = await Promise.all(names.map((n) => stat(path.join(dir, "model", n)).then((s) => s.size, () => 0)));
  return { bytes: sizes.reduce((a, b) => a + b, 0) };
}

/** The adapter folder as a .tar.gz stream (Mac and Windows both open it with a double-click). */
export function modelArchive(id: string): ReadableStream<Uint8Array> {
  const child = spawn("tar", ["-czf", "-", "-C", runDir(id), "model"], { stdio: ["ignore", "pipe", "ignore"] });
  return new ReadableStream({
    start(controller) {
      child.stdout.on("data", (d: Buffer) => controller.enqueue(new Uint8Array(d)));
      child.stdout.on("end", () => controller.close());
      child.on("error", (e) => controller.error(e));
    },
    cancel() { child.kill(); },
  });
}

/** Ask the original and the trained model the same question (Tinker runs). */
export async function chat(id: string, message: string): Promise<{ original: string; trained: string }> {
  runDir(id);
  return JSON.parse(await worker(["chat", "--run", id, "--message", message, "--json"], 300_000));
}

/** Download the trained model and publish it to a private repo on the user's Hugging Face account. */
/** Download the trained adapter to this machine, and (publish) push it to a private Hugging Face repo. */
export async function startFetch(id: string, publish = true): Promise<void> {
  const dir = runDir(id);
  const record = (status: "running" | "done" | "failed", pid?: number) =>
    writeFile(path.join(dir, "fetch.json"), JSON.stringify({ status, pid })).catch(() => {});
  const pid = launch(["fetch", "--run", id, ...(publish ? ["--hf-repo", "auto"] : [])], path.join(dir, "fetch.log"), {
    onExit: (code) => record(code === 0 ? "done" : "failed"),
  });
  await record("running", pid);
}

const stripAnsi = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");

async function readOr<T>(file: string, parse: (s: string) => T, fallback: T): Promise<T> {
  try {
    return parse(await readFile(file, "utf8"));
  } catch {
    return fallback;
  }
}

// Turn a Python traceback's last line into something a person can act on.
function lastError(lines: string[]) {
  const last = [...lines].reverse().find((l) => l.trim() && !l.startsWith(" "));
  if (!last) return "The worker stopped unexpectedly.";
  return last.replace(/^[\w.]+(Error|Exception|Config|Exit)\b:\s*/, "").trim();
}

export async function getStatus(id: string): Promise<RunStatus | null> {
  const dir = runDir(id);
  const run = await readOr<Record<string, unknown> | null>(path.join(dir, "run.json"), JSON.parse, null);
  const web = await readOr<WebState | null>(path.join(dir, "web.json"), JSON.parse, null)
    ?? (run ? { status: "done" as const, exitCode: 0, startedAt: "" } : null);
  if (!web) return null;
  // A data step that stopped without reporting back (the server restarted mid-way, or it crashed)
  if (web.status === "running" && web.pid && !alive(web.pid)) web.status = run ? "done" : "failed";

  const lines = stripAnsi(await readOr(path.join(dir, "prepare.log"), (s) => s, "")).split("\n");
  const uploads = await readdir(path.join(dir, "uploads")).catch(() => [] as string[]);
  const files = await Promise.all(uploads.map(async (name) =>
    ({ name, size: await stat(path.join(dir, "uploads", name)).then((s) => s.size, () => 0) })));
  const sample = await readOr(path.join(dir, "train.jsonl"), (s) =>
    s.split("\n").filter(Boolean).slice(0, 3).map((line) => {
      const msgs = JSON.parse(line).messages as { role: string; content: string }[];
      return { user: msgs.find((m) => m.role === "user")?.content ?? "", assistant: msgs.find((m) => m.role === "assistant")?.content ?? "" };
    }), [] as { user: string; assistant: string }[]);

  const fromDataset = lines.some((l) => / rows from /.test(l));
  const parsed = fromDataset || !!run || lines.some((l) => l.includes(" chunks -> "));
  const failed = web.status === "failed";
  const reportStart = lines.findIndex((l) => / rows valid, /.test(l));
  const report = reportStart >= 0 ? lines.slice(reportStart).filter((l) => /rows valid|ERROR|warn/.test(l)).slice(0, 12) : [];
  const reportHasErrors = report.some((l) => l.includes("ERROR"));

  const read: Stage = parsed ? "done" : failed ? "failed" : "running";
  const write: Stage = !parsed ? "waiting" : run ? "done" : failed ? "failed" : "running";
  const check: Stage = !run ? "waiting" : reportHasErrors ? "failed" : "done";

  return {
    id,
    goal: typeof run?.goal === "string" ? run.goal : web.goal ?? null,
    baseModel: typeof run?.base_model === "string" ? run.base_model : web.baseModel ?? null,
    source: fromDataset || (run && typeof run.source === "object") ? "dataset" : "documents",
    sourceLabel: (() => {
      const src = run?.source as { hf?: string | null; file?: string | null } | undefined;
      return src?.hf ?? (src?.file ? path.basename(src.file) : null);
    })(),
    files,
    status: web.status,
    stages: { read, write, check },
    train: await trainProgress(dir, run),
    error: failed ? lastError(lines) : null,
    rows: typeof run?.rows === "number" ? run.rows : null,
    estTokens: typeof run?.est_tokens === "number" ? run.est_tokens : null,
    systemPrompt: typeof run?.system_prompt === "string" ? run.system_prompt : null,
    sample,
    report,
    settings: {
      // Saved before settings lived under "settings" (worker settings.load folds these in the same way)
      ...(typeof run?.train_epochs === "number" ? { epochs: run.train_epochs } : {}),
      ...(typeof run?.train_lora_rank === "number" ? { lora_rank: run.train_lora_rank } : {}),
      ...(typeof run?.learning_rate === "number" ? { learning_rate: run.learning_rate } : {}),
      ...((run?.settings ?? {}) as Settings),
    },
    testRows: typeof run?.test_rows === "number" ? run.test_rows : null,
    scorecard: await readScorecard(dir),
    scoring: await scoringState(dir),
    sourceInfo: run && typeof run.source === "object" && run.source
      ? (({ hf, config, split, limit }) => ({ hf, config, split, limit }))(run.source as Record<string, never>)
      : null,
  };
}

async function readScorecard(dir: string): Promise<Scorecard | null> {
  type Raw = {
    metric: Scorecard["metric"]; examples: number; updated: string; base_model: string; frontier_model: string | null;
    public: boolean; notes?: string[]; contenders?: Record<string, { score: number; correct: number; reused?: boolean; from?: string }>;
  };
  const card = await readOr<Raw | null>(path.join(dir, "scorecard.json"), JSON.parse, null);
  if (!card) return null;
  // Scores only: the answers themselves stay on disk
  const contenders = Object.fromEntries(Object.entries(card.contenders ?? {})
    .map(([name, c]) => [name, { score: c.score, correct: c.correct, reused: c.reused, from: c.from }]));
  return { metric: card.metric, examples: card.examples, updated: card.updated, baseModel: card.base_model,
    frontierModel: card.frontier_model, public: card.public, notes: card.notes ?? [], contenders };
}

async function scoringState(dir: string): Promise<RunStatus["scoring"]> {
  const s = await readOr<{ status: string; pid?: number; error?: string } | null>(path.join(dir, "score.json"), JSON.parse, null);
  if (!s) return { state: "idle", error: null };
  if (s.status === "running") return alive(s.pid) ? { state: "running", error: null } : { state: "idle", error: null };
  return s.status === "failed" ? { state: "failed", error: s.error ?? "Scoring stopped." } : { state: "idle", error: null };
}

/** Score the original, the trained model and a frontier model on the held-back set (costs a little). */
export async function startScore(id: string, frontier: string | null): Promise<void> {
  const dir = runDir(id);
  const record = (status: string, extra: Record<string, unknown> = {}) =>
    writeFile(path.join(dir, "score.json"), JSON.stringify({ status, ...extra })).catch(() => {});
  let stopped: string | null = null;
  const pid = launch(["score", "--run", id, "--yes", ...(frontier ? ["--frontier", frontier] : ["--no-frontier"])],
    path.join(dir, "score.log"), {
      onText: (text, stop) => {
        const broke = !stopped && outOfCredit(text.slice(-8000));
        if (broke) { stopped = broke; stop(); }
      },
      onExit: (code, text) => record(code === 0 ? "done" : "failed",
        code === 0 ? {} : { error: stopped ?? lastError(text.split("\n")) }),
    });
  await record("running", { pid });
}

/** The most a frontier model can cost to answer this run's held-back questions (OpenRouter's list prices,
 * every answer at full length; real answers are usually shorter). */
export async function frontierEstimate(id: string, model: string): Promise<number | null> {
  const test = await readOr(path.join(runDir(id), "test.jsonl"), (t) => t, "");
  const rows = test.split("\n").filter(Boolean);
  if (!rows.length) return null;
  const res = await fetch("https://openrouter.ai/api/v1/models", { next: { revalidate: 3600 } }).catch(() => null);
  const price = res?.ok ? ((await res.json()).data as { id: string; pricing: { prompt: string; completion: string } }[])
    .find((m) => m.id === model)?.pricing : undefined;
  if (!price) return null;
  const long = rows.some((r) => ((JSON.parse(r).messages as { content: string }[]).at(-1)?.content ?? "").split(/\s+/).length > 6);
  const promptTokens = rows.reduce((n, r) => n + r.length / 4, 0);
  return promptTokens * Number(price.prompt) + rows.length * (long ? 800 : 64) * Number(price.completion);
}

/** Every run, newest first, for History and the sidebar. */
export async function listRuns(): Promise<RunSummary[]> {
  const names = await readdir(RUNS).catch(() => [] as string[]);
  const all = await Promise.all(names.filter((n) => ID.test(n)).map(async (id): Promise<RunSummary | null> => {
    const dir = path.join(RUNS, id);
    const run = await readOr<Record<string, unknown> | null>(path.join(dir, "run.json"), JSON.parse, null);
    const web = await readOr<WebState | null>(path.join(dir, "web.json"), JSON.parse, null);
    if (!run && !web) return null;
    if (web?.status === "running" && web.pid && !alive(web.pid)) web.status = run ? "done" : "failed";
    const updated = await stat(path.join(dir, run ? "run.json" : "web.json")).then((s) => s.mtimeMs, () => 0);
    const job = typeof run?.job_state === "string" ? run.job_state : null;
    const state: RunSummary["state"] =
      job === "JOB_STATE_COMPLETED" ? "trained" : job === "JOB_STATE_FAILED" ? "failed" : job ? "training"
      : run ? "ready" : web?.status === "failed" ? "failed" : "preparing";
    return {
      id, updated, state, started: !!job,
      goal: typeof run?.goal === "string" ? run.goal : web?.goal ?? null,
      baseModel: typeof run?.base_model === "string" ? run.base_model : web?.baseModel ?? null,
      rows: typeof run?.rows === "number" ? run.rows : null,
    };
  }));
  return all.filter((r): r is RunSummary => r !== null).sort((a, b) => b.updated - a.updated);
}
