import "server-only";

import { rootEnv } from "./env";
import { TRAINERS, type Trainer } from "./models";

const auth = (): HeadersInit => (process.env.HF_TOKEN ? { Authorization: `Bearer ${process.env.HF_TOKEN}` } : {});

export type DatasetSort = "trending" | "downloads" | "likes" | "updated";

export type DatasetHit = {
  id: string; downloads: number; likes: number; updated: string | null; gated: boolean;
  /** Row count band from the dataset card, e.g. "10K–100K" */
  size: string | null;
};

/** One page of Hugging Face's dataset catalog, and the cursor for the next page (null at the end). */
export type DatasetPage = { hits: DatasetHit[]; next: string | null };

const SORT_FIELD: Record<DatasetSort, string> = {
  trending: "trendingScore", downloads: "downloads", likes: "likes", updated: "lastModified",
};
const PAGE = 30;

function sizeBand(tags: string[] = []): string | null {
  const t = tags.find((x) => x.startsWith("size_categories:"));
  if (!t) return null;
  const v = t.slice("size_categories:".length);
  const m = v.match(/^(\S+)<n<(\S+)$/);
  return m ? `${m[1]}–${m[2]}` : v.replace("n<", "under ").replace("n>", "over ");
}

/** Every dataset on Hugging Face, `q` to search, 30 at a time. `cursor` continues where the last page ended. */
export async function searchDatasets(q: string, sort: DatasetSort = "trending", cursor?: string): Promise<DatasetPage> {
  const url = new URL("https://huggingface.co/api/datasets");
  if (q) url.searchParams.set("search", q);
  url.searchParams.set("sort", SORT_FIELD[sort]);
  url.searchParams.set("limit", String(PAGE));
  if (cursor) url.searchParams.set("cursor", cursor);
  const res = await fetch(url, { headers: auth(), next: { revalidate: 600 } });
  if (!res.ok) return { hits: [], next: null };
  const rows: { id: string; downloads?: number; likes?: number; lastModified?: string; gated?: unknown; tags?: string[] }[] = await res.json();
  const next = res.headers.get("link")?.match(/[?&]cursor=([^&>]+)[^>]*>;\s*rel="next"/)?.[1] ?? null;
  return {
    hits: rows.map((r) => ({
      id: r.id, downloads: r.downloads ?? 0, likes: r.likes ?? 0, updated: r.lastModified ?? null,
      gated: !!r.gated && r.gated !== "false", size: sizeBand(r.tags),
    })),
    next: next && decodeURIComponent(next),
  };
}

export type DatasetInfo = {
  splits: { config: string; split: string }[];
  /** The subset and split the preview rows come from */
  picked: { config: string; split: string };
  columns: string[];
  rows: Record<string, string>[];
  /** Question/answer columns the worker would pick on its own, if any (worker/src/trainloop/datasets.py) */
  guess: { user: string; assistant: string } | null;
  chat: boolean;
};

// Same pairs the worker recognises, so the preview agrees with what import will do.
const PAIRS = [
  ["instruction", "output"], ["instruction", "response"], ["prompt", "response"], ["prompt", "completion"],
  ["question", "answer"], ["issue", "decision"], ["input", "output"], ["query", "answer"],
];

const cell = (v: unknown) => (v == null ? "" : typeof v === "string" ? v : JSON.stringify(v)).slice(0, 400);

/** Subsets, splits, columns and the first rows, read through the dataset viewer (nothing downloaded). */
export async function datasetInfo(id: string, config?: string, split?: string): Promise<DatasetInfo> {
  const viewer = "https://datasets-server.huggingface.co";
  const res = await fetch(`${viewer}/splits?dataset=${encodeURIComponent(id)}`, { headers: auth(), next: { revalidate: 3600 } });
  if (!res.ok) throw new Error("Hugging Face can't preview this dataset (it may be gated, private or not in a table format).");
  const splits: { config: string; split: string }[] = ((await res.json()).splits ?? [])
    .map((s: { config: string; split: string }) => ({ config: s.config, split: s.split }));
  if (splits.length === 0) throw new Error("This dataset has no splits to train on.");
  const pick = splits.find((s) => s.config === config && s.split === split)
    ?? splits.find((s) => s.config === (config ?? splits[0].config) && s.split === "train")
    ?? splits[0];

  const rowsRes = await fetch(`${viewer}/first-rows?dataset=${encodeURIComponent(id)}&config=${encodeURIComponent(pick.config)}&split=${encodeURIComponent(pick.split)}`,
    { headers: auth(), next: { revalidate: 3600 } });
  if (!rowsRes.ok) throw new Error("Hugging Face couldn't show rows for this split.");
  const body = await rowsRes.json();
  const columns: string[] = (body.features ?? []).map((f: { name: string }) => f.name);
  const rows = (body.rows ?? []).slice(0, 5).map((r: { row: Record<string, unknown> }) =>
    Object.fromEntries(columns.map((c) => [c, cell(r.row[c])])));
  const lower = new Map(columns.map((c) => [c.toLowerCase(), c]));
  const pair = PAIRS.find(([u, a]) => lower.has(u) && lower.has(a));
  return {
    splits,
    picked: pick,
    columns,
    rows,
    guess: pair ? { user: lower.get(pair[0])!, assistant: lower.get(pair[1])! } : null,
    chat: ["messages", "conversations"].some((c) => lower.has(c)),
  };
}

/** Which trainers have keys set in the root .env. Names only; values never leave the server. */
export async function trainerStatus(): Promise<{ trainer: Trainer; connected: boolean }[]> {
  const env = await rootEnv();
  return (Object.keys(TRAINERS) as Trainer[]).map((t) => ({ trainer: t, connected: !!env[TRAINERS[t].env] }));
}
