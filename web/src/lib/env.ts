import "server-only";

import { readFile } from "node:fs/promises";
import path from "node:path";

/** Keys from the repo-root .env (the worker's), which Next.js doesn't load for web/. Values never leave the server. */
export async function rootEnv(): Promise<Record<string, string>> {
  const text = await readFile(path.resolve(process.cwd(), "..", ".env"), "utf8").catch(() => "");
  const out: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const m = line.match(/^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (m && m[2]) out[m[1]] = m[2].replace(/^["']|["']$/g, "");
  }
  return out;
}
