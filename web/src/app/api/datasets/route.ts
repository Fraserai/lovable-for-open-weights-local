import type { NextRequest } from "next/server";
import { datasetInfo, searchDatasets, type DatasetSort } from "@/lib/hub";

export const runtime = "nodejs";

const SORTS: DatasetSort[] = ["trending", "downloads", "likes", "updated"];

// GET /api/datasets?q=legal&sort=downloads&cursor=… -> { hits, next }: all of Hugging Face, 30 at a time
// GET /api/datasets?id=owner/name[&config&split]    -> subsets, splits, columns and first rows
export async function GET(req: NextRequest) {
  const p = req.nextUrl.searchParams;
  const id = p.get("id")?.trim();
  if (id) {
    try {
      return Response.json(await datasetInfo(id, p.get("config") ?? undefined, p.get("split") ?? undefined));
    } catch (e) {
      return Response.json({ error: e instanceof Error ? e.message : "Couldn't read that dataset." }, { status: 400 });
    }
  }
  const sort = SORTS.find((s) => s === p.get("sort")) ?? "trending";
  const cursor = p.get("cursor") ?? undefined;
  if (cursor && !/^[\w=+/%-]{1,2000}$/.test(cursor)) return Response.json({ error: "Bad cursor." }, { status: 400 });
  return Response.json(await searchDatasets(p.get("q")?.trim() ?? "", sort, cursor).catch(() => ({ hits: [], next: null })));
}
