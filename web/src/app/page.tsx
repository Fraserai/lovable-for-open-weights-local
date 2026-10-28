import Studio, { type Tab } from "@/components/studio/Studio";

// The whole product is one page: Train, with Configure / Current Run / History tabs.
// ?tab=run&run=<id> opens a run; ?draft=<id> reopens a dataset that's been checked but not trained.
export default async function Home({ searchParams }: PageProps<"/">) {
  const p = await searchParams;
  const one = (v: string | string[] | undefined) => (typeof v === "string" && v ? v : null);
  const tab = (["configure", "run", "history"] as const).find((t) => t === p.tab) ?? "configure";
  return <Studio initialTab={tab as Tab} initialRun={one(p.run)} initialDraft={one(p.draft)} />;
}
