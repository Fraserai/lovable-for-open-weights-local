"use client";

import { useEffect, useRef, useState } from "react";
import { BRAND } from "@/lib/brand";
import { pageState, runGuideAction, spotlight, type GuideAction, type GuideTarget } from "@/lib/guide";
import { ArrowUp, Check, Close, Logo } from "./icons";
import { btn, Dot } from "./ui";

// The guide: our logo, bottom right. It gives a tour of the Train screen (no AI needed), answers
// questions, and fills the form in when you say what you want. It can't start training.

const TOUR: { target: GuideTarget; title: string; text: string }[] = [
  { target: "tabs", title: "One page", text: "This is Train, the whole tool. Configure sets up a run, Current Run shows it training, History keeps every run." },
  { target: "model", title: "Pick a model", text: "Choose any open model. The badges show which trainer can run it: Tinker, Fireworks, or a GPU on your Hugging Face account. LoRA trains a small adapter on top, so it's quick and cheap." },
  { target: "dataset", title: "Bring what it learns from", text: "Any Hugging Face dataset, your own .jsonl examples, or your documents. A dataset is checked, free, the moment you pick it." },
  { target: "parameters", title: "Settings", text: "Leave them on Auto and we size them for your data. Advanced has every control; anything your trainer can't do is greyed out." },
  { target: "preview", title: "See it before you pay", text: "The Run preview shows what will run, where, and what it costs, before you spend anything. Start Training is the only step that costs money." },
  { target: "configuration", title: "Reuse a setup", text: "Save everything as a YAML file to run it again, or load one you saved." },
  { target: "tabs", title: "Own it", text: "When it's done, Current Run lets you talk to your model and publish it to a private Hugging Face repo. Your Models. Your Intelligence." },
];

const STARTERS = [
  "Take the tour",
  "Train a model on a Hugging Face dataset",
  "Train a model on my documents",
  "What does LoRA rank do?",
  "How much will training cost?",
];

type Msg = { role: "user" | "assistant"; content: string; notes?: string[]; error?: boolean };

const GREETING: Msg = {
  role: "assistant",
  content: "Hi, I'm your guide. Tell me what you want your model to do and I'll set it up, or ask me anything about the tool.",
};

const NAMES = new Set(["set_model", "set_trainer", "set_dataset", "set_settings", "open_tab", "spotlight"]);

/** The server's tool calls as page actions (anything malformed is dropped). */
function toAction({ name, args }: { name: string; args: Record<string, unknown> }): GuideAction | null {
  if (!NAMES.has(name)) return null;
  const str = (k: string) => (typeof args[k] === "string" ? (args[k] as string) : "");
  switch (name) {
    case "set_model": return str("query") ? { type: "set_model", query: str("query") } : null;
    case "set_trainer": return str("trainer") ? { type: "set_trainer", trainer: str("trainer") } : null;
    case "set_dataset": return str("id") ? { type: "set_dataset", id: str("id") } : null;
    case "set_settings": return args.settings && typeof args.settings === "object" ? { type: "set_settings", settings: args.settings as Record<string, never> } : null;
    case "open_tab": return ["configure", "run", "history"].includes(str("tab")) ? { type: "open_tab", tab: str("tab") as "configure" } : null;
    default: return str("target") ? { type: "spotlight", target: str("target") as GuideTarget } : null;
  }
}

export default function Guide() {
  const [open, setOpen] = useState(false);
  const [msgs, setMsgs] = useState<Msg[]>([GREETING]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [tour, setTour] = useState<number | null>(null);
  const list = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLInputElement>(null);

  useEffect(() => {
    list.current?.scrollTo({ top: list.current.scrollHeight, behavior: "smooth" });
  }, [msgs, busy, tour]);
  useEffect(() => {
    if (!open) return;
    box.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  const goTour = (i: number | null) => {
    setTour(i);
    if (i !== null) spotlight(TOUR[i].target);
  };

  const send = async (raw: string) => {
    const q = raw.trim();
    if (!q || busy) return;
    if (q === "Take the tour") return goTour(0);
    setText("");
    setTour(null);
    const history = [...msgs, { role: "user" as const, content: q }];
    setMsgs(history);
    setBusy(true);
    try {
      const res = await fetch("/api/guide", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history.filter((m) => !m.error).map(({ role, content }) => ({ role, content })), page: pageState() }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error ?? "The guide can't answer right now.");
      const notes: string[] = [];
      for (const call of body.actions ?? []) {
        const action = toAction(call);
        if (action) notes.push(await runGuideAction(action));
      }
      setMsgs((m) => [...m, { role: "assistant", content: body.reply || (notes.length ? "Done." : "…"), notes }]);
    } catch (e) {
      setMsgs((m) => [...m, { role: "assistant", content: e instanceof Error ? e.message : "The guide can't answer right now.", error: true }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      {open && (
        <section role="dialog" aria-label={`${BRAND.name} guide`}
                 className="glass-pop fixed right-4 bottom-24 z-50 flex max-h-[min(640px,calc(100dvh-8rem))] w-[min(400px,calc(100vw-2rem))] flex-col overflow-hidden rounded-[28px] sm:right-6">
          <header className="flex items-center gap-3 border-b border-black/5 px-5 py-4">
            <Logo className="size-7" />
            <div className="min-w-0 flex-1">
              <p className="text-[15px] font-medium">{BRAND.name} guide</p>
              <p className="text-xs text-muted">{BRAND.tagline}</p>
            </div>
            <button type="button" onClick={() => setOpen(false)} aria-label="Close the guide" className={btn.quiet}><Close className="size-5" /></button>
          </header>

          <div ref={list} className="flex-1 space-y-3 overflow-y-auto px-5 py-4" aria-live="polite">
            {msgs.map((m, i) => (
              <div key={i} className={m.role === "user" ? "flex justify-end" : ""}>
                <div className={m.role === "user"
                  ? "max-w-[85%] rounded-3xl rounded-br-lg bg-accent px-4 py-2.5 text-[15px] text-white"
                  : `max-w-[92%] text-[15px] leading-relaxed whitespace-pre-wrap ${m.error ? "text-red-600" : ""}`}>
                  {m.content}
                  {m.notes?.map((n) => (
                    <p key={n} className="mt-1.5 flex items-start gap-1.5 text-[13px] text-muted"><Check className="mt-0.5 size-3.5 shrink-0 text-[#34c759]" /> {n}</p>
                  ))}
                </div>
              </div>
            ))}
            {busy && <p className="flex items-center gap-2 text-sm text-muted"><Dot /> Thinking…</p>}

            {tour !== null && (
              <div className="rounded-3xl border border-white/80 bg-white/70 p-4 shadow-sm">
                <p className="text-xs text-faint">Tour · {tour + 1} of {TOUR.length}</p>
                <p className="mt-1 text-[15px] font-medium">{TOUR[tour].title}</p>
                <p className="mt-1 text-[15px] leading-relaxed text-muted">{TOUR[tour].text}</p>
                <div className="mt-3 flex items-center gap-2">
                  {tour > 0 && <button type="button" onClick={() => goTour(tour - 1)} className={btn.ghost}>Back</button>}
                  <button type="button" onClick={() => goTour(tour + 1 < TOUR.length ? tour + 1 : null)} className={`${btn.primary} ml-auto px-4 py-2 text-sm`}>
                    {tour + 1 < TOUR.length ? "Next" : "Done"}
                  </button>
                </div>
              </div>
            )}

            {msgs.length === 1 && tour === null && !busy && (
              <div className="flex flex-wrap gap-2 pt-1">
                {STARTERS.map((s) => (
                  <button key={s} type="button" onClick={() => send(s)}
                          className="rounded-full border border-white/80 bg-white/60 px-3.5 py-1.5 text-left text-sm shadow-sm transition hover:bg-white">
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>

          <form onSubmit={(e) => { e.preventDefault(); send(text); }} className="flex items-center gap-2 border-t border-black/5 p-3">
            <label htmlFor="guide-ask" className="sr-only">Ask the guide</label>
            <input id="guide-ask" ref={box} value={text} onChange={(e) => setText(e.target.value)} disabled={busy}
                   placeholder="What do you want to build?"
                   className="h-11 min-w-0 flex-1 rounded-full border border-white/70 bg-white/70 px-4 text-[15px] outline-none placeholder:text-faint focus:ring-4 focus:ring-accent/15" />
            <button type="submit" disabled={busy || !text.trim()} aria-label="Send"
                    className="grid size-11 shrink-0 place-items-center rounded-full bg-accent text-white transition disabled:opacity-30">
              <ArrowUp className="size-4" />
            </button>
          </form>
        </section>
      )}

      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}
              aria-label={open ? "Close the guide" : `Open the ${BRAND.name} guide`} title={`${BRAND.name} guide`}
              className="glass fixed right-4 bottom-6 z-50 grid size-16 place-items-center rounded-full transition hover:scale-105 active:scale-95 sm:right-6">
        <Logo className="size-8" />
      </button>
    </>
  );
}
