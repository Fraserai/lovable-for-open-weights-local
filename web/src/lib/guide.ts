"use client";

import type { Settings } from "./settings";

// The guide (the logo button) talks to the page through here: it asks for actions, and the page
// keeps it told what's on screen. Actions only fill in the form; nothing here can start training.

export type GuideTarget = "model" | "dataset" | "parameters" | "configuration" | "preview" | "tabs" | "scorecard";

export type GuideAction =
  | { type: "set_model"; query: string }
  | { type: "set_trainer"; trainer: string }
  | { type: "set_dataset"; id: string }
  | { type: "set_settings"; settings: Partial<Settings> }
  | { type: "open_tab"; tab: "configure" | "run" | "history" }
  | { type: "spotlight"; target: GuideTarget };

/** Carries out an action if it's this handler's to do; returns a short note of what happened. */
type Handler = (a: GuideAction) => Promise<string | null> | string | null;

const handlers = new Set<Handler>();

export function onGuideAction(h: Handler) {
  handlers.add(h);
  return () => { handlers.delete(h); };
}

export async function runGuideAction(a: GuideAction): Promise<string> {
  if (a.type === "spotlight") {
    spotlight(a.target);
    return `Showed you ${a.target}.`;
  }
  for (const h of handlers) {
    const note = await h(a);
    if (note) return note;
  }
  return "That isn't on the page right now.";
}

// What's on screen, as the guide sees it. Page parts merge their slice in.
let page: Record<string, unknown> = {};
export const setPageState = (patch: Record<string, unknown>) => { page = { ...page, ...patch }; };
export const pageState = () => page;

/** Scroll a part of the page into view and glow it for a moment. */
export function spotlight(target: GuideTarget) {
  const el = document.querySelector<HTMLElement>(`[data-guide="${target}"]`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("guide-spot");
  void el.offsetWidth; // restart the animation if it's already glowing
  el.classList.add("guide-spot");
  window.setTimeout(() => el.classList.remove("guide-spot"), 2800);
}
