"use client";

import { Chevron, Info } from "./icons";

// Building blocks for the Train page: Unsloth Studio's layout in liquid glass (frosted panels, pill fields,
// uppercase field labels with an info mark, iOS-style segmented controls and sliders).

export const btn = {
  primary: "inline-flex items-center justify-center gap-2 rounded-full bg-gradient-to-b from-[#2b8fff] to-[#0068e6] px-5 py-3 text-[15px] font-semibold text-white shadow-[0_8px_24px_-8px_rgb(0_122_255/0.6),inset_0_1px_0_rgb(255_255_255/0.35)] transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none",
  ghost: "inline-flex items-center justify-center gap-2 rounded-full border border-white/80 bg-white/60 px-4 py-2.5 text-sm shadow-sm transition hover:bg-white/85 disabled:cursor-not-allowed disabled:opacity-40",
  quiet: "inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-sm text-ink transition hover:bg-white/60",
};

export const input =
  "h-12 w-full rounded-2xl border border-white/70 bg-field px-4 text-[15px] shadow-[inset_0_1px_2px_rgb(0_0_0/0.04)] outline-none transition placeholder:text-faint focus:border-accent/40 focus:ring-4 focus:ring-accent/15 disabled:cursor-not-allowed disabled:opacity-50";

export function Card({ icon, tone = "#34c759", title, subtitle, action, children, guide }: {
  icon: React.ReactNode; tone?: string; title: string; subtitle?: string; action?: React.ReactNode; children: React.ReactNode;
  /** Lets the guide point at this card (lib/guide.ts spotlight) */
  guide?: string;
}) {
  return (
    <section data-guide={guide} className="glass @container relative rounded-[32px] p-6 sm:p-8 has-[[aria-haspopup][aria-expanded=true]]:z-40" aria-label={title}>
      <header className="mb-7 flex flex-wrap items-center gap-4">
        <span className="grid size-12 shrink-0 place-items-center rounded-full" style={{ background: `linear-gradient(180deg, color-mix(in srgb, ${tone} 22%, white), color-mix(in srgb, ${tone} 12%, white))`, color: tone, boxShadow: "inset 0 1px 0 rgb(255 255 255 / 0.8)" }}>
          {icon}
        </span>
        <div className="min-w-[12rem] flex-1">
          <h2 className="text-[17px] leading-tight">{title}</h2>
          {subtitle && <p className="mt-0.5 text-[15px] text-muted">{subtitle}</p>}
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </header>
      {children}
    </section>
  );
}

/** Field label. `upper` for the Model/Dataset cards' style (MODEL ⓘ), Title Case elsewhere (Subset ⓘ). */
export function Label({ children, hint, htmlFor, upper, aside, note }: {
  children: React.ReactNode; hint?: string; htmlFor?: string; upper?: boolean; aside?: React.ReactNode; note?: string;
}) {
  return (
    <div className="mb-2.5 flex items-center gap-2">
      <label htmlFor={htmlFor} className={upper ? "text-[13px] tracking-[0.08em] text-muted uppercase" : "text-[15px] text-muted"}>{children}</label>
      {note && <span className="text-xs text-faint">{note}</span>}
      {hint && <span title={hint} aria-label={hint} className="cursor-help text-faint"><Info className="size-4" /></span>}
      {aside && <span className="ml-auto">{aside}</span>}
    </div>
  );
}

export function Select<T extends string>({ id, value, onChange, options, disabled, lead }: {
  id?: string; value: T; onChange: (v: T) => void; options: { value: T; label: string; disabled?: boolean }[];
  disabled?: boolean; lead?: React.ReactNode;
}) {
  return (
    <div className="relative">
      {lead && <span className="pointer-events-none absolute top-1/2 left-4 -translate-y-1/2">{lead}</span>}
      <select id={id} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value as T)}
              className={`${input} appearance-none pr-11 ${lead ? "pl-10" : ""}`}>
        {options.map((o) => <option key={o.value} value={o.value} disabled={o.disabled}>{o.label}</option>)}
      </select>
      <Chevron className="pointer-events-none absolute top-1/2 right-4 size-5 -translate-y-1/2 text-muted" />
    </div>
  );
}

/** Pill tabs: Simple / Advanced, Optimization / Schedule / Memory. */
export function Tabs<T extends string>({ value, onChange, options, label, wide }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: string }[]; label: string; wide?: boolean;
}) {
  return (
    <div role="tablist" aria-label={label} className={`inline-flex rounded-full bg-black/[0.06] p-1 ${wide ? "w-full max-w-[560px]" : ""}`}>
      {options.map((o) => (
        <button key={o.value} type="button" role="tab" aria-selected={o.value === value} onClick={() => onChange(o.value)}
                className={`rounded-full px-5 py-2 text-[15px] text-muted transition hover:text-ink aria-selected:bg-white aria-selected:text-ink aria-selected:shadow-[0_2px_8px_rgb(0_0_0/0.12)] ${wide ? "flex-1" : ""}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** A labelled slider with a value box, like Unsloth's Rank / Alpha / Batch Size rows.
 * `value` undefined = automatic: the knob sits greyed at `auto`, and typing or dragging sets it. */
export function SliderRow({ label, hint, value, auto, min, max, step = 1, onChange, disabled, note, format }: {
  label: string; hint?: string; value: number | undefined; auto: number; min: number; max: number; step?: number;
  onChange: (v: number | undefined) => void; disabled?: boolean; note?: string; format?: (v: number) => string;
}) {
  const shown = value ?? auto;
  const pct = `${Math.max(0, Math.min(100, ((shown - min) / (max - min)) * 100))}%`;
  const id = `s-${label.replace(/\W+/g, "-").toLowerCase()}`;
  return (
    <div className={`grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-6 gap-y-1 py-3 ${disabled ? "opacity-45" : ""}`}>
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <label htmlFor={id} className="text-[15px] text-muted">{label}</label>
          {hint && <span title={hint} aria-label={hint} className="cursor-help text-faint"><Info className="size-4" /></span>}
          {value === undefined && !disabled && <span className="rounded-full bg-white/60 px-2 py-0.5 text-[11px] text-faint">Auto</span>}
        </div>
        {note && <p className="mt-0.5 text-xs text-faint">{note}</p>}
      </div>
      <div className="flex items-center gap-4">
        <input type="range" min={min} max={max} step={step} value={shown} disabled={disabled} aria-label={label}
               onChange={(e) => onChange(Number(e.target.value))}
               className={`slider w-[120px] sm:w-[200px] ${value === undefined ? "auto" : ""}`} style={{ "--pct": pct } as React.CSSProperties} />
        <input id={id} inputMode="decimal" disabled={disabled} value={value === undefined ? "" : String(value)}
               placeholder={format ? format(auto) : String(auto)}
               onChange={(e) => {
                 const t = e.target.value.trim();
                 if (!t) return onChange(undefined);
                 const n = Number(t);
                 if (Number.isFinite(n)) onChange(Math.max(min, Math.min(max, n)));
               }}
               className="h-9 w-20 rounded-full border border-white/70 bg-field px-3 text-right font-mono text-sm outline-none placeholder:text-muted focus:ring-4 focus:ring-accent/15" />
      </div>
    </div>
  );
}

export function Pill({ tone = "muted", children }: { tone?: "muted" | "accent" | "warn" | "bad"; children: React.ReactNode }) {
  const tones = {
    muted: "bg-white/60 text-muted",
    accent: "bg-[#34c759]/15 text-[#1f8a3a]",
    warn: "bg-amber-400/20 text-amber-800",
    bad: "bg-red-500/12 text-red-600",
  };
  return <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium whitespace-nowrap ${tones[tone]}`}>{children}</span>;
}

export function ErrorNote({ children }: { children: React.ReactNode }) {
  return <p role="alert" className="rounded-2xl border border-red-200/70 bg-red-50/70 px-4 py-3 text-sm text-red-700">{children}</p>;
}

export function Dot({ className = "" }: { className?: string }) {
  return <span className={`inline-block size-2 animate-pulse rounded-full bg-accent ${className}`} aria-hidden />;
}

export const money = (n: number | null | undefined) =>
  n == null ? "—" : n < 0.01 ? "under 1¢" : n < 1 ? `${Math.round(n * 100)}¢` : `$${n.toFixed(2)}`;

export const duration = (sec: number | null | undefined) => {
  if (sec == null || !Number.isFinite(sec)) return "—";
  const s = Math.max(0, Math.round(sec));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`;
};

export const ago = (ms: number) => {
  const s = (Date.now() - ms) / 1000;
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(ms).toLocaleDateString(undefined, { day: "numeric", month: "short" });
};

export const sci = (v: number) => {
  const [m, e] = v.toExponential(1).split("e");
  return `${m.replace(/\.0$/, "")}e${e.replace("+", "")}`;
};
