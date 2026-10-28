"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";

type Point = { x: number; y: number };

const H = 200, PAD = { l: 52, r: 12, t: 12, b: 26 };

/** EMA like Unsloth's charts (smoothing 0.6; heavier on long runs), so a noisy loss still shows its trend. */
function ema(points: Point[], alpha = points.length > 100 ? 0.9 : 0.6) {
  let prev: number | null = null;
  return points.map((p) => ({ x: p.x, y: (prev = prev === null ? p.y : alpha * prev + (1 - alpha) * p.y) }));
}

const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : Math.abs(v) >= 0.001 ? v.toFixed(3) : v.toExponential(1));

/** A line chart for per-step training metrics. `smooth` draws the raw series faintly under its EMA. */
export default function Chart({ title, points, smooth = false, format = fmt, empty = "Waiting for the first step…" }: {
  title: string; points: Point[]; smooth?: boolean; format?: (v: number) => string; empty?: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const clipId = useId().replace(/:/g, "");
  const line = useMemo(() => (smooth ? ema(points) : points), [points, smooth]);
  // Draw at the real width so labels stay legible at any size
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(600);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(240, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  // Loss starts far above where it settles: clip the top at the 97th percentile (like Unsloth's p95/p99
  // outlier clipping) so the trend isn't flattened by the first few steps.
  const sorted = [...ys].sort((a, b) => a - b);
  const lo = sorted[0] ?? 0;
  const hi = smooth && sorted.length > 20 ? sorted[Math.floor(sorted.length * 0.97)] : sorted.at(-1) ?? 0;
  const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1;
  const y0 = lo - pad, y1 = hi + pad;
  const sx = (x: number) => PAD.l + (x1 === x0 ? 0.5 : (x - x0) / (x1 - x0)) * (W - PAD.l - PAD.r);
  const sy = (y: number) => PAD.t + (1 - (y - y0) / (y1 - y0)) * (H - PAD.t - PAD.b);
  const path = (ps: Point[]) => ps.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
  // A flat line (constant learning rate) puts all three on one value: draw it once
  const ticks = [...new Set([y1 - pad, (y0 + y1) / 2, y0 + pad])];
  const h = hover !== null ? line[hover] : null;
  const last = line.at(-1);

  return (
    <figure className="glass rounded-[32px] p-6 sm:p-8">
      <figcaption className="flex items-baseline justify-between gap-3">
        <span className="text-[17px]">{title}{smooth && <span className="ml-2 text-xs font-normal text-faint">smoothed</span>}</span>
        <span className="font-mono text-sm text-muted tabular-nums">
          {h ? <>step {h.x} · {format(h.y)}</> : last ? format(last.y) : ""}
        </span>
      </figcaption>
      <div ref={box} className="mt-3">
      {points.length < 2 ? (
        <p className="grid h-[180px] place-items-center text-sm text-faint">{empty}</p>
      ) : (
        <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="block w-full touch-none overflow-visible" role="img" aria-label={`${title}, last value ${last ? format(last.y) : ""}`}
             onPointerMove={(e) => {
               const r = e.currentTarget.getBoundingClientRect();
               const x = x0 + ((((e.clientX - r.left) / r.width) * W - PAD.l) / (W - PAD.l - PAD.r)) * (x1 - x0);
               let best = 0;
               line.forEach((p, i) => { if (Math.abs(p.x - x) < Math.abs(line[best].x - x)) best = i; });
               setHover(best);
             }}
             onPointerLeave={() => setHover(null)}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={PAD.l} x2={W - PAD.r} y1={sy(t)} y2={sy(t)} className="stroke-line" strokeDasharray="3 4" />
              <text x={PAD.l - 8} y={sy(t) + 4} textAnchor="end" className="fill-faint font-mono text-[11px]">{format(t)}</text>
            </g>
          ))}
          <text x={PAD.l} y={H - 6} className="fill-faint font-mono text-[11px]">{x0}</text>
          <text x={W - PAD.r} y={H - 6} textAnchor="end" className="fill-faint font-mono text-[11px]">step {x1}</text>
          <clipPath id={clipId}><rect x={PAD.l} y={PAD.t - 2} width={W - PAD.l - PAD.r} height={H - PAD.t - PAD.b + 4} /></clipPath>
          <g clipPath={`url(#${clipId})`}>
            {smooth && <path d={path(points)} fill="none" className="stroke-accent/25" strokeWidth={1.2} />}
            <path d={path(line)} fill="none" className="stroke-accent" strokeWidth={2.2} strokeLinejoin="round" strokeLinecap="round" />
          </g>
          {h && (
            <g>
              <line x1={sx(h.x)} x2={sx(h.x)} y1={PAD.t} y2={H - PAD.b} className="stroke-faint" strokeDasharray="2 3" />
              <circle cx={sx(h.x)} cy={Math.max(PAD.t, sy(h.y))} r={4} className="fill-accent stroke-surface" strokeWidth={2} />
            </g>
          )}
        </svg>
      )}
      </div>
    </figure>
  );
}
