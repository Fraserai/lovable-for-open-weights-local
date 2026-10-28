type P = { className?: string };
const base = { fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" } as const;

export const Chevron = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} aria-hidden><path d="m6 9 6 6 6-6" /></svg>
);
export const Close = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} aria-hidden><path d="M6 6l12 12M18 6 6 18" /></svg>
);
export const Doc = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} aria-hidden>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5M9 13h6M9 17h4" />
  </svg>
);
export const Check = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} strokeWidth={2.2} aria-hidden><path d="m5 12 5 5 9-10" /></svg>
);
export const Chat = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} aria-hidden>
    <path d="M20 12a8 8 0 0 1-11.6 7.1L4 20l1-4.2A8 8 0 1 1 20 12z" />
  </svg>
);
export const Database = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} aria-hidden>
    <ellipse cx="12" cy="5.5" rx="7.5" ry="2.5" /><path d="M4.5 5.5v13c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5v-13M4.5 12c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5" />
  </svg>
);
export const Sliders = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} aria-hidden>
    <path d="M4 7h10M18 7h2M4 17h4M12 17h8" /><circle cx="16" cy="7" r="2" /><circle cx="10" cy="17" r="2" />
  </svg>
);
export const Target = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} aria-hidden>
    <circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="4" /><circle cx="12" cy="12" r=".6" fill="currentColor" />
  </svg>
);
export const FileCog = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} aria-hidden>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h4M14 3v5h5M14 3l5 5v3" /><circle cx="17" cy="17" r="2.5" />
    <path d="M17 13v1.5M17 19.5V21M13 17h1.5M19.5 17H21" />
  </svg>
);
export const Rocket = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} aria-hidden>
    <path d="M14 4c3-1 5.5-1 6-.5s.5 3-.5 6L13 16l-5-5 6-7z" /><path d="M8 11l-3.5.5L3 14l4 1M13 16l-.5 3.5L10 21l-1-4" />
    <circle cx="15.5" cy="8.5" r="1.3" />
  </svg>
);
export const Upload = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} aria-hidden>
    <path d="M12 16V4M7 9l5-5 5 5M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
  </svg>
);
export const Search = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} aria-hidden><circle cx="11" cy="11" r="6.5" /><path d="m20 20-4-4" /></svg>
);
export const Eye = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} aria-hidden>
    <path d="M2.5 12S6 5 12 5s9.5 7 9.5 7-3.5 7-9.5 7-9.5-7-9.5-7z" /><circle cx="12" cy="12" r="2.8" />
  </svg>
);
export const External = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} aria-hidden><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" /></svg>
);

export const Info = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} strokeWidth={1.6} aria-hidden><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 7.5v.01" /></svg>
);
export const Brain = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} strokeWidth={1.6} aria-hidden>
    <path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3 3 3 0 0 0 3-3V7a3 3 0 0 0-3-3z" />
    <path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3 3 3 0 0 1-3-3V7a3 3 0 0 1 3-3z" />
    <path d="M9 9h1.5M13.5 9H15M8 14h2M14 14h2" />
  </svg>
);
export const Chip = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} strokeWidth={1.6} aria-hidden>
    <rect x="7" y="7" width="10" height="10" rx="2" /><path d="M10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4" />
  </svg>
);
export const CloudUp = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} strokeWidth={1.6} aria-hidden>
    <path d="M7 18a4.5 4.5 0 0 1-.6-9A6 6 0 0 1 18 8a4 4 0 0 1-.5 10" /><path d="M12 13v7M9 15.5 12 13l3 2.5" />
  </svg>
);
export const Server = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} strokeWidth={1.6} aria-hidden>
    <rect x="4" y="4" width="16" height="7" rx="2" /><rect x="4" y="13" width="16" height="7" rx="2" /><path d="M8 7.5h.01M8 16.5h.01" />
  </svg>
);
export const Hug = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} strokeWidth={1.6} aria-hidden>
    <circle cx="12" cy="12" r="8.5" /><path d="M9 10v.01M15 10v.01M8.5 14c1 1.5 2.2 2.2 3.5 2.2s2.5-.7 3.5-2.2" />
  </svg>
);

export const ArrowUp = ({ className }: P) => (
  <svg viewBox="0 0 24 24" className={className} {...base} strokeWidth={2.2} aria-hidden><path d="M12 19V5M5 12l7-7 7 7" /></svg>
);

// The mark: three stacked plates (weights / model layers)
export const Logo = ({ className }: P) => (
  <svg viewBox="0 0 32 32" className={className} aria-hidden>
    <defs>
      <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stopColor="#2dd4bf" /><stop offset=".55" stopColor="#4f73ff" /><stop offset="1" stopColor="#9b7bff" />
      </linearGradient>
    </defs>
    <rect x="3" y="4" width="26" height="6" rx="3" fill="url(#lg)" />
    <rect x="6" y="13" width="20" height="6" rx="3" fill="url(#lg)" opacity=".8" />
    <rect x="9" y="22" width="14" height="6" rx="3" fill="url(#lg)" opacity=".6" />
  </svg>
);
