# web

The Train screen: one page with Configure, Current Run and History tabs, in a liquid-glass theme.
Next.js 16 (see `AGENTS.md`: this version differs from older Next.js; its docs ship in
`node_modules/next/dist/docs/`).

```bash
pnpm install
pnpm dev        # http://localhost:3000 (the worker must be set up: see ../worker/README.md)
```

## Where things are

- `src/app/page.tsx`: the only page. `?tab=run&run=<id>` opens a run; `?draft=<id>` reopens a checked dataset.
- `src/components/studio/`: `Studio` (tabs), `Configure` (the form), `ModelPicker`-backed Model card,
  `DatasetCard`, `ParamsCard`, `RunPreview` (the right-hand rail), `CurrentRun`, `History`.
- `src/components/Guide.tsx`: the guide behind the logo button (bottom right): a tour of the page, and chat that
  explains things and fills the form in. `src/lib/guide.ts` connects it to the page; `src/lib/guide-agent.ts` asks
  OpenRouter (zero data retention only; `AGENT_MODEL` in .env to change model). It can't start training.
- `src/components/ui.tsx`: cards, fields, sliders and tabs. Theme tokens and glass styles: `src/app/globals.css`.
- `src/lib/brand.ts`: name, tagline, mission and brand lines (see ../BRAND.md).
- `src/lib/settings.ts`: the training settings and which trainer honours which (mirrors the worker).
- `src/lib/catalog.ts`: trainable models from the worker's catalog, cached 6h in `../data/`.
- `src/lib/hub.ts`: Hugging Face dataset search and preview, and which trainers have keys set.
- `src/lib/runs.ts`: starts worker commands and reads their progress back from `../runs/<id>/`.
- `src/app/api/`: `models`, `datasets`, `trainers`, `runs` (list, create), `runs/[id]` (status, delete draft),
  `runs/[id]/configure` (settings + plan), `runs/[id]/train`, `runs/[id]/chat`, `runs/[id]/fetch`, `guide`.
