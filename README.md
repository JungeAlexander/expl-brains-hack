# Brain Insight Explorer

Streamlit dashboard for the Vibraint c-Fos dataset (Semaglutide vs Vehicle, ~1.4k
brain regions). Click any dot in the volcano to drill into per-animal densities,
coronal slice overlays, Amass-linked literature, and a Claude-generated narrative
that ties stats to publications.

See [`CHALLENGE_B.md`](CHALLENGE_B.md) for the data spec.

---

## Setup (uv)

If you're cloning this repo for the first time:

```bash
uv sync
```

This reads `pyproject.toml` + `uv.lock` and creates `.venv/` with every dependency
pinned to the same versions everyone else is using.

> **Starting from scratch (no `pyproject.toml`)** — equivalent to what was run once:
> ```bash
> uv init
> uv add -r requirements.txt streamlit-plotly-events
> ```

## Secrets

Copy the example and fill in real keys (shared in the team chat):

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

The file holds:
- `ANTHROPIC_API_KEY` — for Claude narratives and chat
- `AMASS_API_KEY` — for BiomedCore / TrialCore literature

Both `secrets.toml` and `bucket_access/config.py` (Hetzner credentials) are
gitignored. Don't commit them.

## Run

```bash
uv run streamlit run app.py
```

First launch downloads ~5 NIfTI volumes + 3 CSVs from the bucket into
`data_cache/` (~few hundred MB). Subsequent launches read from disk and are fast.

Open <http://localhost:8501>.

---

## Layout

```
app.py                          Streamlit entry (sidebar + volcano + drilldown + chat)
modules/
  data.py                       bucket fetch (cached), CSV loaders, NIfTI bundle, centroids
  nifti.py                      coronal slice triptych (G001 / G002 / diff, red contour)
  viz.py                        volcano, per-animal strip plot, tour bar chart
  amass.py                      BiomedCore + TrialCore client, 4-step fallback chain
  prompts.py                    Claude system prompts (region / landing / chat)
  llm.py                        narrate_region, narrate_landing, chat_turn (3 tools)
bucket_access/                  Hetzner S3 helpers (provided by hackathon org)
.streamlit/secrets.toml         API keys (gitignored)
data_cache/                     downloaded data, populated on first run (gitignored)
```

## Features

- **Volcano** with `streamlit-plotly-events` click → updates drill-down.
- **Sidebar filters**: corrected-p slider, |log2fc| slider, hierarchy-level
  multiselect, "significant only" toggle, guided tour (top 5 by
  `|log2fc| × −log10(p)`).
- **Drill-down**: stats metrics, per-animal strip + box, coronal triptych
  (anatomy base, magma G001/G002, diverging RdBu_r diff, red region contour).
- **Literature panel** (Amass): paper list with JUFO + citation badges,
  broadened-search badge when the fallback chain fires, expandable linked
  clinical trials from TrialCore.
- **Claude narrative** per region (4 sentences: what the region does → what the
  data shows → connection to cited literature → clinical relevance).
- **Chat** with three tools: `query_stats`, `get_region_detail`, `plot_region`
  (the last triggers an `st.rerun()` so the drill-down updates inline).

## Common things to tweak

- **Model**: change `MODEL` in `modules/llm.py` (default `claude-haiku-4-5`).
- **Fallback search chain**: edit the `chain` list in `modules/amass.py::search_region_literature`.
- **Tour size**: `_top_tour_regions(n=5)` in `app.py`.
- **Slice axis**: `render_triptych` slices along Y (coronal). Swap the indexing
  in `modules/nifti.py` for sagittal/axial.
