"""Brain Insight Explorer — Streamlit entry.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from modules import amass, llm
from modules.data import load_all
from modules.nifti import render_triptych
from modules.viz import strip_figure, tour_bar_figure, volcano_figure

try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except ImportError:
    HAS_PLOTLY_EVENTS = False


st.set_page_config(
    page_title="Brain Insight Explorer",
    page_icon="🧠",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

data = load_all()
stats: pd.DataFrame = data["stats"]
quant_long: pd.DataFrame = data["quant_long"]
hier: pd.DataFrame = data["hier"]
nii = data["nii"]


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

ss = st.session_state
ss.setdefault("selected_acronym", None)
ss.setdefault("chat_history", [])
ss.setdefault("chat_display", [])  # list of {role, text} for UI
ss.setdefault("tour_idx", 0)
ss.setdefault("tour_active", False)
ss.setdefault("evidence_for", None)  # acronym the user requested an evidence summary for


def _label_id_for(acronym: str) -> int | None:
    """Map acronym → integer label in regions.nii.gz.

    Prefer stats.region_id (already on the row the user clicked); fall back to atlas CSV.
    """
    sub = stats[stats["acronym"] == acronym]
    if not sub.empty and "region_id" in sub.columns and pd.notna(sub.iloc[0]["region_id"]):
        try:
            return int(sub.iloc[0]["region_id"])
        except (TypeError, ValueError):
            pass
    id_col = "id" if "id" in hier.columns else ("label" if "label" in hier.columns else None)
    if id_col and "acronym" in hier.columns:
        row = hier[hier["acronym"] == acronym]
        if not row.empty:
            try:
                return int(row.iloc[0][id_col])
            except (TypeError, ValueError):
                return None
    return None


def _stats_row(acronym: str) -> pd.Series | None:
    sub = stats[stats["acronym"] == acronym]
    if sub.empty:
        return None
    return sub.iloc[0]


def _sibling_rows(row: pd.Series, k: int = 3) -> list[pd.Series]:
    """Pick a few related regions by hierarchy_level for narrative context."""
    if "hierarchy_level" not in stats.columns:
        return []
    level = row.get("hierarchy_level")
    if pd.isna(level):
        return []
    pool = stats[(stats["hierarchy_level"] == level) & (stats["acronym"] != row["acronym"])]
    if pool.empty:
        return []
    pool = pool.assign(_d=(pool["log2_fold_change"] - row["log2_fold_change"]).abs())
    return [r for _, r in pool.nsmallest(k, "_d").iterrows()]


def _top_tour_regions(n: int = 5) -> list[str]:
    df = stats.copy()
    df = df.dropna(subset=["log2_fold_change", "p_corrected"])
    p_safe = df["p_corrected"].clip(lower=1e-300)
    df = df.assign(_score=df["log2_fold_change"].abs() * (-np.log10(p_safe)))
    return df.sort_values("_score", ascending=False).head(n)["acronym"].tolist()


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    max_p = st.slider(
        "Max corrected p-value",
        min_value=0.001, max_value=0.20, value=0.05, step=0.005, format="%.3f",
    )
    min_abs_lfc = st.slider(
        "Min |log2 fold change|",
        min_value=0.0, max_value=3.0, value=0.0, step=0.05,
    )

    hierarchy_levels = sorted(stats["hierarchy_level"].dropna().unique().tolist()) if "hierarchy_level" in stats.columns else []
    chosen_levels = st.multiselect(
        "Atlas hierarchy levels",
        hierarchy_levels,
        default=hierarchy_levels,
    ) if hierarchy_levels else []

    only_sig = st.checkbox("Only corrected-significant regions", value=False)

    st.divider()
    st.caption("Groups")
    st.markdown("🔵 **G001 — Vehicle**\n\n🔴 **G002 — Semaglutide**")

    st.divider()
    st.subheader("Guided tour")
    if st.button("Start tour" if not ss["tour_active"] else "Restart tour"):
        ss["tour_active"] = True
        ss["tour_idx"] = 0
        tour = _top_tour_regions(5)
        if tour:
            ss["selected_acronym"] = tour[0]
            ss["_tour_list"] = tour
    if ss["tour_active"] and ss.get("_tour_list"):
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("◀ Prev", use_container_width=True):
                ss["tour_idx"] = max(0, ss["tour_idx"] - 1)
                ss["selected_acronym"] = ss["_tour_list"][ss["tour_idx"]]
        with col_b:
            if st.button("Next ▶", use_container_width=True):
                ss["tour_idx"] = min(len(ss["_tour_list"]) - 1, ss["tour_idx"] + 1)
                ss["selected_acronym"] = ss["_tour_list"][ss["tour_idx"]]
        st.caption(f"{ss['tour_idx']+1} / {len(ss['_tour_list'])} — {ss['selected_acronym']}")


# ----------------------------------------------------------------------------
# Filtered view
# ----------------------------------------------------------------------------

view = stats.copy()
view = view.dropna(subset=["log2_fold_change", "p_corrected"])
if chosen_levels and "hierarchy_level" in view.columns:
    view = view[view["hierarchy_level"].isin(chosen_levels)]
if only_sig and "significant_corrected" in view.columns:
    view = view[view["significant_corrected"].fillna(False).astype(bool)]


# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------

st.title("🧠 Brain Insight Explorer")
st.caption("Semaglutide vs Vehicle (c-Fos) — point-and-click drill-down with literature-grounded narratives.")

with st.container(border=True):
    try:
        summary_key = llm.landing_summary_payload(view)
        landing_md = llm.narrate_landing(summary_key)
    except Exception as e:
        landing_md = f"_Landing narrative unavailable: {e}_"
    st.markdown(landing_md)


# ----------------------------------------------------------------------------
# Main two-column layout: volcano (left), drill-down (right)
# ----------------------------------------------------------------------------

col_left, col_right = st.columns([1.1, 1.0], gap="large")

with col_left:
    st.subheader("Volcano — click a dot to explore")
    fig = volcano_figure(view, max_p=max_p, min_abs_lfc=min_abs_lfc)

    if HAS_PLOTLY_EVENTS:
        events = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=540,
            key="volcano",
        )
        if events:
            ev = events[0]
            # customdata in plotly is index-aligned. Recover by matching x+y.
            try:
                x = ev.get("x")
                y = ev.get("y")
                near = view.assign(
                    _neglog10=-np.log10(view["p_corrected"].clip(lower=1e-300))
                )
                near = near.assign(_d=(near["log2_fold_change"] - x).abs() + (near["_neglog10"] - y).abs())
                ss["selected_acronym"] = near.sort_values("_d").iloc[0]["acronym"]
            except Exception:
                pass
    else:
        st.plotly_chart(fig, use_container_width=True)
        st.warning("`streamlit-plotly-events` not installed — click-to-drill disabled. "
                   "`pip install streamlit-plotly-events` for the full experience.")

    # Manual region picker as a fallback / convenience.
    picker_options = view.sort_values("p_corrected")["acronym"].tolist()
    if picker_options:
        default_idx = picker_options.index(ss["selected_acronym"]) if ss["selected_acronym"] in picker_options else 0
        chosen = st.selectbox("…or pick a region:", picker_options, index=default_idx, key="region_picker")
        if chosen != ss["selected_acronym"]:
            ss["selected_acronym"] = chosen


with col_right:
    acronym = ss["selected_acronym"]
    if acronym is None:
        st.info("Click a dot in the volcano (or use the picker) to drill in.")
    else:
        row = _stats_row(acronym)
        if row is None:
            st.warning(f"No statistics row for {acronym}.")
        else:
            st.subheader(f"{row.get('region_name', acronym)} · {acronym}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("log2 FC", f"{row['log2_fold_change']:+.2f}")
            m2.metric("p (corrected)", f"{row['p_corrected']:.2g}")
            m3.metric("mean G002 / G001", f"{row.get('mean_A', float('nan')):.0f} / {row.get('mean_B', float('nan')):.0f}")
            m4.metric("n (G002/G001)", f"{int(row.get('n_A_eff', 0))} / {int(row.get('n_B_eff', 0))}")

            st.plotly_chart(strip_figure(quant_long, acronym), use_container_width=True)


st.divider()


# ----------------------------------------------------------------------------
# Spatial triptych + literature + narrative
# ----------------------------------------------------------------------------

acronym = ss["selected_acronym"]
if acronym is not None:
    row = _stats_row(acronym)
    if row is not None:
        st.subheader(f"Spatial view — {row.get('region_name', acronym)}")
        label_id = _label_id_for(acronym)
        if label_id is None:
            st.info("No atlas label found for this acronym; spatial view unavailable.")
        else:
            try:
                fig = render_triptych(nii, label_id, row.get("region_name", acronym))
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Triptych render failed: {e}")

        # On-demand evidence: button → Amass citations first, then Claude summary
        st.markdown("### 🔬 Literature evidence")
        st.caption(
            "Pull papers from Amass BiomedCore that link this region to Semaglutide / "
            "GLP-1, then summarize what they say."
        )
        if st.button(
            f"Find evidence for {row.get('region_name', acronym)}",
            type="primary",
            key=f"evidence_btn_{acronym}",
        ):
            ss["evidence_for"] = acronym

        if ss.get("evidence_for") == acronym:
            with st.spinner("Searching Amass literature…"):
                try:
                    lit = amass.search_region_literature(
                        row.get("region_name", acronym),
                        drug="semaglutide",
                        k=5,
                    )
                except Exception as e:
                    lit = {"papers": [], "query_used": "", "broadened": False}
                    st.warning(f"Amass error: {e}")

            papers = lit.get("papers", [])

            # 1) Citations FIRST — always shown above the summary.
            st.markdown("#### Citations (Amass · BiomedCore)")
            if lit.get("broadened"):
                st.info(f"Broadened to: `{lit.get('query_used')}`")
            elif lit.get("query_used"):
                st.caption(f"Query: `{lit.get('query_used')}`")

            if not papers:
                st.caption("No papers returned for this region.")
            else:
                for i, p in enumerate(papers[:5], start=1):
                    d = amass.paper_display(p)
                    title = d["title"]
                    if d.get("url"):
                        st.markdown(f"**[{i}] [{title}]({d['url']})**")
                    else:
                        st.markdown(f"**[{i}] {title}**")
                    meta = " · ".join(s for s in [
                        d.get("authors") or "",
                        d.get("journal") or "",
                        str(d.get("year") or ""),
                    ] if s)
                    badges = []
                    if d.get("jufo"):
                        badges.append(f"JUFO {d['jufo']}")
                    if d.get("citations"):
                        badges.append(f"{d['citations']} citations")
                    if badges:
                        meta += " · " + " · ".join(badges)
                    st.caption(meta)

                with st.expander("Linked clinical trials"):
                    try:
                        trials = amass.linked_trials(papers, max_trials=3)
                    except Exception as e:
                        trials = []
                        st.caption(f"Trial lookup error: {e}")
                    if not trials:
                        st.caption("No linked trials found in TrialCore.")
                    else:
                        for t in trials:
                            td = amass.trial_display(t)
                            cond = td["conditions"]
                            cond_str = ", ".join(cond) if isinstance(cond, list) else str(cond)
                            st.markdown(f"**{td['title']}**")
                            st.caption(
                                " · ".join(s for s in [
                                    f"Phase: {td['phase']}" if td["phase"] else "",
                                    f"Status: {td['status']}" if td["status"] else "",
                                    cond_str,
                                    td.get("nct") or "",
                                ] if s)
                            )

            # 2) Evidence summary AFTER the citations — grounded in those same papers.
            st.markdown("#### Evidence summary")
            with st.spinner("Synthesizing evidence from the citations above…"):
                try:
                    narrative = llm.narrate_region(row, papers, _sibling_rows(row))
                except Exception as e:
                    narrative = f"_LLM unavailable: {e}_"
            st.markdown(narrative)


st.divider()


# ----------------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------------

st.subheader("Ask the data")
st.caption(
    "Examples: *Which hindbrain regions went up most?* · "
    "*Show me regions where Semaglutide reduced activation* · "
    "*What's the most surprising significant region?*"
)

for turn in ss["chat_display"]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])

user_msg = st.chat_input("Ask about the data…")
if user_msg:
    ss["chat_display"].append({"role": "user", "text": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                text, directive = llm.chat_turn(ss["chat_history"], user_msg, stats)
            except Exception as e:
                text = f"_Chat error: {e}_"
                directive = {}
        st.markdown(text)
    ss["chat_display"].append({"role": "assistant", "text": text})
    if directive.get("plot_region"):
        ss["selected_acronym"] = directive["plot_region"]
        st.rerun()


st.divider()
st.caption(
    "Data: Vibraint c-Fos imaging (Semaglutide vs Vehicle). "
    "Literature: Amass BiomedCore + TrialCore. Narratives: Claude (claude-haiku-4-5)."
)
