"""Claude calls: narrative paragraphs + chat-with-tools."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import streamlit as st

from . import amass
from .prompts import CHAT_SYSTEM, LANDING_SYSTEM, REGION_SYSTEM

MODEL = "claude-haiku-4-5"


def _client():
    import anthropic

    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        key = None
    key = key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured (in .streamlit/secrets.toml or env).")
    return anthropic.Anthropic(api_key=key)


def _row_to_dict(row: pd.Series) -> dict:
    keep = [
        "acronym", "region_name", "hierarchy_level", "log2_fold_change",
        "p_corrected", "p_value", "mean_A", "mean_B", "median_A", "median_B",
        "n_A_eff", "n_B_eff", "significant_corrected",
    ]
    out = {}
    for k in keep:
        if k in row and pd.notna(row[k]):
            v = row[k]
            if hasattr(v, "item"):
                v = v.item()
            out[k] = v
    return out


def _papers_compact(papers: list[dict], max_n: int = 5) -> list[dict]:
    out = []
    for i, p in enumerate(papers[:max_n], start=1):
        d = amass.paper_display(p)
        snippet = (d.get("abstract") or "").strip().replace("\n", " ")[:500]
        out.append({
            "n": i,
            "title": d["title"],
            "journal": d["journal"],
            "year": d["year"],
            "abstract_snippet": snippet,
        })
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def narrate_landing(stats_records: tuple) -> str:
    """Hero blurb. Cached on a small derived summary, not the full DataFrame."""
    # stats_records is a hashable tuple of (n_sig, top_up, top_down) — caller pre-computes.
    n_sig, top_up, top_down = stats_records
    payload = {
        "n_significant_corrected": n_sig,
        "top_upregulated":   top_up,
        "top_downregulated": top_down,
    }
    try:
        client = _client()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=LANDING_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )
        return _join_text(resp.content)
    except Exception as e:
        return _landing_fallback(payload, str(e))


def landing_summary_payload(stats: pd.DataFrame) -> tuple:
    """Pre-compute a small hashable summary for cache stability."""
    sig = stats[stats["significant_corrected"].fillna(False).astype(bool)]
    n_sig = int(len(sig))
    top_up = sig.nlargest(5, "log2_fold_change")[["acronym", "region_name", "log2_fold_change"]]
    top_down = sig.nsmallest(5, "log2_fold_change")[["acronym", "region_name", "log2_fold_change"]]
    return (
        n_sig,
        tuple((r.acronym, r.region_name, round(float(r.log2_fold_change), 2)) for r in top_up.itertuples()),
        tuple((r.acronym, r.region_name, round(float(r.log2_fold_change), 2)) for r in top_down.itertuples()),
    )


def _landing_fallback(payload: dict, err: str) -> str:
    top_up = ", ".join(f"{a} ({lfc:+.2f})" for a, _n, lfc in payload["top_upregulated"][:3])
    top_dn = ", ".join(f"{a} ({lfc:+.2f})" for a, _n, lfc in payload["top_downregulated"][:3])
    return (
        f"**{payload['n_significant_corrected']} regions reached corrected significance.** "
        f"Top up in Semaglutide: {top_up}. Top down: {top_dn}. "
        "Click any dot to explore. _(LLM offline: showing computed summary.)_"
    )


def narrate_region(
    row: pd.Series,
    papers: list[dict],
    siblings: list[pd.Series] | None = None,
) -> str:
    sibs = [_row_to_dict(s) for s in (siblings or [])]
    payload = {
        "region": {
            "name": row.get("region_name", ""),
            "acronym": row.get("acronym", ""),
            "hierarchy_level": row.get("hierarchy_level", None),
        },
        "stats": _row_to_dict(row),
        "siblings": sibs,
        "literature": _papers_compact(papers),
    }
    try:
        client = _client()
        resp = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=REGION_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )
        return _join_text(resp.content)
    except Exception as e:
        return (
            f"_LLM unavailable ({e})._ "
            f"**{payload['region']['name']}** ({payload['region']['acronym']}): "
            f"log2fc {payload['stats'].get('log2_fold_change', 0):+.2f}, "
            f"p_corrected {payload['stats'].get('p_corrected', float('nan')):.2g}. "
            f"{len(payload['literature'])} related papers shown below."
        )


def _join_text(content_blocks) -> str:
    return "".join(b.text for b in content_blocks if getattr(b, "type", None) == "text").strip()


# ---------------------------------------------------------------------------
# Tool-use chat
# ---------------------------------------------------------------------------

CHAT_TOOLS = [
    {
        "name": "query_stats",
        "description": (
            "Search the statistics table for brain regions matching filters. "
            "Returns up to top_n regions sorted by |log2_fold_change| × −log10(p_corrected)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_log2fc": {"type": "number", "description": "Minimum signed log2 fold change (use negative for down-regulation)."},
                "max_log2fc": {"type": "number", "description": "Maximum signed log2 fold change."},
                "min_abs_log2fc": {"type": "number", "description": "Minimum absolute log2 fold change."},
                "max_p_corrected": {"type": "number", "description": "Maximum corrected p-value (e.g. 0.05)."},
                "acronym_contains": {"type": "string", "description": "Substring filter on region acronym."},
                "name_contains": {"type": "string", "description": "Substring filter on region_name (case-insensitive)."},
                "hierarchy_level": {"type": "integer", "description": "Restrict to this atlas hierarchy level."},
                "top_n": {"type": "integer", "description": "Max regions to return. Default 10."},
                "direction": {"type": "string", "enum": ["up", "down", "either"], "description": "Filter by direction of change."},
            },
        },
    },
    {
        "name": "get_region_detail",
        "description": "Fetch full statistics + literature snippet for one region by acronym.",
        "input_schema": {
            "type": "object",
            "properties": {"acronym": {"type": "string"}},
            "required": ["acronym"],
        },
    },
    {
        "name": "plot_region",
        "description": "Ask the app to render the drill-down panel for a region (volcano click equivalent).",
        "input_schema": {
            "type": "object",
            "properties": {"acronym": {"type": "string"}},
            "required": ["acronym"],
        },
    },
]


def _tool_query_stats(stats: pd.DataFrame, **kw) -> list[dict]:
    df = stats.copy()
    if (v := kw.get("min_log2fc")) is not None:
        df = df[df["log2_fold_change"] >= v]
    if (v := kw.get("max_log2fc")) is not None:
        df = df[df["log2_fold_change"] <= v]
    if (v := kw.get("min_abs_log2fc")) is not None:
        df = df[df["log2_fold_change"].abs() >= v]
    if (v := kw.get("max_p_corrected")) is not None:
        df = df[df["p_corrected"] <= v]
    if (v := kw.get("acronym_contains")):
        df = df[df["acronym"].astype(str).str.contains(v, case=False, na=False)]
    if (v := kw.get("name_contains")):
        df = df[df["region_name"].astype(str).str.contains(v, case=False, na=False)]
    if (v := kw.get("hierarchy_level")) is not None and "hierarchy_level" in df.columns:
        df = df[df["hierarchy_level"] == v]
    direction = kw.get("direction", "either")
    if direction == "up":
        df = df[df["log2_fold_change"] > 0]
    elif direction == "down":
        df = df[df["log2_fold_change"] < 0]
    top_n = int(kw.get("top_n") or 10)

    p_safe = df["p_corrected"].clip(lower=1e-300)
    df = df.assign(_score=df["log2_fold_change"].abs() * _neglog10(p_safe))
    df = df.sort_values("_score", ascending=False).head(top_n)
    keep = ["acronym", "region_name", "log2_fold_change", "p_corrected", "mean_A", "mean_B",
            "n_A_eff", "n_B_eff", "significant_corrected"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].to_dict(orient="records")
    for r in out:
        for k, v in list(r.items()):
            if hasattr(v, "item"):
                r[k] = v.item()
    return out


def _neglog10(s: pd.Series) -> pd.Series:
    import numpy as np
    return -np.log10(s)


def _tool_get_region_detail(stats: pd.DataFrame, acronym: str) -> dict:
    sub = stats[stats["acronym"] == acronym]
    if sub.empty:
        return {"error": f"No region with acronym '{acronym}'."}
    row = sub.iloc[0]
    res = amass.search_region_literature(row.get("region_name", acronym))
    papers = [amass.paper_display(p) for p in res.get("papers", [])][:3]
    return {
        "stats": _row_to_dict(row),
        "literature": papers,
        "broadened_search": res.get("broadened", False),
        "query_used": res.get("query_used"),
    }


def _run_tool(name: str, args: dict, stats: pd.DataFrame) -> Any:
    if name == "query_stats":
        return _tool_query_stats(stats, **args)
    if name == "get_region_detail":
        return _tool_get_region_detail(stats, args.get("acronym", ""))
    if name == "plot_region":
        return {"acronym": args.get("acronym", ""), "rendered": True}
    return {"error": f"unknown tool {name}"}


def chat_turn(history: list[dict], user_msg: str, stats: pd.DataFrame) -> tuple[str, dict]:
    """One conversational turn with tool use.

    Returns (assistant_markdown, render_directive).
      - history is updated IN PLACE with the new user+assistant turn (Anthropic message format).
      - render_directive is {} or {"plot_region": "<acronym>"} if the model called plot_region.
    """
    client = _client()
    history.append({"role": "user", "content": user_msg})

    render_directive: dict = {}
    # Up to 4 tool-use loops, then break.
    for _ in range(4):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=CHAT_SYSTEM,
            tools=CHAT_TOOLS,
            messages=history,
        )

        # Append the assistant's full response (text + tool_use blocks) to history.
        assistant_blocks = [b.model_dump() for b in resp.content]
        history.append({"role": "assistant", "content": assistant_blocks})

        if resp.stop_reason != "tool_use":
            return _join_text(resp.content) or "_(empty response)_", render_directive

        # Execute every tool_use block, collect results, send back as tool_result message.
        tool_results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            args = block.input or {}
            try:
                result = _run_tool(block.name, args, stats)
            except Exception as e:
                result = {"error": f"tool {block.name} crashed: {e}"}
            if block.name == "plot_region" and isinstance(result, dict) and result.get("acronym"):
                render_directive["plot_region"] = result["acronym"]
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })
        history.append({"role": "user", "content": tool_results})

    return "_(stopped after 4 tool-use rounds)_", render_directive
