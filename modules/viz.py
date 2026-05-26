"""Plotly visualizations: volcano + per-animal strip plot + tour bar chart."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _safe_neglog10(p: pd.Series) -> pd.Series:
    p = p.clip(lower=1e-300)
    return -np.log10(p)


def volcano_figure(stats: pd.DataFrame, max_p: float = 0.05, min_abs_lfc: float = 0.0):
    """Volcano: log2_fold_change vs -log10(p_value). Color by uncorrected-p + |lfc| pass.

    Uncorrected p is used on the y-axis because Bonferroni-corrected p values in this
    dataset are all ≥ 0.43, which would compress every dot into a thin band near y=0.
    A point is highlighted (red/blue) if its uncorrected p ≤ max_p AND |log2fc| ≥
    min_abs_lfc; otherwise gray.

    Built with go.Scatter (not px.scatter) so streamlit-plotly-events can JSON-encode
    customdata reliably — NaN values in customdata via px silently break the render.
    """
    df = stats.copy()
    df = df.dropna(subset=["log2_fold_change", "p_value"])
    df["neg_log10_p"] = _safe_neglog10(df["p_value"])

    passes = (
        (df["p_value"] <= max_p)
        & (df["log2_fold_change"].abs() >= min_abs_lfc)
    )
    df["category"] = np.where(
        ~passes,
        "n.s.",
        np.where(df["log2_fold_change"] > 0, "Up in Semaglutide", "Down in Semaglutide"),
    )

    color_map = {
        "n.s.": "#bdbdbd",
        "Up in Semaglutide": "#d7263d",
        "Down in Semaglutide": "#1f77b4",
    }

    # Render-order: n.s. first, then significant on top.
    cat_order = ["n.s.", "Down in Semaglutide", "Up in Semaglutide"]
    fig = go.Figure()
    for cat in cat_order:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        # Build customdata as a 2-D array; replace NaN with "" so the JS bridge
        # in streamlit-plotly-events doesn't silently drop the trace.
        p_corr_str = sub["p_corrected"].apply(
            lambda v: f"{v:.2g}" if pd.notna(v) else "—"
        )
        custom = np.stack([
            sub["acronym"].astype(str).to_numpy(),
            sub["region_name"].astype(str).to_numpy(),
            sub["log2_fold_change"].to_numpy(),
            sub["p_value"].to_numpy(),
            p_corr_str.to_numpy(),
        ], axis=1)
        fig.add_trace(go.Scatter(
            x=sub["log2_fold_change"],
            y=sub["neg_log10_p"],
            mode="markers",
            name=cat,
            marker=dict(
                size=9,
                color=color_map[cat],
                line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
                "log2 FC: %{customdata[2]:.2f}<br>"
                "p (uncorrected): %{customdata[3]:.2g}<br>"
                "p (corrected): %{customdata[4]}"
                "<extra></extra>"
            ),
        ))
    fig.add_hline(
        y=-np.log10(max_p),
        line_dash="dot",
        line_color="gray",
        annotation_text=f"p = {max_p}",
        annotation_position="top right",
    )
    if min_abs_lfc > 0:
        fig.add_vline(x=min_abs_lfc,  line_dash="dot", line_color="gray")
        fig.add_vline(x=-min_abs_lfc, line_dash="dot", line_color="gray")
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
        xaxis_title="log2(fold change) — G002 / G001",
        yaxis_title="−log10(uncorrected p)",
    )
    fig.update_xaxes(zeroline=True, zerolinecolor="lightgray")
    fig.update_yaxes(zeroline=True, zerolinecolor="lightgray")
    return fig


def strip_figure(quant_long: pd.DataFrame, acronym: str):
    """Per-animal density strip+box plot for one region, split by group."""
    sub = quant_long[quant_long["acronym"] == acronym].dropna(subset=["density"])
    if sub.empty:
        return go.Figure().update_layout(title=f"No quantification rows for {acronym}")

    group_label = {"G001": "Vehicle (G001)", "G002": "Semaglutide (G002)"}
    sub = sub.assign(group=sub["group_nr"].map(group_label).fillna(sub["group_nr"]))

    fig = px.strip(
        sub,
        x="group",
        y="density",
        color="group",
        color_discrete_map={"Vehicle (G001)": "#1f77b4", "Semaglutide (G002)": "#d7263d"},
        hover_data={"animal_nr": True, "scan_name": True, "density": ":.1f"},
        labels={"density": "c-Fos density (cells / mm³)", "group": ""},
    )
    fig.update_traces(jitter=0.4, marker=dict(size=11, opacity=0.85))

    for g, color in [("Vehicle (G001)", "#1f77b4"), ("Semaglutide (G002)", "#d7263d")]:
        gs = sub[sub["group"] == g]["density"]
        if len(gs):
            fig.add_trace(
                go.Box(
                    x=[g] * len(gs),
                    y=gs,
                    boxpoints=False,
                    fillcolor="rgba(0,0,0,0)",
                    line=dict(color=color, width=1.5),
                    name=g,
                    showlegend=False,
                    width=0.4,
                )
            )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        plot_bgcolor="white",
    )
    return fig


def tour_bar_figure(stats: pd.DataFrame, acronyms: list[str]):
    """Horizontal bar chart of log2 fold change for the selected regions."""
    sub = stats[stats["acronym"].isin(acronyms)].copy()
    sub = sub.assign(_order=sub["acronym"].map({a: i for i, a in enumerate(acronyms)}))
    sub = sub.sort_values("_order")
    sub["color"] = np.where(sub["log2_fold_change"] >= 0, "#d7263d", "#1f77b4")

    fig = go.Figure(
        go.Bar(
            x=sub["log2_fold_change"],
            y=sub["acronym"],
            orientation="h",
            marker=dict(color=sub["color"]),
            text=sub["region_name"],
            hovertemplate="<b>%{y}</b> — %{text}<br>log2fc=%{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(220, 32 * len(sub)),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="log2(fold change)",
        plot_bgcolor="white",
    )
    fig.add_vline(x=0, line_color="black", line_width=1)
    return fig
