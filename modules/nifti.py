"""Spatial slice rendering: triptych of G001 / G002 / diff with region overlay."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .data import NiftiBundle


def region_centroid(regions: np.ndarray, label_id: int) -> tuple[int, int, int]:
    coords = np.argwhere(regions == label_id)
    if coords.size == 0:
        return (regions.shape[0] // 2, regions.shape[1] // 2, regions.shape[2] // 2)
    cz, cy, cx = coords.mean(axis=0).astype(int)
    return int(cz), int(cy), int(cx)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min/max normalize a 2D slice for grayscale display."""
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a[np.isfinite(a)], (1, 99)) if np.isfinite(a).any() else (0, 1)
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def render_triptych(
    bundle: NiftiBundle,
    label_id: int,
    region_name: str,
) -> Figure:
    """Coronal slice triptych at the centroid Y of the region.

    Returns a single matplotlib Figure with three panels: G001 median, G002 median, diff.
    Anatomy as grayscale base. Diff overlay uses RdBu_r centered at 0. Region outlined in red.
    """
    centroid = bundle.centroids.get(int(label_id))
    if centroid is None:
        centroid = region_centroid(bundle.regions, label_id)
    cz, cy, cx = centroid

    # Coronal = slice along Y axis (anterior-posterior).
    y = int(np.clip(cy, 0, bundle.anatomy.shape[1] - 1))

    anatomy_sl = bundle.anatomy[:, y, :]
    g001_sl    = bundle.g001[:, y, :]
    g002_sl    = bundle.g002[:, y, :]
    diff_sl    = bundle.diff[:, y, :]
    mask_sl    = (bundle.regions[:, y, :] == label_id)

    anat_norm = _normalize(anatomy_sl)
    g001_norm = _normalize(g001_sl)
    g002_norm = _normalize(g002_sl)

    diff_abs = np.nanpercentile(np.abs(diff_sl), 99) if np.isfinite(diff_sl).any() else 1.0
    if not np.isfinite(diff_abs) or diff_abs <= 0:
        diff_abs = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    fig.patch.set_facecolor("white")
    titles = ["Vehicle (G001)", "Semaglutide (G002)", "Diff (G002 − G001)"]
    bases = [g001_norm, g002_norm, anat_norm]

    for ax, base, title in zip(axes, bases, titles):
        ax.imshow(anat_norm, cmap="gray", origin="lower")
        if title.startswith("Diff"):
            ax.imshow(
                diff_sl,
                cmap="RdBu_r",
                vmin=-diff_abs,
                vmax=diff_abs,
                origin="lower",
                alpha=0.75,
            )
        else:
            ax.imshow(base, cmap="magma", origin="lower", alpha=0.55)
        if mask_sl.any():
            ax.contour(mask_sl.astype(float), levels=[0.5], colors="red", linewidths=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{region_name} — coronal slice y={y}", fontsize=12)
    return fig
