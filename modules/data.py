"""Data loading: bucket fetch (cached), CSVs, NIfTI bundle, centroid pickle."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from bucket_access.bucket_utils import download_file

DATA_CACHE = Path(__file__).resolve().parent.parent / "data_cache"
DATA_CACHE.mkdir(exist_ok=True)

# Bucket key -> local filename
BUCKET_FILES = {
    "challengeB/tabular_data_quantification/cfos_object_density_quantification.csv":
        "cfos_quantification.csv",
    "challengeB/tabular_data_quantification/cfos_object_density_statistics_G002_vs_G001.csv":
        "cfos_statistics.csv",
    "challengeB/spatial_brain_maps/atlas_regions.csv":
        "atlas_hierarchy.csv",
    "challengeB/spatial_brain_maps/brain_atlas_anatomy.nii.gz":
        "anatomy.nii.gz",
    "challengeB/spatial_brain_maps/brain_atlas_regions.nii.gz":
        "regions.nii.gz",
    "challengeB/spatial_brain_maps/cfos_G001_median.nii.gz":
        "cfos_G001.nii.gz",
    "challengeB/spatial_brain_maps/cfos_G002_median.nii.gz":
        "cfos_G002.nii.gz",
    "challengeB/spatial_brain_maps/cfos_group_median_difference_G002_vs_G001.nii.gz":
        "diff_map.nii.gz",
}


def _ensure_file(s3_key: str, local_name: str) -> Path:
    local_path = DATA_CACHE / local_name
    if not local_path.exists():
        download_file(s3_key, str(local_path))
    return local_path


def ensure_all_files() -> dict[str, Path]:
    """Download every Challenge B asset once. Returns local-name -> Path."""
    paths = {}
    for s3_key, local_name in BUCKET_FILES.items():
        paths[local_name] = _ensure_file(s3_key, local_name)
    return paths


@dataclass
class NiftiBundle:
    anatomy: np.ndarray   # (Z, Y, X)
    regions: np.ndarray   # (Z, Y, X) int
    g001: np.ndarray      # (Z, Y, X)
    g002: np.ndarray      # (Z, Y, X)
    diff: np.ndarray      # (Z, Y, X)
    spacing: tuple        # (x, y, z) mm
    centroids: dict       # label_id -> (z, y, x)


def _load_nifti_bundle(paths: dict[str, Path]) -> NiftiBundle:
    import SimpleITK as sitk

    def arr(name):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(paths[name])))

    anatomy_img = sitk.ReadImage(str(paths["anatomy.nii.gz"]))
    spacing = anatomy_img.GetSpacing()

    regions = arr("regions.nii.gz").astype(np.int32)

    centroid_pkl = DATA_CACHE / "centroids.pkl"
    if centroid_pkl.exists():
        with open(centroid_pkl, "rb") as f:
            centroids = pickle.load(f)
    else:
        centroids = _compute_centroids(regions)
        with open(centroid_pkl, "wb") as f:
            pickle.dump(centroids, f)

    return NiftiBundle(
        anatomy=sitk.GetArrayFromImage(anatomy_img),
        regions=regions,
        g001=arr("cfos_G001.nii.gz"),
        g002=arr("cfos_G002.nii.gz"),
        diff=arr("diff_map.nii.gz"),
        spacing=spacing,
        centroids=centroids,
    )


def _compute_centroids(regions: np.ndarray) -> dict:
    """For every non-zero label_id, compute the (z, y, x) centroid of the mask."""
    centroids: dict[int, tuple[int, int, int]] = {}
    labels = np.unique(regions)
    for lab in labels:
        if lab == 0:
            continue
        coords = np.argwhere(regions == lab)
        if len(coords) == 0:
            continue
        cz, cy, cx = coords.mean(axis=0).astype(int)
        centroids[int(lab)] = (int(cz), int(cy), int(cx))
    return centroids


@st.cache_resource(show_spinner="Downloading data + loading volumes…")
def load_all() -> dict:
    """Single source of truth. Returns {'stats', 'quant', 'quant_long', 'hier', 'nii'}."""
    paths = ensure_all_files()

    stats = pd.read_csv(paths["cfos_statistics.csv"])
    quant = pd.read_csv(paths["cfos_quantification.csv"])
    hier = pd.read_csv(paths["atlas_hierarchy.csv"])

    id_cols = [c for c in ["scan_name", "animal_nr", "group_nr"] if c in quant.columns]
    region_cols = [c for c in quant.columns if c not in id_cols]
    quant_long = quant.melt(
        id_vars=id_cols,
        value_vars=region_cols,
        var_name="acronym",
        value_name="density",
    )

    nii = _load_nifti_bundle(paths)

    return {
        "stats": stats,
        "quant": quant,
        "quant_long": quant_long,
        "hier": hier,
        "nii": nii,
        "paths": paths,
    }
