"""
bucket_access/bucket_utils.py

Read-only utilities for accessing hackathon data from the Hetzner object storage bucket.
All data lives under challengeA/ and challengeB/ in the bucket.

Challenge A — image patches and embeddings:
    list_files(prefix)                          list files in bucket
    download_file(s3_key, local_path)           download any file
    read_h5_patches(s3_key)                     load patches + metadata directly from bucket
    read_h5_embeddings(s3_key)                  load embeddings directly from bucket
    get_h5_info_remote(s3_key)                  inspect raw brain H5 metadata without downloading
    read_h5_slice_remote(s3_key, ...)           read a subregion of a raw brain H5

Challenge B — tabular data and spatial brain maps:
    list_files(prefix)                          list files in bucket
    download_file(s3_key, local_path)           download CSVs and NIfTI files locally
"""

import h5py
import numpy as np
import pandas as pd
import s3fs
from boto3 import client
from botocore.client import Config

from .config import (
    HETZNER_ACCESS_KEY,
    HETZNER_SECRET_KEY,
    HETZNER_BUCKET_NAME,
    HETZNER_ENDPOINT,
)

# boto3 client for simple operations
s3 = client(
    's3',
    endpoint_url=HETZNER_ENDPOINT,
    aws_access_key_id=HETZNER_ACCESS_KEY,
    aws_secret_access_key=HETZNER_SECRET_KEY,
    config=Config(signature_version='s3v4'),
)

BUCKET = HETZNER_BUCKET_NAME


# ============================================================================
# General utilities
# ============================================================================

def list_files(prefix: str = '') -> list:
    """
    List files in the bucket under a given prefix.

    Args:
        prefix: Path prefix to filter by, e.g. 'challengeA/patches/'

    Returns:
        List of S3 keys (file paths)

    Example:
        list_files('challengeA/patches/')
        list_files('challengeB/')
    """
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    if 'Contents' not in response:
        return []
    return [obj['Key'] for obj in response['Contents']]


def download_file(s3_key: str, local_path: str):
    """
    Download a file from the bucket to a local path.

    Args:
        s3_key:     Path in bucket, e.g. 'challengeB/tabular_data_quantification/cfos_statistics.csv'
        local_path: Local destination path, e.g. 'cfos_statistics.csv'

    Example:
        download_file('challengeB/tabular_data_quantification/cfos_object_density_statistics_G002_vs_G001.csv',
                      'statistics.csv')
    """
    s3.download_file(BUCKET, s3_key, local_path)
    print(f"✓ Downloaded → {local_path}")


# ============================================================================
# Challenge A — patches and embeddings
# ============================================================================

def read_h5_patches(s3_key: str):
    """
    Load patches and metadata from a patches H5 file directly from the bucket.
    Downloads the full file into memory — patches files are ~50 MB each.

    Args:
        s3_key: Path in bucket, e.g. 'challengeA/patches/{scan_name}_patches.h5'

    Returns:
        patches:  (N, 256, 256) uint16 numpy array
        metadata: pandas DataFrame with one row per patch
        attrs:    dict of file-level attributes (scan_name, condition, voxel_size, etc.)

    Example:
        patches, metadata, attrs = read_h5_patches(
            'challengeA/patches/260219_AN0B7_G002_mouse_brain_MB1_SCAN0_16-11-05_patches.h5'
        )
        print(patches.shape)       # (N, 256, 256)
        print(attrs['condition'])  # 'Semaglutide'
    """
    fs = _get_s3fs()
    s3_path = f'{BUCKET}/{s3_key}'

    with fs.open(s3_path, 'rb') as f:
        with h5py.File(f, 'r') as h:
            patches  = h['patches'][:]
            metadata = pd.DataFrame(h['metadata'][:])
            attrs    = dict(h.attrs)

    print(f"✓ Loaded {patches.shape[0]} patches from {s3_key.split('/')[-1]}")
    return patches, metadata, attrs


def read_h5_embeddings(s3_key: str):
    """
    Load embeddings from an embeddings H5 file directly from the bucket.
    Embeddings files are small (~2 MB each).

    Args:
        s3_key: Path in bucket, e.g. 'challengeA/embeddings/{scan_name}_embeddings.h5'

    Returns:
        embeddings: (N, 512) float32 numpy array, L2 normalized
        attrs:      dict of file-level attributes (scan_name, model, normalized, etc.)

    Note:
        Embeddings are L2 normalized — cosine similarity between two patches
        equals their dot product: sim = embeddings[i] @ embeddings[j]

    Example:
        embeddings, attrs = read_h5_embeddings(
            'challengeA/embeddings/260219_AN0B7_G002_mouse_brain_MB1_SCAN0_16-11-05_embeddings.h5'
        )
        print(embeddings.shape)   # (N, 512)

        # cosine similarity
        sim = embeddings[0] @ embeddings[1]
    """
    fs = _get_s3fs()
    s3_path = f'{BUCKET}/{s3_key}'

    with fs.open(s3_path, 'rb') as f:
        with h5py.File(f, 'r') as h:
            embeddings = h['embeddings'][:]
            attrs      = dict(h.attrs)

    print(f"✓ Loaded {embeddings.shape[0]} embeddings from {s3_key.split('/')[-1]}")
    return embeddings, attrs


# ============================================================================
# Challenge A — raw whole-brain volumes (reference only, use slices not full files)
# ============================================================================

def get_h5_info_remote(s3_key: str) -> dict:
    """
    Get shape and dtype of a raw brain H5 volume without downloading it.

    Args:
        s3_key: Path in bucket, e.g. 'challengeA/raw_whole_brain_data/{scan_name}.h5'

    Returns:
        dict with keys: shape, dtype, size_gb, all_keys

    Example:
        info = get_h5_info_remote('challengeA/raw_whole_brain_data/260219_AN0B7_..._SCAN0_16-11-05.h5')
        print(info['shape'])    # (1498, 2878, 2000)
        print(info['size_gb'])  # ~5.0
    """
    fs = _get_s3fs()
    s3_path = f'{BUCKET}/{s3_key}'

    with fs.open(s3_path, 'rb') as f:
        with h5py.File(f, 'r') as h:
            all_keys = list(h.keys())
            if 'data' not in h:
                return {'has_data': False, 'all_keys': all_keys}
            ds = h['data']
            return {
                'has_data': True,
                'shape':    ds.shape,
                'dtype':    str(ds.dtype),
                'size_gb':  ds.size * ds.dtype.itemsize / (1024 ** 3),
                'all_keys': all_keys,
            }


def read_h5_slice_remote(
    s3_key:  str,
    z_range: tuple = None,
    y_range: tuple = None,
    x_range: tuple = None,
) -> np.ndarray:
    """
    Read a subregion of a raw brain H5 volume without downloading the full file.
    Volume shape is (Z, Y, X).

    Args:
        s3_key:  Path in bucket, e.g. 'challengeA/raw_whole_brain_data/{scan_name}.h5'
        z_range: (start, end) tuple for Z axis, or None for full range
        y_range: (start, end) tuple for Y axis, or None for full range
        x_range: (start, end) tuple for X axis, or None for full range

    Returns:
        numpy array of the requested subregion, dtype uint16

    Examples:
        # single Z slice at position 500
        sl = read_h5_slice_remote(key, z_range=(500, 501))

        # 3D slab: Z 400-600, full Y and X
        slab = read_h5_slice_remote(key, z_range=(400, 600))

        # specific XY region for all Z
        roi = read_h5_slice_remote(key, y_range=(1000, 1500), x_range=(800, 1300))
    """
    fs = _get_s3fs()
    s3_path = f'{BUCKET}/{s3_key}'

    with fs.open(s3_path, 'rb') as f:
        with h5py.File(f, 'r') as h:
            if 'data' not in h:
                raise KeyError(f"'data' key not found in {s3_key}. Available: {list(h.keys())}")
            z_sl = slice(*z_range) if z_range else slice(None)
            y_sl = slice(*y_range) if y_range else slice(None)
            x_sl = slice(*x_range) if x_range else slice(None)
            data = h['data'][z_sl, y_sl, x_sl]

    print(f"✓ Read slice {data.shape} from {s3_key.split('/')[-1]}")
    return data


# ============================================================================
# Internal
# ============================================================================

def _get_s3fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        key=HETZNER_ACCESS_KEY,
        secret=HETZNER_SECRET_KEY,
        client_kwargs={'endpoint_url': HETZNER_ENDPOINT},
    )
