# Challenge B — Guided brain data exploration for biological insight

## The problem

Brain scans go through signal extraction and quantification in Vibraint's image analysis
pipeline. The final output consists of spreadsheets that summarize quantified signal per
brain region and sample, and statistical analysis results comparing experimental groups
across brain regions. This creates a rich but complex dataset that is difficult to
visualize in an informative and intuitive way, both across the whole brain and for
individual regions. Another challenge is how to identify, in an unbiased manner, the
brain regions where the differences between experimental groups are the most relevant.
In practice, the dataset is also simply large and difficult to query, navigate, and
interpret efficiently.

## Your goal

1. **Visualize** patterns across all brain regions and allow detailed inspection of
   individual regions
2. **Highlight automatically** the regions where differences between experimental groups
   are most pronounced or most biologically interesting
3. **Build a dashboard** that generates these visualizations on demand

**Extensions:**

- Add a natural language interface — let users ask questions about the data, request
  specific plots, or explore which brain regions were identified as most interesting
  between groups. As an example, a user could ask which brain regions are involved in
  hunger and satiety regulation, and the system would look up those regions in the study
  data to show whether they were differentially activated — combining LLM knowledge about
  neuroanatomy with the quantitative results from this experiment.
- Use the spatial brain maps (group median signal maps, difference map, anatomy, and
  region boundaries) to build richer visualizations — for example, when a user selects
  a region of interest from a plot, show a corresponding image of that region in both
  study groups using the spatial NIfTI data, so statistical results and spatial signal
  patterns can be inspected side by side.

---

## The biology (brief)

The dataset comes from mouse brains imaged with **light sheet fluorescence microscopy**
at 5×5×5 µm voxel resolution. The marker is **c-Fos**, a protein expressed by recently
active neurons — a proxy for neuronal activation across the whole brain. Two conditions:

- **G001 — Vehicle:** control mice
- **G002 — Semaglutide:** mice treated with semaglutide (active ingredient in Ozempic/Wegovy)

Vibraint's pipeline detected and quantified c-Fos positive cells across hundreds of
brain regions, and ran statistical comparisons between the two groups.

---

## Data

```
bucket/challengeB/
├── tabular_data_quantification/
│   ├── cfos_object_density_quantification.csv
│   └── cfos_object_density_statistics_G002_vs_G001.csv
└── spatial_brain_maps/
    ├── brain_atlas_anatomy.nii.gz
    ├── brain_atlas_regions.nii.gz
    ├── cfos_G001_median.nii.gz
    ├── cfos_G002_median.nii.gz
    ├── cfos_group_median_difference_G002_vs_G001.nii.gz
    └── atlas_hierarchy.csv
```

---

## Tabular data — `challengeB/tabular_data_quantification/`

### `cfos_object_density_quantification.csv`

c-Fos cell densities per brain region per animal. Rows = animals, columns = brain regions.

| Column | Description |
|--------|-------------|
| `scan_name` | Unique scan identifier |
| `animal_nr` | Animal identifier |
| `group_nr` | `G001` (Vehicle) or `G002` (Semaglutide) |
| *(remaining columns)* | Brain region acronyms (e.g. `PVT`, `CEA`, `BST`) containing c-Fos cell density in cells/mm³ |

```python
from bucket_access.bucket_utils import download_file
import pandas as pd

download_file('challengeB/tabular_data_quantification/cfos_object_density_quantification.csv',
              'cfos_quantification.csv')
df = pd.read_csv('cfos_quantification.csv')

# reshape to long format
regions = [c for c in df.columns if c not in ['scan_name', 'animal_nr', 'group_nr']]
df_long = df.melt(id_vars=['scan_name', 'animal_nr', 'group_nr'],
                  value_vars=regions, var_name='region', value_name='density')
```

### `cfos_object_density_statistics_G002_vs_G001.csv`

Statistical comparison between Semaglutide (G002) and Vehicle (G001), one row per brain region.

| Column | Description |
|--------|-------------|
| `region_id` | Numeric atlas region ID |
| `region_name` | Full brain region name |
| `acronym` | Short region identifier (e.g. `PVT`, `CEA`, `NTS`) |
| `hierarchy_level` | Depth in atlas hierarchy — higher = more specific subregion |
| `is_lowest_level` | `True` if leaf node (finest granularity) |
| `n_A`, `n_B` | Total number of animals in G002 and G001 |
| `n_A_eff`, `n_B_eff` | Effective sample size — animals where the region was present in the scan. A region can be missing due to sample damage or imperfect registration to the atlas. Statistics are computed on effective sample sizes only. |
| `mean_A`, `mean_B` | Mean density in G002 and G001 |
| `median_A`, `median_B` | Median density |
| `std_A`, `std_B` | Standard deviation |
| `sem_A`, `sem_B` | Standard error of the mean |
| `mean_diff` | Mean difference (G002 − G001) |
| `ci_low`, `ci_high` | 95% confidence interval of the difference |
| `fold_change` | Fold change G002 / G001 |
| `log2_fold_change` | Log2 fold change — positive = higher in Semaglutide |
| `log2fc_ci_low`, `log2fc_ci_high` | CI of log2 fold change |
| `p_value` | Uncorrected p-value |
| `p_corrected` | Corrected p-value |
| `significant_uncorrected` | Boolean — significant at p<0.05 uncorrected |
| `significant_corrected` | Boolean |

```python
download_file('challengeB/tabular_data_quantification/cfos_object_density_statistics_G002_vs_G001.csv',
              'cfos_statistics.csv')
stats = pd.read_csv('cfos_statistics.csv')

# significantly different regions (corrected)
sig  = stats[stats['significant_corrected'] == True].sort_values('p_corrected')

# top upregulated in Semaglutide
up   = stats.nlargest(20, 'log2_fold_change')[['acronym', 'region_name', 'log2_fold_change', 'p_corrected']]

# top downregulated
down = stats.nsmallest(20, 'log2_fold_change')[['acronym', 'region_name', 'log2_fold_change', 'p_corrected']]
```

---

## Spatial brain maps — `challengeB/spatial_brain_maps/`

NIfTI volumes (`.nii.gz`) registered to the mouse brain atlas space. All volumes share
the same coordinate space and can be overlaid directly.

| File | Description |
|------|-------------|
| `brain_atlas_anatomy.nii.gz` | Average mouse brain anatomy — structural reference |
| `brain_atlas_regions.nii.gz` | Region segmentation — integer label per voxel, maps to `atlas_hierarchy.csv` |
| `cfos_G001_median.nii.gz` | Median c-Fos signal map across Vehicle animals |
| `cfos_G002_median.nii.gz` | Median c-Fos signal map across Semaglutide animals |
| `cfos_group_median_difference_G002_vs_G001.nii.gz` | Difference map G002 − G001 — positive = higher in Semaglutide |
| `atlas_hierarchy.csv` | Brain region names, acronyms, and integer labels linking to the region segmentation |

The integer labels in `brain_atlas_regions.nii.gz` map to `atlas_hierarchy.csv` — use
this to go from a voxel label to a region name, or to extract a binary mask for a
specific region.

NIfTI files can be opened locally in [ITK-SNAP](http://www.itksnap.org) (free,
cross-platform) for interactive 3D inspection. Load the anatomy as the base image and
overlay the region segmentation or signal maps on top.

### Loading NIfTI files

```python
from bucket_access.bucket_utils import download_file
import SimpleITK as sitk
import numpy as np
import pandas as pd

download_file('challengeB/spatial_brain_maps/brain_atlas_anatomy.nii.gz',     'anatomy.nii.gz')
download_file('challengeB/spatial_brain_maps/brain_atlas_regions.nii.gz',     'regions.nii.gz')
download_file('challengeB/spatial_brain_maps/cfos_G001_median.nii.gz',        'cfos_G001.nii.gz')
download_file('challengeB/spatial_brain_maps/cfos_G002_median.nii.gz',        'cfos_G002.nii.gz')
download_file('challengeB/spatial_brain_maps/cfos_group_median_difference_G002_vs_G001.nii.gz',
                                                                               'diff_map.nii.gz')
download_file('challengeB/spatial_brain_maps/atlas_hierarchy.csv',            'atlas_hierarchy.csv')

# load volumes
anatomy_img  = sitk.ReadImage('anatomy.nii.gz')
regions_img  = sitk.ReadImage('regions.nii.gz')
cfos_g001    = sitk.ReadImage('cfos_G001.nii.gz')
cfos_g002    = sitk.ReadImage('cfos_G002.nii.gz')
diff_img     = sitk.ReadImage('diff_map.nii.gz')

# convert to numpy arrays — shape is (X, Y, Z) in SimpleITK
anatomy      = sitk.GetArrayFromImage(anatomy_img)   # returns (Z, Y, X)
regions      = sitk.GetArrayFromImage(regions_img).astype(int)
diff_map     = sitk.GetArrayFromImage(diff_img)

# voxel size in mm
voxel_size   = anatomy_img.GetSpacing()   # (x, y, z) in mm

# atlas hierarchy
hierarchy    = pd.read_csv('atlas_hierarchy.csv')

# coronal slice at midpoint (axis 1 = anterior-posterior in Z,Y,X array)
mid          = anatomy.shape[1] // 2
anatomy_slice = anatomy[:, mid, :]
diff_slice    = diff_map[:, mid, :]

# binary mask for a specific region by acronym
label_id     = hierarchy[hierarchy['acronym'] == 'PVT']['label'].values[0]
region_mask  = regions == label_id
```
