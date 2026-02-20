---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-use-datasets)=

# Use Built-in Datasets

ktch includes example datasets for learning and testing. Datasets are
divided into two categories: bundled datasets shipped with the package,
and remote datasets downloaded on first use.

## Available datasets

### Bundled datasets

These are included in the package and require no extra dependencies.

| Function | Type | Description |
|----------|------|-------------|
| `load_landmark_mosquito_wings` | Landmarks | 18 landmarks from 127 mosquito wings |
| `load_landmark_trilobite_cephala` | Landmarks + Curves | 16 landmarks and 4 curves from 301 trilobite cephala |
| `load_outline_mosquito_wings` | Outlines | 100-point outlines from 126 mosquito wings |
| `load_outline_leaf_bending` | Outlines (3D) | 200-point 3D outlines from 60 simulated leaves |

### Remote datasets

These are downloaded and cached locally on first use.
Install the optional `data` extras first:

```bash
pip install ktch[data]
```

| Function | Type | Description |
|----------|------|-------------|
| `load_image_passiflora_leaves` | Images | 25 leaf scan images of 10 *Passiflora* species |

## Load landmark data

```{code-cell} ipython3
from ktch.datasets import load_landmark_mosquito_wings

data = load_landmark_mosquito_wings()

print(f"Coordinates shape: {data.coords.shape}")  # (n_specimens * n_landmarks, n_dim)
print(f"Metadata keys: {list(data.meta.keys())}")
```

## Load as pandas DataFrame

```{code-cell} ipython3
data = load_landmark_mosquito_wings(as_frame=True)

print("Coordinates DataFrame:")
print(data.coords.head())
```

## Load outline data

```{code-cell} ipython3
from ktch.datasets import load_outline_mosquito_wings

data = load_outline_mosquito_wings()
print(f"Outlines shape: {data.coords.shape}")  # (n_specimens * n_points, n_dim)
```

## Load remote datasets

Remote datasets support version pinning via the `version` parameter.
When omitted, the default version for the current ktch release is used.

```python
from ktch.datasets import load_image_passiflora_leaves

# Use the default version
data = load_image_passiflora_leaves()

# Pin to a specific version
data = load_image_passiflora_leaves(version="2")
```

The dataset is downloaded once and cached locally. Subsequent calls
load from the cache.

## Dataset description

Each dataset includes a description accessible via `data.DESCR`:

```{code-cell} ipython3
data = load_landmark_mosquito_wings()
print(data.DESCR[:200] + "...")
```

```{seealso}
- {mod}`ktch.datasets` API reference
```
