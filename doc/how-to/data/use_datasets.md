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

ktch includes example datasets for learning and testing.

## Available datasets

| Dataset | Type | Description |
|---------|------|-------------|
| `load_landmark_mosquito_wings` | Landmarks | 18 landmarks from 127 mosquito wings |
| `load_landmark_trilobite_cephala` | Landmarks + Curves | 16 landmarks and 4 curves from 301 trilobite cephala |
| `load_outline_mosquito_wings` | Outlines | 100-point outlines from 126 mosquito wings |

## Load landmark data

```{code-cell} ipython3
from ktch.datasets import load_landmark_mosquito_wings

data = load_landmark_mosquito_wings()

print(f"Coordinates shape: {data.coords.shape}")  # (n_specimens, n_landmarks, n_dim)
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
print(f"Outlines shape: {data.coords.shape}")  # (n_specimens, n_points, n_dim)
```

## Dataset description

Each dataset includes a description:

```{code-cell} ipython3
data = load_landmark_mosquito_wings()
print(data.DESCR[:200] + "...")
```

```{seealso}
- {mod}`ktch.datasets` API reference
```
