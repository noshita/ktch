---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-load-tps)=

# Load TPS Files

TPS is a standard format for landmark data, commonly used with tpsDig software.

## Read landmarks from a TPS file

```{code-cell} ipython3
import tempfile
from ktch.io import read_tps, write_tps

# Create a sample TPS file for demonstration
tps_content = """LM=4
0.0 0.0
1.0 0.0
1.0 1.0
0.0 1.0
ID=specimen_001

LM=4
0.1 0.0
1.1 0.0
1.0 1.1
0.0 1.0
ID=specimen_002
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.tps', delete=False) as f:
    f.write(tps_content)
    tps_path = f.name

# Read the TPS file
landmarks = read_tps(tps_path)
print(f"Shape: {landmarks.shape}")  # (n_specimens, n_landmarks, n_dim)
```

## Write landmarks to a TPS file

```{code-cell} ipython3
import numpy as np

# Prepare data
landmarks = np.array([
    [[0, 0], [1, 0], [1, 1], [0, 1]],
    [[0.1, 0], [1.1, 0], [1, 1.1], [0, 1]],
], dtype=float)

# Write to TPS file
with tempfile.NamedTemporaryFile(mode='w', suffix='.tps', delete=False) as f:
    output_path = f.name

write_tps(output_path, landmarks)

# Verify
coords = read_tps(output_path)
print(f"Written and read back: {coords.shape}")
```

```{seealso}
- {doc}`use_datasets` for built-in example datasets
- {doc}`../../tutorials/landmark/gpa_from_tps` for a complete TPS workflow
```
