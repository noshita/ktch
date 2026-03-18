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
import numpy as np
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
tps_data = read_tps(tps_path)
print(f"Number of specimens: {len(tps_data)}")
print(f"First specimen: {tps_data[0].specimen_name}")
print(f"Landmark shape: {tps_data[0].to_numpy().shape}")

# Stack into a single array if needed
landmarks = np.array([t.to_numpy() for t in tps_data])
print(f"Stacked shape: {landmarks.shape}")  # (n_specimens, n_landmarks, n_dim)
```

## Write landmarks to a TPS file

```{code-cell} ipython3
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
tps_data = read_tps(output_path)
print(f"Written and read back: {len(tps_data)} specimens")
```

```{seealso}
- {doc}`use_datasets` for built-in example datasets
- {doc}`../../tutorials/landmark/gpa_from_tps` for a complete TPS workflow
```
