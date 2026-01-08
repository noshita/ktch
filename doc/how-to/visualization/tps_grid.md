---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-tps-grid)=

# Draw TPS Deformation Grid

Visualize shape deformation using thin plate spline grids.

## Basic TPS grid

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from ktch.plot import tps_grid_2d_plot

# Reference shape: unit square with corner and edge landmarks
# Use slight offsets to avoid grid points coinciding with landmarks
reference = np.array([
    [0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98],
    [0.52, 0.02], [0.98, 0.52], [0.52, 0.98], [0.02, 0.52],
], dtype=float)

# Target shape: slightly deformed
target = reference.copy()
target[2] = [1.12, 1.08]
target[6] = [0.57, 1.03]

fig, ax = plt.subplots(figsize=(6, 6))
tps_grid_2d_plot(reference, target, grid_size=0.1, ax=ax)
ax.set_aspect('equal')
plt.show()
```

```{seealso}
- {doc}`../../tutorials/landmark/thin_plate_spline` for detailed examples
```
