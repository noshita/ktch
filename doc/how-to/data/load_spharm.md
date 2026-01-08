---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-load-spharm)=

# Load SPHARM Coefficients

ktch can read spherical harmonic coefficients from SPHARM-PDM format files.

## Read SPHARM-PDM coefficients

The `.coef` file is an output of the `ParaToSPHARMMesh` step of [SPHARM-PDM](https://www.nitrc.org/projects/spharm-pdm).

```{code-cell} ipython3
from ktch.io import read_spharmpdm_coef

# Path to a sample .coef file (from ktch test data)
sample_coef_path = "../../../ktch/io/tests/data/andesred_07_allSegments_SPHARM.coef"

coefficients = read_spharmpdm_coef(sample_coef_path)
print(f"Number of coefficient arrays: {len(coefficients)}")
print(f"First array shape: {coefficients[0].shape}")
```

## Coefficient structure

SPHARM-PDM coefficients are organized by spherical harmonic degree:

```{code-cell} ipython3
# Each array contains coefficients for one harmonic degree
for i, coef in enumerate(coefficients[:3]):
    print(f"Degree {i}: shape {coef.shape}")
```

```{seealso}
- {doc}`../../explanation/harmonic` for background on spherical harmonic analysis
```
