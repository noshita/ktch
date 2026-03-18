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

data = read_spharmpdm_coef(sample_coef_path)
print(f"Specimen name: {data.specimen_name}")
print(f"Maximum degree (l_max): {data.l_max}")
print(f"Coefficient array shape: {data.to_numpy().shape}")
```

## Coefficient structure

SPHARM-PDM coefficients are organized by spherical harmonic degree:

```{code-cell} ipython3
# Each element contains coefficients for one harmonic degree
for i, coef in enumerate(data.coeffs[:3]):
    print(f"Degree {i}: shape {coef.shape}")
```

```{seealso}
- {doc}`../../explanation/harmonic` for background on spherical harmonic analysis
```
