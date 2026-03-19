---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-load-nef)=

# Load NEF Files

NEF (Normalized Elliptic Fourier) format stores normalized elliptic Fourier
descriptors produced by SHAPE software's `Chc2Nef` program.

## Read NEF coefficients

```{code-cell} ipython3
import tempfile
import numpy as np
from ktch.io import read_nef, write_nef, NefData

# Create a sample NEF file for demonstration
nef_content = """#CONST 1 1 1 0
#HARMO 3
specimen_001
  1.0 0.0 0.0 0.0
  0.5 0.1 0.2 0.3
  0.1 0.05 0.02 0.01
specimen_002
  1.0 0.0 0.0 0.0
  0.4 0.2 0.1 0.2
  0.08 0.03 0.01 0.02
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.nef', delete=False) as f:
    f.write(nef_content)
    nef_path = f.name

# Read the NEF file
nef_data = read_nef(nef_path)
print(f"Number of specimens: {len(nef_data)}")
print(f"First specimen: {nef_data[0].specimen_name}")
print(f"Harmonics: {nef_data[0].n_harmonics}")
print(f"Coefficient shape: {nef_data[0].to_numpy().shape}")
```

## Convert to EFA coefficients

NEF data can be converted to flat coefficient vectors compatible with
`EllipticFourierAnalysis.inverse_transform()`:

```{code-cell} ipython3
from ktch.io import nef_to_efa_coeffs

efa_coeffs = nef_to_efa_coeffs(nef_data)
print(f"EFA coefficient shape: {efa_coeffs.shape}")
# Layout: [a_0..a_n, b_0..b_n, c_0..c_n, d_0..d_n]
```

## Write NEF coefficients

```{code-cell} ipython3
from ktch.io import efa_coeffs_to_nef

# Convert back to NefData
nef_list = efa_coeffs_to_nef(efa_coeffs, specimen_names=["sp1", "sp2"])

# Write to file
with tempfile.NamedTemporaryFile(mode='w', suffix='.nef', delete=False) as f:
    output_path = f.name

write_nef(output_path, [n.coeffs for n in nef_list], sample_names=["sp1", "sp2"])

# Verify
result = read_nef(output_path)
print(f"Written and read back: {len(result)} specimens")
```

```{seealso}
- {doc}`load_chc` for reading outline chain codes (input for SHAPE's Chc2Nef)
- {doc}`../../tutorials/harmonic/elliptic_Fourier_analysis` for EFA workflow
```
