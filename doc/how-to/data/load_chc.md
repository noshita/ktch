---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-load-chc)=

# Load CHC Files

CHC (Chain Code) format stores outline data as directional codes.

## Chain code format

Chain codes represent 2D contours using directional codes from 0 to 7:

```
3 2 1
4 * 0
5 6 7
```

The file format is:
```
[Sample name] [X] [Y] [Area (mm2) per pixel] [Area (pixels)] [Chain code] -1
```

## Read and write chain codes

```{code-cell} ipython3
import tempfile
from pathlib import Path
import numpy as np
from ktch.io import read_chc, write_chc

# Create a sample chain code (a small square)
# 0=right, 2=up, 4=left, 6=down
chain_code = np.array([0, 0, 2, 2, 4, 4, 6, 6])

# Write to a temporary file
with tempfile.TemporaryDirectory() as tmpdir:
    chc_path = Path(tmpdir) / "sample.chc"
    write_chc(chc_path, chain_code, sample_names="square")

    # Read as coordinates (default)
    coords = read_chc(chc_path)
    print(f"Coordinates shape: {coords.shape}")
    print(f"Coordinates:\n{coords}")
```

## Read as raw chain codes

```{code-cell} ipython3
with tempfile.TemporaryDirectory() as tmpdir:
    chc_path = Path(tmpdir) / "sample.chc"
    write_chc(chc_path, chain_code, sample_names="square")

    # Read as raw chain codes
    raw_codes = read_chc(chc_path, as_coordinates=False)
    print(f"Raw chain codes: {raw_codes}")
```

## Read as DataFrame

```{code-cell} ipython3
with tempfile.TemporaryDirectory() as tmpdir:
    chc_path = Path(tmpdir) / "sample.chc"
    write_chc(chc_path, chain_code, sample_names="square")

    # Read as pandas DataFrame
    df = read_chc(chc_path, as_frame=True)
    print(df)
```

```{seealso}
- {doc}`use_datasets` for built-in outline datasets
- {doc}`../../tutorials/harmonic/elliptic_Fourier_analysis` for EFA workflow
```
