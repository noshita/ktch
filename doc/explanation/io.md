(io)=

# I/O and format conversion

The `ktch.io` module handles reading and writing morphometrics file formats and converting between file-format representations and analysis-ready data structures.

## Module structure

The module provides:

- Readers and writers (`read_tps`, `write_nef`, etc.) that parse or serialize specific file formats
- Data containers (`NefData`, `SpharmPdmData`, etc.) that hold parsed data in a format-faithful representation
- Converter functions that bridge between file-format containers and the coefficient formats used by analysis classes

Readers, writers, and data containers are organized by file format. Converters are in a single internal module (`_converters.py`) for discoverability.

## Converters

Bridge converters translate between a data container and the flat coefficient array that a `ktch.harmonic` transformer produces or consumes:

```python
from ktch.io import (
    nef_to_efa_coeffs, efa_coeffs_to_nef,           # NEF <-> EFA
    spharmpdm_to_sha_coeffs, sha_coeffs_to_spharmpdm,  # SPHARM-PDM <-> SHA
)
```

Each pair follows the naming pattern `{format}_to_{tool}_coeffs` / `{tool}_coeffs_to_{format}`, where `format` refers to the file format and `tool` refers to the analysis class.

A typical workflow:

```python
from ktch.io import read_spharmpdm_coef, spharmpdm_to_sha_coeffs
from ktch.harmonic import SphericalHarmonicAnalysis

# Read file -> data container -> analysis-ready coefficients
data = read_spharmpdm_coef("specimen.coef")
sha_coeffs = spharmpdm_to_sha_coeffs(data)

# Reconstruct surface
sha = SphericalHarmonicAnalysis(n_harmonics=data.l_max)
surface = sha.inverse_transform(sha_coeffs.reshape(1, 3, -1))
```

Two utility functions convert coordinate DataFrames (from `read_tps` or `read_chc` with `as_frame=True`) into formats suitable for analysis: `convert_coords_df_to_list` produces a list of per-specimen arrays, and `convert_coords_df_to_df_sklearn_transform` produces a wide DataFrame compatible with sklearn transformers.

Low-level format packing and unpacking functions (e.g., converting between SPHARM-PDM's interleaved storage layout and a structured coefficient list) are private (`_` prefix) and not part of the public API. Bridge converters call them internally, so users do not need to interact with them directly.

## Data containers

Each file format has a corresponding dataclass that holds parsed data:

| Container | File format | Content |
|---|---|---|
| `NefData` | `.nef` | Normalized elliptic Fourier descriptors |
| `ChainCodeData` | `.chc` | Freeman chain codes |
| `TPSData` | `.tps` | Landmark coordinates |
| `SpharmPdmData` | `.coef` | SPHARM-PDM spherical harmonic coefficients |

All containers implement the `MorphoData` protocol, providing `to_numpy()`, `to_dataframe()`, and the `__array__` interface.

Data containers store coefficients in a representation faithful to the file format. For example, `SpharmPdmData.coeffs` uses complex-valued arrays reflecting the SPHARM-PDM convention, even though `SphericalHarmonicAnalysis` internally uses real spherical harmonics. The bridge converter handles the basis conversion.

## Separation from analysis modules

Converter functions live in `ktch.io`, not in the analysis modules (`ktch.harmonic`, `ktch.landmark`, etc.). The `io` module does not import from any analysis module, and analysis modules do not import from `io`.

This separation follows the ports-and-adapters pattern. Each analysis module defines its own input/output interface (the "port") — for example, `SphericalHarmonicAnalysis` expects a flat float64 coefficient array, and `GeneralizedProcrustesAnalysis` expects a list of coordinate arrays. The `io` module provides adapters that translate between external file formats and these interfaces. File-format details do not leak into the analysis modules, and analysis modules do not need to know about any particular file format.

When a bridge converter requires domain-specific logic (e.g., complex-to-real spherical harmonic basis conversion for SPHARM-PDM), the formulas are implemented locally in `_converters.py` rather than imported from the analysis module. This avoids circular dependencies and keeps the modules independently testable.

```{seealso}
- {doc}`harmonic` for background on harmonic analysis methods
- {doc}`../api/io` for API reference
- {doc}`../how-to/data/load_spharm` for loading SPHARM-PDM data
```
