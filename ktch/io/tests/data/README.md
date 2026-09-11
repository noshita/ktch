# Test data

## SPHARM-PDM coefficient files

`*_SPHARM.coef` is SPHARM-PDM's output with the parameter sphere aligned
to the first-order ellipsoid (right-handed axes, no flip template).
`*_SPHARM_ellalign.coef` is the same specimen with the coordinates
additionally rotated onto the ellipsoid axes, degree 0 zeroed, no scaling. The pairs are regression fixtures for
`SphericalHarmonicRegistration(method="first_order", align_parameter=False)`.

| specimen | source | l_max | note |
|---|---|---|---|
| `andesred_07_allSegments` | potato tuber, ktch examples | 25 | raw only, near-symmetric |
| `B_keratocytes000703` | Simionato et al. (2021) red blood cells, `spharmDegree=15`, `subdiv=10` | 15 | clearly triaxial (semi-axis ratios 1.96, 1.56) |
| `0.33_echinocyte_I000346` | same | 15 | clearly triaxial (1.60, 2.15) |
| `-1.00_spherocyte000090` | same | 15 | near-symmetric, two shortest axes equal (1.12, 1.00) |
| `0.33_echinocyte_I000325` | same | 15 | near-symmetric, two longest axes equal (1.00, 2.09) |

Simionato, G. et al. (2021) Red blood cell phenotyping from 3D confocal
images using artificial neural networks. PLoS Computational Biology
17(5): e1008934. Coefficients computed with SPHARM-PDM.
