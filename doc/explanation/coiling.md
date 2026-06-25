(coiling)=

# Theoretical morphological models of coiling

The `ktch.coiling` module provides morphological analyses using 
theoretical morphological models of shell coiling. For
hands-on use, see the tutorials linked at the end.

## Theoretical morphological models

A theoretical morphological model is a model that mimics 
morphologies and morphogenesis: it maps (a few interpretable) parameters to 
a form (parameters to form). The parameters span a theoretical morphospace, 
so the model can describe not only forms that exist but also those
that are geometrically possible yet unobserved. 

In practice the main interest is usually generating forms (generative map).
ktch currently provides the generative map as `inverse_transform`, 
and parameter estimation from measurement data is planned. 

## Coiling models

The `ktch.coiling` module describes an unbranched accretionary tube: a
generating curve (the aperture shape) swept along a generating spiral (also
called the growth trajectory), the space curve traced by a reference point as the
organism grows.

The module covers forms from a straight tube to a tightly coiled tube, including
shells, horns, tusks, and beaks. 

### Raup's model

Raup's model (Raup & Michelson 1965) represents a shell as a generating curve
growing along a logarithmic (equiangular) spiral. 

The tube radius and the locus of the generating curve's reference point are

$$
r_R(\theta) = r_0\, W_R^{\theta / 2\pi},
$$

$$
p_R(\theta) = r_0\, W_R^{\theta / 2\pi}\, R_z(\theta)
\begin{pmatrix} \dfrac{2 D_R}{1 - D_R} + 1 \\ 0 \\ 2 T_R \left( \dfrac{D_R}{1 - D_R} + 1 \right) \end{pmatrix},
$$

where $\theta$ is the coiling angle and $R_z(\theta)$ is the rotation about the
coiling axis (Noshita 2014). The form is determined by the whorl expansion rate 
$W_R$ (`w_r`, $W_R > 1$), the translation rate $T_R$ (`t_r`), and 
the relative distance of the generating curve from the axis 
$D_R$ (`d_r`, $-1 < D_R < 1$); $r_0$ is the initial radius (`r0`). 
ktch implements this model as the `raup` function and the `RaupModel` class.

Raup's model, and the morphospace studies it enabled, are the origin
of theoretical morphology (Raup & Michelson 1965, Raup 1966).

### Growing tube model

The growing tube model takes a local, differential-geometric view: the shell is
a tube of radius $r(s)$ swept along a trajectory, and the form follows from how
the trajectory and the radius change at each growth stage $s$. 
A Frenet frame $(\xi_1, \xi_2, \xi_3)$ and the radius evolve as

$$
\frac{d}{ds} \begin{pmatrix} \xi_1 \\ \xi_2 \\ \xi_3 \end{pmatrix}
= \begin{pmatrix} 0 & C_G & 0 \\ -C_G & 0 & T_G \\ 0 & -T_G & 0 \end{pmatrix}
\begin{pmatrix} \xi_1 \\ \xi_2 \\ \xi_3 \end{pmatrix},
\qquad
\frac{dr}{ds} = E_G\, r,
$$

and the trajectory integrates the tangent,
$p_G(s) = p_0 + \int_0^s r(s')\, \xi_1(s')\, ds'$, with the arc length $l$
satisfying $dl/ds = r$. Here $E_G$ is the expansion rate (`e_g`), and $C_G$
(`c_g`) and $T_G$ (`t_g`) are the standardized curvature and torsion, that is,
curvature and torsion expressed relative to the tube radius. ktch implements this
as the `growing_tube` function and the `GrowingTubeModel` class. 

The model was introduced by Okamoto (1988) to analyse heteromorph ammonoids ($E_G$ is the logarithm of Okamoto's
original expansion rate $E$; Noshita 2014). Because the parameters are specified
locally, each may vary along growth, passed as a function of $s$ rather than a
constant. This yields heteromorph (irregularly coiled) shells, such as the
meandering ammonite *Nipponites*, which no constant parameter set can describe.

## Relationships between models

The coiling models are closely related. A future revision of this page will
describe their correspondence, including the conversion between their parameters.

## References

- Okamoto, T., 1988. Analysis of heteromorph ammonoids by differential geometry.
  Palaeontology 31, 35–52.
- Noshita, K., 2014. Quantification and geometric analysis of coiling patterns in
  gastropod shells based on 3D and 2D image data. Journal of Theoretical Biology
  363, 93–104.
- Raup, D.M., 1966. Geometric analysis of shell coiling: general problems.
  Journal of Paleontology 40, 1178–1190.
- Raup, D.M., Michelson, A., 1965. Theoretical morphology of the coiled shell.
  Science 147, 1294–1295.

## See also

- {doc}`../tutorials/coiling/raup_model` and
  {doc}`../tutorials/coiling/growing_tube_model` for generating shells with each
  model.
- {doc}`../api/coiling` for the API reference.
