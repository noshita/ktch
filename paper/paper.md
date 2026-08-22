---
title: 'ktch: model-based morphometrics in Python'
tags:
  - Python
  - morphometrics
  - geometric morphometrics
  - shape analysis
  - theoretical morphology
  - scikit-learn
authors:
  - name: Koji Noshita
    orcid: 0000-0002-9510-5897
    affiliation: 1
affiliations:
  - name: Kyushu University, Japan
    index: 1
    ror: "00p4k0j84"
date: 22 August 2026
bibliography: paper.bib
---

<!--
Skeleton only; every section below is unwritten.
Target 1,200-1,500 words, within the 750-1,750 word limit JOSS enforces.
Word budgets per section are noted inline.
-->

# Summary

<!--
~150 words, non-specialist register, no jargon. Morphological variation is
quantified either by describing observed shapes or by generating them from
growth models; ktch provides both behind one scikit-learn-compatible API.
-->

# Statement of need

<!--
~250 words. The Python ecosystem has no integrated morphometrics stack;
researchers assemble one from single-purpose packages or leave Python for R.
Name the audience: biologists, palaeontologists, and morphometricians already
working in the Python scientific stack, plus the materials-science uptake.
-->

# State of the field

<!--
~350 words, the load-bearing section. Compare against the R packages that
define the field and the Python packages that overlap in part, then close with
an explicit build-vs-contribute justification on two axes: no Python package
covers landmark and harmonic methods behind one pipeline-composable API, and
no package in either language treats theoretical morphology models as fitted
estimators. Verify every package claim and DOI before it enters paper.bib.
-->

# Software design

<!--
~300 words. Why scikit-learn estimator semantics for morphometrics; what
fit/transform/inverse_transform mean for a shape method and what that buys
(pipelines, cross-validation, metadata routing); the trade-offs accepted in
return (ragged inputs sit awkwardly in the sklearn contract, auxiliary inputs
need metadata routing); why theoretical morphology models are transformers
estimating parameters from observed form rather than a separate simulation API.
-->

# Research impact statement

<!--
~200 words. Lead with the independent third-party publications, then the
workshop and teaching activity, naming dates, venues, and material links.
-->

# AI usage disclosure

<!--
~120 words. Tools and versions, where used (code, documentation, paper),
nature and scope of the assistance, and confirmation of human review and
design ownership. Vague wording is rated unacceptable.
-->

# Acknowledgements

<!-- ~60 words. Name the funding grants. -->

# References
