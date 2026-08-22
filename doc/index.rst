ktch: model-based morphometrics in Python
==========================================

ktch is written for researchers and engineers who need to quantify
morphological properties, whatever the field.

It is a Python package for model-based morphometrics: the quantitative study of
morphological properties and diversity through explicit models. It covers
theoretical morphology models, harmonic descriptors, and landmark methods.
Although each approach assumes a different model and applies to objects of
different dimensions, topologies, and structures, they all share one underlying
idea. ktch implements that idea as a scikit-learn compatible API. Thus the
methods compose into pipelines, and a study can quantify different aspects of
morphological properties within one workflow.

ktch provides the functionality specific to morphometrics and leaves the rest
to the Python data analysis ecosystem. An analysis can therefore keep using the
tools it already relies on for preprocessing, model selection, and
visualization.

The :doc:`elliptic Fourier analysis tutorial <tutorials/harmonic/elliptic_Fourier_analysis>`
shows what this looks like in practice, including the scikit-learn estimator API
(e.g., ``fit_transform``).


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: How-to Guides

   how-to/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Explanation

   explanation/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   api/index


Installation
-------------------------------------------------

Get started by :doc:`installing ktch <installation>`. ktch is available from
`PyPI <https://pypi.org/project/ktch/>`_ and
`conda-forge <https://anaconda.org/conda-forge/ktch>`_.

Tutorials
-------------------------------------------------

:doc:`Step-by-step guides <tutorials/index>` for learning ktch through
hands-on examples, from landmark methods to harmonic analysis and coiling
models.

How-to guides
-------------------------------------------------

:doc:`Task-oriented guides <how-to/index>` for common operations, such as
reading morphometric file formats and visualizing results.

Explanation
-------------------------------------------------

:doc:`Conceptual explanations <explanation/index>` of morphometric methods
and theory, including what each method assumes about the target morphological
properties.

API reference
-------------------------------------------------

:doc:`Complete API documentation <api/index>` for all classes, functions,
and modules.

Getting help
-------------------------------------------------

Bug reports, feature requests, and questions are welcome via
`GitHub Issues <https://github.com/noshita/ktch/issues>`_.
