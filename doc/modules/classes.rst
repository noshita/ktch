.. _api_ref:

=============
API Reference
=============

This is the class and function reference of ktch.

.. _landmark_ref:

:mod:`ktch.landmark`: landmark classes and utility functions
===================================================================


.. automodule:: ktch.landmark
   :no-members:
   :no-inherited-members:

Classes
---------

.. currentmodule:: ktch

.. autosummary::
   :toctree: generated/
   :template: class.rst

   landmark.GeneralizedProcrustesAnalysis

Functions
---------

.. currentmodule:: ktch

.. autosummary::
   :toctree: generated/
   :template: function.rst

   landmark.centroid_size
   landmark.tps_grid_2d_plot


.. _outline_ref:

:mod:`ktch.outline`: outline classes and utility functions
===================================================================


.. automodule:: ktch.outline
   :no-members:
   :no-inherited-members:

Classes
---------

.. currentmodule:: ktch

.. autosummary::
   :toctree: generated/
   :template: class.rst

   outline.EllipticFourierAnalysis

Functions
---------

.. currentmodule:: ktch

.. autosummary::
   :toctree: generated/
   :template: function.rst

   outline.spharm

:mod:`ktch.io`: I/O utility functions
===================================================================

Functions
---------

.. currentmodule:: ktch

.. autosummary::
   :toctree: generated/
   :template: function.rst

   io.read_tps
   io.write_tps

:mod:`ktch.datasets`: datasets utility functions
===================================================================

Functions
---------

.. currentmodule:: ktch

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.load_landmark_mosquito_wings
   datasets.load_outline_mosquito_wings
   datasets.convert_coords_df_to_list
   datasets.convert_coords_df_to_df_sklearn_transform
   .. datasets.load_outline_bottles
   .. datasets.load_coefficient_bottles


Plotting
---------

.. currentmodule:: ktch

.. autosummary::
   :toctree: generated/
   :template: class.rst

   outline.PCContribDisplay
