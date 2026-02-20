.. _image_passiflora_leaves_dataset:

Passiflora Leaf Scan Images dataset
------------------------------------

**Dataset Characteristics:**

===================   ===================================
Classes               10 (species)
Samples total         25 (scan images)
Image size            1268 x 1748 pixels, RGB
Image format          PNG
Features              uint8, [0, 255]
Total size            ~45 MB
License               CC0 (Public Domain)
===================   ===================================

A subset of the *Passiflora* leaf scan dataset [1]_ [2]_, containing 25
images of 10 species selected to span various leaf morphologies from simple
elliptical to deeply lobed forms.

Each image is a flatbed scan of multiple leaves from one plant individual,
arranged from tip (youngest) to base (oldest). Images have been resized to
1/4 of the original resolution using area-based interpolation.

Metadata (``metadata.csv``):

================  ====  ================================================
Column            Type  Description
================  ====  ================================================
``image_id``      str   Filename stem (e.g., ``Pcae1_1_8``)
``abbreviation``  str   4-letter species code (e.g., ``Pcae``)
``genus``         str   Genus name (always ``Passiflora``)
``species``       str   Specific epithet (e.g., ``caerulea``)
``individual_id`` str   Plant identifier (e.g., ``Pcae1``)
``leaf_start``    int   First leaf number in the scan (tip side)
``leaf_end``      int   Last leaf number in the scan (base side)
================  ====  ================================================

Species:

==============  ====================  ===================  ======  ======
Abbreviation    Species               Leaf morphology      Images  Leaves
==============  ====================  ===================  ======  ======
``Pcor``        *P. coriacea*         Simple elliptical         1       7
``Prub``        *P. rubra*            2-lobed                   1      10
``Pmal``        *P. malacophylla*     Simple round              2      11
``Pedu``        *P. edulis*           3-lobed                   5      20
``Pcae``        *P. caerulea*         3-lobed                   2      16
``Pcin``        *P. cincinnata*       3-lobed (deep)            4      14
``Pmis``        *P. misera*           Deep 2-lobed              1      14
``Pgra``        *P. gracilis*         3-lobed (narrow)          3      18
``Pcri``        *P. cristalina*       5-lobed                   4      19
``Psub``        *P. suberosa*         Variable 3-lobed          2       9
==============  ====================  ===================  ======  ======

References:

.. [1] Chitwood, D. H. and Otoni, W. C. (2017).
   "Morphometric analysis of *Passiflora* leaves: the relationship between
   landmarks of the vasculature and elliptical Fourier descriptors of the
   blade." *GigaScience*, 6(1), giw008.
   https://doi.org/10.1093/gigascience/giw008

.. [2] Chitwood, D. H. and Otoni, W. C. (2016).
   "Supporting data for 'Morphometric analysis of Passiflora leaves'."
   *GigaScience Database*.
   https://doi.org/10.5524/100251

CC0 1.0 Universal (Public Domain Dedication).
