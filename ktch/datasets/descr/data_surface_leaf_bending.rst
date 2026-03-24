.. _surface_leaf_bending_dataset:

Synthetic 3D Leaf Bending Surface Dataset
------------------------------------------

**Dataset Characteristics:**

============================   ==================================
Specimens                      60
Mesh format                    OFF (Object File Format)
Mesh types                     surface mesh + parameter mesh
Coordinate dimensionality      3
Features                       real
Total size                     ~77 MB (zip)
============================   ==================================

A synthetic dataset of 3D wheat-like leaf surfaces with bending
deformation, corresponding to the outline dataset
:func:`~ktch.datasets.load_outline_leaf_bending`.

Each specimen consists of two triangular meshes stored as OFF files:

- ``surf_bending_three/leaf_NNN.off`` -- the 3D surface mesh.
- ``surf_para_bending_three/leaf_NNN.off`` -- the disk parameterization
  mesh (vertices on the unit disk, z = 0).

The parameter mesh provides a mapping from the unit disk to the 3D
surface, suitable as input for Disk Harmonic Analysis (DHA).

Mesh topologies (vertex and face counts) may vary across specimens.

Metadata (``meta_surface_leaf_bending.csv``):

====================  =====  =============================================
Column                Type   Description
====================  =====  =============================================
``specimen_id``       int    Specimen identifier (1--60)
``lmax``              float  Maximum leaf length
``c``                 float  Shape parameter c
``alpha``             float  Bending amplitude
``A``                 float  Scaling factor
``alphaB``            float  Bending angle
``kB``                float  Bending curvature
``alpha_random``      float  Randomized bending amplitude
``alphaB_random``     float  Randomized bending angle
====================  =====  =============================================

The 60 specimens span 6 parameter combinations (3 ``alphaB`` values x
2 ``alpha`` values), with 10 replicates each.
