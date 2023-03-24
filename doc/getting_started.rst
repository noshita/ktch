#####################################
Getting Started
#####################################

ktch allows you to conduct model-based morphometrics via scikit-learn compatible API efficiently.

Install
===================================================

From PyPI
-------------------------------------

ktch is currently available on PyPI. You can install it with pip::

    $ pip install ktch


Quick start
===================================================

This example loads the mosquito wing outline dataset (Rohlf and Archie 1984), and calculates elliptic Fourier descriptors (EFDs).
Finally, it performs principal component analysis (PCA) on the EFDs.

::

    from sklearn.decomposition import PCA

    from ktch.datasets import load_outline_mosquito_wings
    from ktch.outline import EllipticFourierAnalysis

    # load data
    data_outline_mosquito_wings = load_outline_mosquito_wings()
    X = data_outline_mosquito_wings.coords.to_numpy().reshape(-1,100,2)

    # EFD
    efa = EllipticFourierAnalysis(n_components=20)
    coef = efa.transform(X)

    # PCA
    pca = PCA(n_components=3)
    pcscores = pca.fit_transform(coef)



