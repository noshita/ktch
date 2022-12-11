class PCContribDisplay:
    """"""

    def __init__(self, pca, *, display_labels=None) -> None:
        self.pca = pca
        self.display_labels = display_labels

    def plot(
        self, *, n_PCs=(1, 2, 3), sd_values=(-2, -1, 0, 1, 2), , ax=None, im_kw=None
    ):

        return self

    @classmethod
    def from_estimators(cls, estimator, X, y, *, labels=None):
        return cls.from_predictions()

    @classmethod
    def from_predictions(cls, y_true, y_pred, *, labels=None):
        disp = cls(pca=pca)
        return disp.plot()
