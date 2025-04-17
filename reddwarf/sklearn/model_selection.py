from sklearn.model_selection import GridSearchCV
import numpy as np

class GridSearchNonCV(GridSearchCV):
    """
    `sklearn.model_selection.GridSearchCV`, but modified to score against the
    full dataset (ie. not cross-validated).

    Normally, `GridSearchCV` splits up the `X` data and scores each "fold" of data.
    This is identical, but we automatically use the full dataset in each fold.
    """
    def __init__(self, estimator, param_grid, scoring=None, refit=True, **kwargs):
        # Default CV is a single fold: train = test = full dataset
        self._default_cv = None  # we'll set it in fit() when we have data size

        # User can override cv via kwargs
        self._user_provided_cv = 'cv' in kwargs
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            refit=refit,
            cv=None,  # will be overwritten in fit()
            **kwargs
        )

    def fit(self, X, y=None, **fit_params):
        if not self._user_provided_cv:
            # Create full-fold cross-validation only if user didn’t specify their own.
            # This scores the full dataset for each branch of grid.
            # See: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#:~:text=An%20iterable%20yielding%20(train%2C%20test)%20splits%20as%20arrays%20of%20indices.
            n_samples = len(X)
            full_idx = np.arange(n_samples)
            train_idx = full_idx
            test_idx = full_idx
            # Use full dataset for training/testing
            full_fold = [(train_idx, test_idx)]
            self.cv = full_fold
        return super().fit(X, y, **fit_params)