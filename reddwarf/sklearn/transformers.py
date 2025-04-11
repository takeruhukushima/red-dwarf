import numpy as np
from typing import Optional
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

class SparsityAwareScaler(BaseEstimator, TransformerMixin):
    """
    Scale projected points (participant/statements) based on sparsity of vote
    matrix, to account for any small number of votes by a participant and
    prevent those participants from bunching up in the center.

    Attributes:
        X_sparse (np.ndarray | None): A sparse array with shape (n_features,)
    """
    def __init__(self, X_sparse: Optional[NDArray[np.float64]] = None):
        self.X_sparse = X_sparse

    # See: https://scikit-learn.org/stable/modules/generated/sklearn.utils.Tags.html#sklearn.utils.Tags
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # Suppresses warning caused by fit() not being required before usage in transform().
        tags.requires_fit = False
        return tags

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scale_factors = self._calculate_scaling_factors()
        return X * scale_factors[:, np.newaxis]

    def inverse_transform(self, X):
        scale_factors = self._calculate_scaling_factors()
        return X / scale_factors[:, np.newaxis]

    def _calculate_scaling_factors(self):
        if self.X_sparse is None:
            raise AttributeError(
                "Missing `X_sparse`. Pass `X_sparse` when initializing SparsityAwareScaler."
            )

        # The total number of statements in matrix.
        _, n_cols = self.X_sparse.shape
        n_total_statements = n_features = n_cols

        # List of votes per participant row.
        vote_mask = ~np.isnan(self.X_sparse)
        n_participant_votes = n_non_missing = np.count_nonzero(vote_mask, axis=1)
        # Ref: https://hyp.is/x6nhItMMEe-v1KtYFgpOiA/gwern.net/doc/sociology/2021-small.pdf
        # Ref: https://github.com/compdemocracy/polis/blob/15aa65c9ca9e37ecf57e2786d7d81a4bd4ad37ef/math/src/polismath/math/pca.clj#L155-L1

        # scaling_factors = np.sqrt(n_features / np.maximum(1, n_non_missing))
        scaling_factors = np.sqrt(n_total_statements / np.maximum(1, n_participant_votes))

        return scaling_factors