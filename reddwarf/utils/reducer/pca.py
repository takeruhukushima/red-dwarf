from numpy.typing import ArrayLike
import numpy as np
from reddwarf.utils.matrix import VoteMatrix
from typing import Tuple

from reddwarf.utils.reducer.base import run_reducer

from reddwarf.utils.reducer.base import ReducerModel

def run_pca(
        vote_matrix: VoteMatrix,
        n_components: int = 2,
) -> Tuple[ np.ndarray, np.ndarray | None, ReducerModel ]:
    """
    Process a prepared vote matrix to be imputed and return projected participant data,
    as well as eigenvectors and eigenvalues.

    The vote matrix should not yet be imputed, as this will happen within the method.

    Args:
        vote_matrix (pd.DataFrame): A vote matrix of data. Non-imputed values are expected.
        n_components (int): Number n of principal components to decompose the `vote_matrix` into.

    Returns:
        projected_data (pd.DataFrame): A dataframe of projected xy coordinates for each `vote_matrix` row/participant.
        pca (PCA): PCA object, notably with the following attributes:
            - components_ (List[List[float]]): Eigenvectors of principal `n` components, one per column/statement/feature.
            - explained_variance_ (List[float]): Explained variance of each principal component.
            - mean_ (list[float]): Means/centers of each column/statements/features.
    """
    projected_participants, projected_statements, pca = run_reducer(
        vote_matrix=vote_matrix.values, n_components=n_components, reducer="pca"
    )

    return projected_participants, projected_statements, pca

def calculate_extremity(projections: ArrayLike):
    # Compute extremity as vector magnitude on rows.
    # vector magnitude = Euclidean norm = hypotenuse of xy
    return np.linalg.norm(projections, axis=0)