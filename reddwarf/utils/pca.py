from __future__ import annotations
from numpy.typing import ArrayLike
import pandas as pd
import numpy as np
from reddwarf.utils.matrix import VoteMatrix, generate_virtual_vote_matrix
from reddwarf.sklearn.transformers import SparsityAwareCapturer, SparsityAwareScaler
from reddwarf.sklearn.pipeline import PatchedPipeline
from typing import Tuple, TYPE_CHECKING

from sklearn.impute import SimpleImputer

if TYPE_CHECKING:
    from pacmap import PaCMAP, LocalMAP
    from sklearn.decomposition import PCA


def get_reducer(
    reducer: str, n_components=2, random_state=None, **kwargs
) -> PCA | PaCMAP | LocalMAP:
    # Setting n_neighbors to None defaults to 10 below 10,000 samples, and
    # slowly increases it according to a formula beyond that.
    # See: https://github.com/YingfanWang/PaCMAP?tab=readme-ov-file#parameters
    N_NEIGHBORS = None
    match reducer:
        case "pacmap":
            from pacmap import PaCMAP

            return PaCMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=N_NEIGHBORS,
                **kwargs,
            )
        case "localmap":
            from pacmap import LocalMAP

            return LocalMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=N_NEIGHBORS,
                **kwargs,
            )
        case "pca" | _:
            from sklearn.decomposition import PCA

            return PCA(
                n_components=n_components,
                random_state=random_state,
                **kwargs,
            )


def run_pca(
    vote_matrix: VoteMatrix,
    n_components: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Process a prepared vote matrix to be imputed and return projected participant data,
    as well as eigenvectors and eigenvalues.

    The vote matrix should not yet be imputed, as this will happen within the method.

    Args:
        vote_matrix (pd.DataFrame): A vote matrix of data. Non-imputed values are expected.
        n_components (int): Number n of principal components to decompose the `vote_matrix` into.

    Returns:
        projected_data (pd.DataFrame): A dataframe of projected xy coordinates for each `vote_matrix` row/participant.
        reducer (PCA): PCA object, notably with the following attributes:
            - components_ (List[List[float]]): Eigenvectors of principal `n` components, one per column/statement/feature.
            - explained_variance_ (List[float]): Explained variance of each principal component.
            - mean_ (list[float]): Means/centers of each column/statements/features.
    """
    pipeline = PatchedPipeline(
        [
            ("capture", SparsityAwareCapturer()),
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("reduce", get_reducer("pca", n_components=n_components)),
            ("scale", SparsityAwareScaler(capture_step="capture")),
        ]
    )

    pipeline.fit(vote_matrix.values)

    # Generate projections of participants.
    X_participants = pipeline.transform(vote_matrix.values)

    # Generate projections of statements via virtual vote matrix.
    # This projects unit vectors for each feature/statement into PCA space to
    # understand their placement.
    n_statements = len(vote_matrix.columns)
    virtual_vote_matrix = generate_virtual_vote_matrix(n_statements)
    X_statements = pipeline.transform(virtual_vote_matrix)

    DEFAULT_DIMENSION_LABELS = ["x", "y", "z"]
    dimension_labels = DEFAULT_DIMENSION_LABELS[:n_components]

    projected_participants = pd.DataFrame(
        X_participants,
        index=pd.Index(vote_matrix.index, name="participant_id"),
        columns=np.asarray(dimension_labels),
    )

    projected_statements = pd.DataFrame(
        X_statements,
        index=pd.Index(vote_matrix.columns, name="statement_id"),
        columns=np.asarray(dimension_labels),
    )

    reducer = pipeline.named_steps["reduce"]

    return projected_participants, projected_statements, reducer


def calculate_extremity(projections: ArrayLike):
    # Compute extremity as vector magnitude on rows.
    # vector magnitude = Euclidean norm = hypotenuse of xy
    return np.linalg.norm(projections, axis=0)
