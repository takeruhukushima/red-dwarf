from numpy.typing import ArrayLike
import pandas as pd
import numpy as np
from reddwarf.utils.matrix import VoteMatrix, generate_virtual_vote_matrix
from reddwarf.sklearn.transformers import SparsityAwareCapturer, SparsityAwareScaler
from reddwarf.sklearn.pipeline import PatchedPipeline
from typing import Literal, Tuple, Union, TYPE_CHECKING, TypeAlias

from sklearn.impute import SimpleImputer

if TYPE_CHECKING:
    from pacmap import PaCMAP, LocalMAP
    from sklearn.decomposition import PCA

ReducerType: TypeAlias = Literal["pca", "pacmap", "localmap"]
ReducerModel: TypeAlias = Union["PCA", "PaCMAP", "LocalMAP"]


def get_reducer(
    reducer: ReducerType, n_components=2, random_state=None, **kwargs
) -> ReducerModel:
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
                n_neighbors=N_NEIGHBORS,  # type:ignore
                **kwargs,
            )
        case "localmap":
            from pacmap import LocalMAP

            return LocalMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=N_NEIGHBORS,  # type:ignore
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
) -> Tuple[ pd.DataFrame, pd.DataFrame | None, ReducerModel ]:
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
        vote_matrix=vote_matrix, n_components=n_components, reducer="pca"
    )

    return projected_participants, projected_statements, pca


def run_reducer(
    vote_matrix: VoteMatrix,
    n_components: int = 2,
    reducer: ReducerType = "pca",
) -> Tuple[pd.DataFrame, pd.DataFrame | None, ReducerModel]:
    """
    Process a prepared vote matrix to be imputed and return projected participant data,
    as well as eigenvectors and eigenvalues.

    The vote matrix should not yet be imputed, as this will happen within the method.

    Args:
        vote_matrix (pd.DataFrame): A vote matrix of data. Non-imputed values are expected.
        n_components (int): Number n of principal components to decompose the `vote_matrix` into.
        reducer (Literal["pca", "pacmap", "localmap"]): Dimensionality reduction method to use.

    Returns:
        projected_data (pd.DataFrame): A dataframe of projected xy coordinates for each `vote_matrix` row/participant.
        reducer_model (ReducerModel): Reducer estimator, notably with the following attributes:
            - components_ (List[List[float]]): Eigenvectors of principal `n` components, one per column/statement/feature.
            - explained_variance_ (List[float]): Explained variance of each principal component.
            - mean_ (list[float]): Means/centers of each column/statements/features.
    """
    match reducer:
        case "pca":
            pipeline = PatchedPipeline(
                [
                    ("capture", SparsityAwareCapturer()),
                    ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
                    ("reduce", get_reducer(reducer, n_components=n_components)),
                    ("scale", SparsityAwareScaler(capture_step="capture")),
                ]
            )
        case "pacmap" | "localmap":
            pipeline = PatchedPipeline(
                [
                    ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
                    ("reduce", get_reducer(reducer, n_components=n_components)),
                ]
            )

    pipeline.fit(vote_matrix.values)

    DEFAULT_DIMENSION_LABELS = ["x", "y", "z"]
    dimension_labels = DEFAULT_DIMENSION_LABELS[:n_components]

    # Generate projections of participants.
    X_participants = pipeline.transform(vote_matrix.values)

    projected_participants = pd.DataFrame(
        X_participants,
        index=pd.Index(vote_matrix.index, name="participant_id"),
        columns=np.asarray(dimension_labels),
    )

    if reducer == "pca":
        # Generate projections of statements via virtual vote matrix.
        # This projects unit vectors for each feature/statement into PCA space to
        # understand their placement.
        n_statements = len(vote_matrix.columns)
        virtual_vote_matrix = generate_virtual_vote_matrix(n_statements)
        X_statements = pipeline.transform(virtual_vote_matrix)

        projected_statements = pd.DataFrame(
            X_statements,
            index=pd.Index(vote_matrix.columns, name="statement_id"),
            columns=np.asarray(dimension_labels),
        )
    else:
        projected_statements = None

    reducer_model: ReducerModel = pipeline.named_steps["reduce"]

    return projected_participants, projected_statements, reducer_model


def calculate_extremity(projections: ArrayLike):
    # Compute extremity as vector magnitude on rows.
    # vector magnitude = Euclidean norm = hypotenuse of xy
    return np.linalg.norm(projections, axis=0)