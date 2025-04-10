from numpy.typing import ArrayLike
import pandas as pd
import numpy as np
from reddwarf.utils.matrix import VoteMatrix
from reddwarf.sklearn.transformers import SparsityAwareScaler
from typing import Tuple

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def run_pca(
        vote_matrix: VoteMatrix,
        n_components: int = 2,
) -> Tuple[ pd.DataFrame, pd.DataFrame, PCA ]:
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
    pipeline = Pipeline([
        ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("pca", PCA(n_components=n_components)),
        ("scale", SparsityAwareScaler()),
    ])

    pipeline.fit(vote_matrix.values)
    pca = pipeline.named_steps["pca"]

    def generate_projections(sparse_vote_matrix, fitted_pipeline):
        fitted_pipeline.named_steps["scale"].X_sparse = sparse_vote_matrix
        X_projected = fitted_pipeline.transform(sparse_vote_matrix)

        return X_projected

    # Create a matrix of virtual participants that each vote once on a single statement.
    def generate_virtual_vote_matrix(n_statements: int):
        # Build an basic identity matrix
        virtual_vote_matrix = np.eye(n_statements)

        # Replace 1s with +1 and 0s with NaN
        # TODO: Why does Polis use -1 (disagree) here? is it the same? BUG?
        AGREE_VAL = 1
        MISSING_VAL = np.nan
        virtual_vote_matrix = np.where(virtual_vote_matrix == 1, AGREE_VAL, MISSING_VAL)

        return virtual_vote_matrix

    X_participants = generate_projections(sparse_vote_matrix=vote_matrix.values, fitted_pipeline=pipeline)

    n_statements = len(vote_matrix.columns)
    virtual_vote_matrix = generate_virtual_vote_matrix(n_statements)
    X_statements = generate_projections(sparse_vote_matrix=virtual_vote_matrix, fitted_pipeline=pipeline)

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

    return projected_participants, projected_statements, pca

# TODO: Clean up variables and docs.
def sparsity_aware_project_ptpt(participant_votes, statement_components, statement_means):
    """
    Projects a sparse vote vector into PCA space while adjusting for sparsity.

    Args:
        participant_votes (list): List of participant votes on each statement
        statement_components (list[list[float]]): Two lists of floats corresponding to the two principal components
        statement_means (list[float]): List of floats corresponding to the centers/means of each statement

    Returns:
        projected_coords (list[list[float]]): Two lists corresponding to projected xy coordinates.
    """
    statement_components = np.array(statement_components)  # Shape: (2, n_features)
    statement_means = np.array(statement_means)  # Shape: (n_features,)

    # TODO: This included zerod out (moderated) statements. Should it?
    n_statements = len(participant_votes)

    participant_votes = np.array(participant_votes)
    mask = ~np.isnan(participant_votes)  # Only consider non-null values

    # Extract relevant values
    x_vals = participant_votes[mask] - statement_means[mask]  # Centered values
    # TODO: Extend this to work in 3D
    pc1_vals, pc2_vals = statement_components[:, mask]  # Select only used components

    # Compute dot product projection
    p1 = np.dot(x_vals, pc1_vals)
    p2 = np.dot(x_vals, pc2_vals)

    # Non-null votes count
    n_votes = np.count_nonzero(mask)
    scale = np.sqrt(n_statements / max(n_votes, 1))

    projected_coord = scale * np.array([p1, p2])

    return projected_coord

# TODO: Clean up variables and docs.
def sparsity_aware_project_ptpts(vote_matrix, statement_components, statement_means):
    """
    Apply sparsity-aware projection to multiple vote vectors.
    """
    return np.array([
        sparsity_aware_project_ptpt(participant_votes, statement_components, statement_means)
        for participant_votes in vote_matrix]
    )

# TODO: Clean up variables and docs.
def pca_project_cmnts(statement_components, statement_means):
    """
    Projects unit vectors for each feature into PCA space to understand their placement.
    """

    # Create a matrix of virtual participants that each vote once on a single statement.

    # Build an basic identity matrix
    n_statements = len(statement_means)
    virtual_vote_matrix = np.eye(n_statements)

    # Replace 1s with -1 and 0s with NaN
    # TODO: Why does Polis use -1 (disagree) here? is it the same? BUG?
    AGREE_VAL = 1
    MISSING_VAL = np.nan
    virtual_vote_matrix = np.where(virtual_vote_matrix == 1, AGREE_VAL, MISSING_VAL)

    statement_projections = sparsity_aware_project_ptpts(
        virtual_vote_matrix,
        statement_components,
        statement_means,
    )

    return statement_projections

def calculate_extremity(projections: ArrayLike):
    # Compute extremity as vector magnitude on rows.
    # vector magnitude = Euclidean norm = hypotenuse of xy
    return np.linalg.norm(projections, axis=0)

# TODO: Clean up variables and docs.
def with_proj_and_extremity(pca):
    """
    Compute projection and extremity, then merge into PCA results.
    """
    statement_projections = pca_project_cmnts(
        statement_components=pca["comps"],
        statement_means=pca["center"],
    )

    # Flip the axes to get all x together and y together.
    # 2 sets of 40. shape (2, 40)
    statement_projections = statement_projections.transpose()

    statement_extremities = calculate_extremity(statement_projections)

    pca["comment-projection"] = statement_projections.tolist()
    pca["comment-extremity"] = statement_extremities.tolist()

    return pca
