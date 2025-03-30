from typing import Optional
from numpy.typing import NDArray
from pandas._typing import DataFrame
from reddwarf.utils.matrix import generate_raw_matrix, simple_filter_matrix, get_participant_ids
from reddwarf.utils.pca import run_pca
from reddwarf.utils.clustering import find_optimal_k, run_kmeans


def run_clustering(
    votes: list[dict],
    mod_out_statement_ids: list[int] = [],
    min_user_vote_threshold: int = 7,
    keep_participant_ids: list[int] = [],
    init_centers: Optional[list[list[float]]] = None,
    max_group_count: int = 5,
    force_group_count: Optional[int] = None,
    random_state: Optional[int] = None,
) -> tuple[DataFrame, NDArray, NDArray, NDArray, NDArray | None]:
    """
    An essentially feature-complete implementation of the Polis clustering algorithm.

    Args:
        votes (list[dict]): Raw list of vote dicts, with keys for "participant_id", "statement_id", "vote" and "modified"
        mod_out_statement_ids (list[int]): List of statement IDs to moderate/zero out
        min_user_vote_threshold (int): Minimum number of votes a participant must make to be included in clustering
        keep_participant_ids (list[int]): List of participant IDs to keep in clustering algorithm, regardless of normal filters.
        max_group_count (): Max number of group (k-values) to test using k-means and silhouette scores
        init_centers (list[list[float]]): Initial guesses of [x,y] coordinates for k-means (Length of list must match max_group_count)
        force_group_count (int): Instead of using silhouette scores, force a specific number of groups (k value)
        random_state (int): If set, will force determinism during k-means clustering

    Returns:
        projected_data (DataFrame): Dataframe of projected participants, with columns "x", "y", "cluster_id"
        comps (list[list[float]]): List of principal components for each statement
        eigenvalues (list[float]): List of eigenvalues for each principal component
        center (list[float]): List of centers/means for each statement
        cluster_centers (list[list[float]]): List of center xy coordinates for each cluster
    """
    vote_matrix = generate_raw_matrix(votes=votes)
    participant_ids_in = get_participant_ids(vote_matrix, vote_threshold=min_user_vote_threshold)
    if keep_participant_ids:
        participant_ids_in = list(set(participant_ids_in + keep_participant_ids))

    vote_matrix = simple_filter_matrix(
        vote_matrix=vote_matrix,
        mod_out_statement_ids=mod_out_statement_ids,
    )
    projected_data, comps, eigenvalues, center = run_pca(vote_matrix=vote_matrix)

    projected_data = projected_data.loc[participant_ids_in, :]

    # To match Polis output, we need to reverse signs for centers and projections
    # TODO: Investigate why this is. Perhaps related to signs being flipped on agree/disagree back in the day.
    projected_data, center = -projected_data, -center

    if force_group_count:
        cluster_labels, cluster_centers = run_kmeans(
            dataframe=projected_data,
            n_clusters=force_group_count,
            init_centers=init_centers,
            random_state=random_state,
        )
    else:
        _, _, cluster_labels, cluster_centers = find_optimal_k(
            projected_data=projected_data,
            max_group_count=max_group_count,
            init_centers=init_centers,
            random_state=random_state,
        )
    projected_data = projected_data.assign(cluster_id=cluster_labels)

    return projected_data, comps, eigenvalues, center, cluster_centers