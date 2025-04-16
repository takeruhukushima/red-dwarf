from typing import Optional
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.decomposition import PCA
from reddwarf.sklearn.cluster import PolisKMeans
from reddwarf.utils.matrix import generate_raw_matrix, simple_filter_matrix, get_clusterable_participant_ids
from reddwarf.utils.pca import run_pca
from reddwarf.utils.clustering import find_optimal_k
from dataclasses import dataclass

from reddwarf.utils.stats import calculate_comment_statistics_dataframes

@dataclass
class PolisClusteringResult:
    raw_vote_matrix: DataFrame
    filtered_vote_matrix: DataFrame
    pca: PCA
    projected_participants: DataFrame
    projected_statements: DataFrame
    kmeans: PolisKMeans | None
    group_aware_consensus: DataFrame

def run_clustering(
    votes: list[dict],
    mod_out_statement_ids: list[int] = [],
    min_user_vote_threshold: int = 7,
    keep_participant_ids: list[int] = [],
    init_centers: Optional[list[list[float]]] = None,
    max_group_count: int = 5,
    force_group_count: Optional[int] = None,
    random_state: Optional[int] = None,
) -> PolisClusteringResult:
    """
    An essentially feature-complete implementation of the Polis clustering algorithm.

    Still missing:
        - base-cluster calculations (so can't match output of conversations larger than 100 participants),
        - k-smoothing, which holds back k-value (group count) until re-calculated 3 consecutive times,
        - some advanced participant filtering that involves past state (you can use keep_participant_ids to mimic manually).

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
        PolisClusteringResult: A dataclass containing clustering information with fields:
            - raw_vote_matrix (DataFrame): Raw sparse vote matrix before any processing.
            - filtered_vote_matrix (DataFrame): Raw sparse vote matrix with moderated statements zero'd out.
            - pca (PCA): PCA model fitted to vote matrix, including `mean_`, `expected_variance_` (eigenvalues) and `components_` (eigenvectors).
            - projected_participants (DataFrame): Dataframe of projected participants, with columns "x", "y", "cluster_id"
            - projected_statements (DataFrame): Dataframe of projected statements, with columns "x", "y".
            - kmeans (PolisKMeans): Scikit-Learn KMeans object for selected group count, including `labels_` and `cluster_centers_`. See `PolisKMeans`.
    """
    raw_vote_matrix = generate_raw_matrix(votes=votes)

    filtered_vote_matrix = simple_filter_matrix(
        vote_matrix=raw_vote_matrix,
        mod_out_statement_ids=mod_out_statement_ids,
    )

    projected_participants, projected_statements, pca = run_pca(vote_matrix=filtered_vote_matrix)

    participant_ids_clusterable = get_clusterable_participant_ids(raw_vote_matrix, vote_threshold=min_user_vote_threshold)
    if keep_participant_ids:
        participant_ids_clusterable = list(set(participant_ids_clusterable + keep_participant_ids))

    if force_group_count:
        k_bounds = [force_group_count, force_group_count]
    else:
        k_bounds = [2, max_group_count]

    projected_participants_clusterable = projected_participants.loc[participant_ids_clusterable, :]
    _, _, kmeans = find_optimal_k(
        projected_data=projected_participants_clusterable,
        k_bounds=k_bounds,
        # Force polis strategy of initiating cluster centers. See: PolisKMeans.
        init="polis",
        init_centers=init_centers,
        random_state=random_state,
    )
    projected_participants_clusterable = projected_participants_clusterable.assign(
        cluster_id=kmeans.labels_ if kmeans else None,
    )

    group_stats_df, gac_df = calculate_comment_statistics_dataframes(
        vote_matrix=raw_vote_matrix.loc[participant_ids_clusterable, :],
        cluster_labels=kmeans.labels_,
    )

    return PolisClusteringResult(
        raw_vote_matrix=raw_vote_matrix,
        filtered_vote_matrix=filtered_vote_matrix,
        pca=pca,
        projected_participants=projected_participants_clusterable,
        projected_statements=projected_statements,
        kmeans=kmeans,
        group_aware_consensus=gac_df,
    )