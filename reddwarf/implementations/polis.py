

from reddwarf.utils.matrix import generate_raw_matrix, simple_filter_matrix, get_participant_ids
from reddwarf.utils.pca import run_pca
from reddwarf.utils.clustering import find_optimal_k


def run_clustering(
    votes: list[dict],
    mod_out_statement_ids: list[int],
    min_user_vote_threshold = 7,
    keep_participant_ids: list[int] = [],
    init_centers = None,
    max_group_count = 5,
):
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

    _, _, cluster_labels = find_optimal_k(
        projected_data=projected_data,
        max_group_count=max_group_count,
        init_centers=init_centers,
    )
    projected_data = projected_data.assign(cluster_id=cluster_labels)

    return projected_data, comps, eigenvalues, center