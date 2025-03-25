

from reddwarf.utils.matrix import generate_raw_matrix, simple_filter_matrix
from reddwarf.utils.pca import run_pca


def run_clustering(
    votes: list[dict],
    mod_out: list[int],
):
    vote_matrix = generate_raw_matrix(votes=votes)
    vote_matrix = simple_filter_matrix(
        vote_matrix=vote_matrix,
        statement_ids_mod_out=mod_out,
    )
    projected_data, comps, eigenvalues, center = run_pca(vote_matrix=vote_matrix)

    # To match Polis output, we need to reverse signs for centers and projections
    # TODO: Investigate why this is. Perhaps related to signs being flipped on agree/disagree back in the day.
    return -projected_data, comps, eigenvalues, -center