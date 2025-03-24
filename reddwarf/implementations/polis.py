

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
    results = run_pca(vote_matrix=vote_matrix)
    return results