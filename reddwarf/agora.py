from reddwarf.types.agora import Conversation, ClusteringResult, ClusteringOptions
from reddwarf import utils

DEFAULT_MIN_USER_VOTE_THRESHOLD = 7
DEFAULT_MAX_CLUSTERS = 5
DEFAULT_KMEANS_RANDOM_STATE = 123456789

def run_clustering(
    conversation: Conversation,
    options: ClusteringOptions = {},
) -> ClusteringResult:
    vote_matrix = utils.generate_raw_matrix(votes=conversation["votes"])
    # Any statements with votes are included.
    all_statement_ids = vote_matrix.columns
    vote_matrix = utils.filter_matrix(
        vote_matrix=vote_matrix,
        min_user_vote_threshold=options.get("min_user_vote_threshold", DEFAULT_MIN_USER_VOTE_THRESHOLD),
        active_statement_ids=all_statement_ids,
    )
    projected_data, _, _ = utils.run_pca(vote_matrix=vote_matrix)
    projected_data = utils.scale_projected_data(
        projected_data=projected_data,
        vote_matrix=vote_matrix,
    )

    _, _, cluster_labels = utils.find_optimal_k(
        projected_data=projected_data,
        max_group_count=options.get("max_clusters", DEFAULT_MAX_CLUSTERS),
        # Ensure reproducible kmeans calculation between runs.
        random_state=DEFAULT_KMEANS_RANDOM_STATE,
    )

    # Add cluster label column to dataframe.
    projected_data = projected_data.assign(cluster_label=cluster_labels)
    # Convert participant_id index into regular column, for ease of transformation.
    projected_data = projected_data.reset_index()

    result: ClusteringResult = {
        "clusters": [
            {
                "label": cluster_label,
                "participants": [
                    {
                        "id": row.participant_id,
                        "x": row.x,
                        "y": row.y,
                    }
                    for row in group.itertuples(index=False)
                ]
            }
            for cluster_label, group in projected_data.groupby("cluster_label")
        ]
    }

    return result
