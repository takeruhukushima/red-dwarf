def pad_to_size(lst, size):
    return list(lst) + [[0., 0.]]*(size - len(lst))

def transform_base_clusters_to_participant_coords(base_clusters):
    """
    Transform base clusters data into a list of dictionaries with participant_id and xy coordinates.

    Args:
        base_clusters (dict): A dictionary containing base clusters data with 'id', 'members', 'x', and 'y' keys.

    Returns:
        list: A list of dictionaries, each containing a participant_id and their xy coordinates.
    """
    # For now, ensure failure if a base-cluster has more than one member, as the test assumes that.
    def get_only_member_or_raise(members):
        if len(members) != 1:
            raise Exception("A base-cluster has more than one member when it cannot")

        return members[0]

    return [
        {
            "participant_id": get_only_member_or_raise(members),
            "xy": [x, y]
        }
        for members, x, y in zip(
            base_clusters['members'],
            base_clusters['x'],
            base_clusters['y']
        )
    ]

# Not used right now. Maybe later.
def groupsort_pids_by_cluster(df):
    """
    Helper function to gather statement IDS in clusters and sort them for easy
    comparison.

    This make comparison easy, even when kmeans gives different numeric labels.

    Args:
        df (pd.DataFrame): A dataframe with projected participants, columns "x",
        "y", "cluster_id"
    Returns:
        (list[list[int]]): A list of lists, each containing statement IDs in a
        cluster.
    """
    # Group by cluster_id and collect indices
    grouped = df.groupby('cluster_id').apply(lambda x: list(x.index))

    # Sort the groups by their length (number of members) in descending order
    sorted_groups = sorted(grouped, key=len, reverse=True)

    # Convert each inner list to integers
    return [list(map(int, group)) for group in sorted_groups]