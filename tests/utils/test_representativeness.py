from reddwarf import utils
from tests.fixtures import small_convo_math_data
import numpy as np
import pytest

from reddwarf.data_loader import Loader
from reddwarf.polis import PolisClient

def expand_group_clusters_to_participants(math_data):
    """
    Transform group-clusters to include direct participant IDs instead of base-cluster IDs.

    Args:
        math_data (dict): The math_data object containing base-clusters and group-clusters

    Returns:
        list: A list of group-clusters with expanded participant IDs
    """
    # Extract base clusters and their members
    base_clusters_ids = math_data["base-clusters"]["id"]
    base_clusters_members = math_data["base-clusters"]["members"]

    # Create a mapping from base cluster ID to its participant members
    base_cluster_to_participants = {
        base_clusters_ids[i]: base_clusters_members[i]
        for i in range(len(base_clusters_ids))
    }

    # Create the new group-clusters-participant list
    group_clusters_participant = []

    for group in math_data["group-clusters"]:
        # Get all participant IDs for this group by expanding the base cluster members
        all_participants = []
        for base_cluster_id in group["members"]:
            all_participants.extend(base_cluster_to_participants[base_cluster_id])

        # Create the new group object with expanded participant IDs
        new_group = {
            "id": group["id"],
            "members": all_participants
        }

        group_clusters_participant.append(new_group)

    return group_clusters_participant

def get_clusters_by_participant_id(group_clusters_participant):
    """
    Transform group_clusters_participant into an ordered list of cluster IDs
    sorted by participant ID.

    Args:
        group_clusters_participant (list): List of group clusters with expanded participant IDs

    Returns:
        list: A list of cluster IDs ordered by participant ID
    """
    # Create a list of (participant_id, cluster_id) pairs
    participant_cluster_pairs = []

    for cluster in group_clusters_participant:
        cluster_id = cluster["id"]
        for participant_id in cluster["members"]:
            participant_cluster_pairs.append((participant_id, cluster_id))

    # Sort the list by participant_id
    sorted_pairs = sorted(participant_cluster_pairs, key=lambda x: x[0])

    # Extract just the cluster IDs in order
    cluster_ids_in_order = [pair[1] for pair in sorted_pairs]

    return cluster_ids_in_order

def get_all_participant_ids(group_clusters_participant):
    """
    Extract all unique participant IDs from group_clusters_participant.

    Args:
        group_clusters_participant (list): List of group clusters with expanded participant IDs

    Returns:
        list: A sorted list of all unique participant IDs
    """
    # Gather all participant IDs from all clusters
    all_participant_ids = []

    for cluster in group_clusters_participant:
        all_participant_ids.extend(cluster["members"])

    # Remove duplicates and sort
    unique_participant_ids = sorted(set(all_participant_ids))

    return unique_participant_ids

def test_calculate_representativeness_real_data(small_convo_math_data):
    math_data, path, _ = small_convo_math_data
    client = PolisClient(is_strict_moderation=False)
    client.load_data(filepaths=[
        f'{path}/votes.json',
        f'{path}/comments.json',
    ])

    # Get group-clusters, but with members as participant ids rather than base cluster ids
    group_clusters_participant = expand_group_clusters_to_participants(math_data)
    group_count = len(group_clusters_participant)

    # Get list of all active participant ids, since Polis has some edge-cases
    # that keep specific participants, and we need to keep them from being filtered out.
    all_participant_ids = get_all_participant_ids(group_clusters_participant)
    client.keep_participant_ids = all_participant_ids
    vote_matrix = client.get_matrix(is_filtered=True)

    # Get sorted list of cluster labels from
    cluster_labels = get_clusters_by_participant_id(group_clusters_participant)

    # Generate representativeness data for all groups and all statements.
    group_repness = {}
    for gid in range(group_count):
        group_repness[gid] = utils.calculate_representativeness(
            vote_matrix=vote_matrix,
            cluster_labels=np.array(cluster_labels),
            group_id=gid,
        )

    # Cycle through all the expected data calculated by Polis platform
    repness = math_data['repness']
    for group_id, statements in math_data['repness'].items():
        group_id = int(group_id)
        for st in statements:
            expected_repr = st["repness"]
            expected_repr_test = st["repness-test"]
            expected_prob = st["p-success"]
            expected_prob_test = st["p-test"]

            # Fetch matching calculated values for comparison.
            keys = ["prob", "prob_test", "repness", "repness_test"]
            if st["repful-for"] == "agree":
                key_map = dict(zip(keys, ["pa", "pat", "ra", "rat"]))
            elif st["repful-for"] == "disagree":
                key_map = dict(zip(keys, ["pd", "pdt", "rd", "rdt"]))

            actual = {
                k: group_repness[group_id].loc[st["tid"], v]
                for k,v in key_map.items()
            }

            assert actual["prob"] == pytest.approx(expected_prob)
            assert actual["prob_test"] == pytest.approx(expected_prob_test)
            assert actual["repness"] == pytest.approx(expected_repr)
            assert actual["repness_test"] == pytest.approx(expected_repr_test)
