from reddwarf import utils
from tests.fixtures import small_convo_math_data
import numpy as np
import pytest

from reddwarf.data_loader import Loader
from reddwarf.polis import PolisClient

from typing import TypedDict, TypeAlias

IncrementingId: TypeAlias = int
BaseClusterId: TypeAlias = IncrementingId
GroupId: TypeAlias = IncrementingId
ParticipantId: TypeAlias = IncrementingId

class PolisGroupCluster(TypedDict):
    id: GroupId
    members: list[BaseClusterId]
    center: list[float]

class PolisGroupClusterExpanded(TypedDict):
    id: GroupId
    members: list[ParticipantId]
    center: list[float]

class PolisBaseClusters(TypedDict):
    # Each outer list will be the same length, and will be 100 items or less.
    id: list[BaseClusterId]
    members: list[list[ParticipantId]]
    x: list[float]
    y: list[float]
    count: list[int]

def expand_group_clusters_with_participants(
    group_clusters: list[PolisGroupCluster],
    base_clusters: PolisBaseClusters,
) -> list[PolisGroupClusterExpanded]:
    """
    Expand group clusters to include direct participant IDs instead of base cluster IDs.

    Args:
        group_clusters (list[PolisGroupCluster]): Group cluster data from Polis math data, with base cluster ID membership.
        base_clusters (PolisBaseClusters): Base cluster data from Polis math data, with participant ID membership.

    Returns:
        group_clusters_with_participant_ids (list[PolisGroupClusterExpanded]): A list of group clusters with participant IDs.
    """
    # Extract base clusters and their members
    base_cluster_ids = base_clusters["id"]
    base_cluster_participant_ids = base_clusters["members"]

    # Create a mapping from base cluster ID to member participant IDs
    base_to_participant_mapping = dict(zip(base_cluster_ids, base_cluster_participant_ids))

    # Create the new group-clusters-participant list
    group_clusters_with_participant_ids = []

    for group in group_clusters:
        # Get all participant IDs for this group by expanding and flattening the base cluster members.
        group_participant_ids = [
            participant_id
            for base_cluster_id in group["members"]
            for participant_id in base_to_participant_mapping[base_cluster_id]
        ]

        # Create the new group object with expanded participant IDs
        expanded_group = {
            **group,
            "id": group["id"],
            "members": group_participant_ids,
        }

        group_clusters_with_participant_ids.append(expanded_group)

    return group_clusters_with_participant_ids

def generate_cluster_labels(
    group_clusters_with_pids: list[PolisGroupClusterExpanded],
) -> np.ndarray[GroupId]:
    """
    Transform group_clusters_with_pid into an ordered list of cluster IDs
    sorted by participant ID, suitable for cluster labels.

    Args:
        group_clusters_with_pids (list[PolisGroupClusterExpanded]): List of group clusters with expanded participant IDs

    Returns:
        np.ndarray[GroupId]: A list of cluster IDs ordered by participant ID
    """
    # Create a list of (participant_id, cluster_id) pairs
    participant_cluster_pairs = []

    for cluster in group_clusters_with_pids:
        cluster_id = cluster["id"]
        for participant_id in cluster["members"]:
            participant_cluster_pairs.append((participant_id, cluster_id))

    # Sort the list by participant_id
    sorted_pairs = sorted(participant_cluster_pairs, key=lambda x: x[0])

    # Extract just the cluster IDs in order
    cluster_ids_in_order = [pair[1] for pair in sorted_pairs]

    return np.asarray(cluster_ids_in_order)

def get_all_participant_ids(
    group_clusters_with_pids: list[PolisGroupClusterExpanded],
) -> list[ParticipantId]:
    """
    Extract all unique participant IDs from group_clusters_with_pids.

    Args:
        group_clusters_with_pids (list): List of group clusters with expanded participant IDs

    Returns:
        list: A sorted list of all unique participant IDs
    """
    # Gather all participant IDs from all clusters
    all_participant_ids = []

    for cluster in group_clusters_with_pids:
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
    group_clusters_with_pids = expand_group_clusters_with_participants(
        group_clusters=math_data["group-clusters"],
        base_clusters=math_data["base-clusters"],
    )
    group_count = len(group_clusters_with_pids)

    # Get list of all active participant ids, since Polis has some edge-cases
    # that keep specific participants, and we need to keep them from being filtered out.
    all_participant_ids = get_all_participant_ids(group_clusters_with_pids)
    client.keep_participant_ids = all_participant_ids
    vote_matrix = client.get_matrix(is_filtered=True)

    # Get sorted list of cluster labels from
    cluster_labels = generate_cluster_labels(group_clusters_with_pids)

    # Generate stats all groups and all statements.
    group_stats = utils.calculate_comment_statistics_by_group(
        vote_matrix=vote_matrix,
        cluster_labels=np.array(cluster_labels),
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
                k: group_stats[group_id].loc[st["tid"], v]
                for k,v in key_map.items()
            }

            assert actual["prob"] == pytest.approx(expected_prob)
            assert actual["prob_test"] == pytest.approx(expected_prob_test)
            assert actual["repness"] == pytest.approx(expected_repr)
            assert actual["repness_test"] == pytest.approx(expected_repr_test)
