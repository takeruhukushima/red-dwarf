from reddwarf import utils
from tests.fixtures import small_convo_math_data
import numpy as np
import pytest

from reddwarf.data_loader import Loader
from reddwarf.polis import PolisClient

def test_calculate_representativeness_real_data(small_convo_math_data):
    math_data, path, _ = small_convo_math_data
    client = PolisClient(is_strict_moderation=False)
    client.load_data(filepaths=[
        f'{path}/votes.json',
        f'{path}/comments.json',
    ])

    # Get group-clusters, but with members as participant ids rather than base cluster ids
    group_clusters_with_pids = utils.expand_group_clusters_with_participants(
        group_clusters=math_data["group-clusters"],
        base_clusters=math_data["base-clusters"],
    )
    group_count = len(group_clusters_with_pids)

    # Get list of all active participant ids, since Polis has some edge-cases
    # that keep specific participants, and we need to keep them from being filtered out.
    all_participant_ids = utils.get_all_participant_ids(group_clusters_with_pids)
    client.keep_participant_ids = all_participant_ids
    vote_matrix = client.get_matrix(is_filtered=True)

    # Get sorted list of cluster labels from
    cluster_labels = utils.generate_cluster_labels(group_clusters_with_pids)

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
