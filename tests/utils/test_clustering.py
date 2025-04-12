import pytest
from reddwarf.utils.clustering import run_kmeans, find_optimal_k
from tests.fixtures import polis_convo_data
from tests.helpers import transform_base_clusters_to_participant_coords
from reddwarf.utils.clustering import pad_centroid_list_to_length
import pandas as pd

@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_run_kmeans_real_data_reproducible(polis_convo_data):
    fixture = polis_convo_data

    expected_cluster_centers = [group["center"] for group in fixture.math_data["group-clusters"]]
    cluster_count = len(expected_cluster_centers)

    projected_participants = transform_base_clusters_to_participant_coords(fixture.math_data["base-clusters"])
    projected_participants_df = pd.DataFrame([
        {
            "participant_id": item["participant_id"],
            "x": item["xy"][0],
            "y": item["xy"][1],
        }
        for item in projected_participants
    ]).set_index("participant_id")

    _, calculated_cluster_centers = run_kmeans(
        dataframe=projected_participants_df,
        init_centers=expected_cluster_centers,
        n_clusters=cluster_count,
    )

    # Ensure same number of clusters
    assert len(calculated_cluster_centers) == len(expected_cluster_centers)

    # Ensure each is the same
    for i, _ in enumerate(calculated_cluster_centers):
        assert calculated_cluster_centers.tolist()[i] == pytest.approx(expected_cluster_centers[i])

# NOTE: "small-no-meta" fixture doesn't work because wants to find 4 clusters, whereas real data from polismath says 3.
# This is likely due to k-smoothing holding back the k value at 3 in polismath, and we're finding the real current one.
@pytest.mark.parametrize("polis_convo_data", ["small-with-meta"], indirect=True)
def test_find_optimal_k_real_data(polis_convo_data):
    fixture = polis_convo_data
    MAX_GROUP_COUNT = 5

    # Get centers from polismath.
    expected_centers = [group["center"] for group in fixture.math_data["group-clusters"]]

    projected_participants = transform_base_clusters_to_participant_coords(fixture.math_data["base-clusters"])
    projected_participants_df = pd.DataFrame([
        {
            "participant_id": item["participant_id"],
            "x": item["xy"][0],
            "y": item["xy"][1],
        }
        for item in projected_participants
    ]).set_index("participant_id")

    results = find_optimal_k(
        projected_data=projected_participants_df,
        max_group_count=MAX_GROUP_COUNT,
        # Pad center guesses to have enough values for testing up to max k groups.
        init_centers=pad_centroid_list_to_length(expected_centers, MAX_GROUP_COUNT)
    )
    optimal_k, silhouette_score, cluster_labels, cluster_centers = results # for documentation

    calculated_centers = cluster_centers.tolist()
    print(expected_centers)
    print(calculated_centers)

    assert len(expected_centers) == len(calculated_centers)
    for i, _ in enumerate(expected_centers):
        assert expected_centers[i] == pytest.approx(calculated_centers[i])

@pytest.mark.skip
def test_pad_centroid_list_to_length():
    raise NotImplementedError
