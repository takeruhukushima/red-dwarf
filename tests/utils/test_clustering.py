import pytest
from reddwarf.utils.clustering import run_kmeans, find_optimal_k
from tests.fixtures import polis_convo_data
from tests.helpers import pad_to_size, transform_base_clusters_to_participant_coords
import pandas as pd

@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_run_kmeans_real_data_reproducible(polis_convo_data):
    math_data, data_path, *_ = polis_convo_data

    expected_cluster_centers = [group["center"] for group in math_data["group-clusters"]]
    cluster_count = len(expected_cluster_centers)

    projected_participants = transform_base_clusters_to_participant_coords(math_data["base-clusters"])
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

@pytest.mark.skip
def test_find_optimal_k(polis_convo_data):
    math_data, data_path, *_ = polis_convo_data
    MAX_GROUP_COUNT = 5

    # Get centers and pad to have enough values for testing up to max k groups.
    expected_centers = [group["center"] for group in math_data["group-clusters"]]
    expected_centers = pad_to_size(expected_centers, MAX_GROUP_COUNT)

    results = find_optimal_k()
