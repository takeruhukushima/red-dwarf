import pytest
from tests.fixtures import polis_convo_data
from reddwarf.implementations.polis import run_clustering
from reddwarf.data_loader import Loader
from reddwarf.utils.statements import process_statements
from reddwarf.utils.polismath import extract_data_from_polismath
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_frame_equal
from tests import helpers
import numpy as np

# This test will only match polismath for sub-100 participant convos.
# For testing our code agaisnt real data with against larger conversations,
# we'll need to implement base clustering.
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta"], indirect=True)
def test_run_clustering_real_data(polis_convo_data):
    fixture = polis_convo_data
    math_data = helpers.flip_signs_by_key(nested_dict=fixture.math_data, keys=["pca.center", "pca.comment-projection", "base-clusters.x", "base-clusters.y", "group-clusters[*].center"])

    # We hardcode this because Polis has some bespoke rules that keep these IDs in for clustering.
    # TODO: Try to determine why each pid is kept. Can maybe determine by incrementing through vote history.
    #  5 -> 1 vote @ statement #26 (participant #2's)
    # 10 -> 2 vote @ statements #21 (participant #1's) & #29 (their own, moderated in).
    # 11 -> 1 vote @ statement #29 (participant #10's)
    # 14 -> 1 vote @ statement #27 (participant #6's)

    # We're moving this into polis_convo_data, for better parametrization.
    # keep_participant_ids = [ 5, 10, 11, 14 ]

    # We force kmeans to find a specific value because Polis does this for something they call k-smoothing.
    # TODO: Get k-smoothing working, and simulate how k is held back.
    force_group_count = len(math_data["group-clusters"])

    # Transpose base cluster coords into participant_ids
    expected_projected_ptpts = helpers.transform_base_clusters_to_participant_coords(math_data["base-clusters"])

    max_group_count = 5
    init_centers = [group["center"] for group in math_data["group-clusters"]]
    init_centers = helpers.pad_centroid_list_to_length(init_centers, max_group_count)

    loader = Loader(filepaths=[
        f"{fixture.data_dir}/votes.json",
        # Loading these helps generate mod_out_statement_ids
        f"{fixture.data_dir}/comments.json",
        f"{fixture.data_dir}/conversation.json",
    ])

    _, _, mod_out_statement_ids, _ = process_statements(statement_data=loader.comments_data)

    result = run_clustering(
        votes=loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
        keep_participant_ids=fixture.keep_participant_ids,
        max_group_count=max_group_count,
        init_centers=init_centers,
        force_group_count=force_group_count,
    )

    assert pytest.approx(result.components[0]) == math_data["pca"]["comps"][0]
    assert pytest.approx(result.components[1]) == math_data["pca"]["comps"][1]
    assert pytest.approx(result.means) == math_data["pca"]["center"]

    # Test projected statements
    # Convert [[x_1, y_1], [x_2, y_2], ...] into [[x_1, x_2, ...], list[y_1, y_2, ...]]
    actual = result.projected_statements.values.transpose()
    expected = math_data["pca"]["comment-projection"]
    assert_array_almost_equal(actual, expected)

    # Ensure we have as many expected coords as calculated coords.
    assert len(result.projected_participants.index) == len(expected_projected_ptpts)

    for projection in expected_projected_ptpts:
        expected_xy = projection["xy"]
        calculated_xy = result.projected_participants.loc[projection["participant_id"], ["x", "y"]].values

        assert calculated_xy == pytest.approx(expected_xy)

    # Check that the cluster labels all match when K is forced to match.
    _, expected_cluster_labels = extract_data_from_polismath(math_data)
    calculated_cluster_labels = result.projected_participants["cluster_id"].values
    assert_array_equal(calculated_cluster_labels, expected_cluster_labels) # type:ignore

    # from reddwarf.data_presenter import generate_figure
    # print(projected_ptpts)
    # # Flip these to render figure properly.
    # projected_ptpts[["x", "y"]] = -projected_ptpts[["x", "y"]]
    # labels = projected_ptpts["cluster_id"].values
    # generate_figure(projected_ptpts, labels)

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta"], indirect=True)
def test_run_clustering_is_reproducible(polis_convo_data):
    fixture = polis_convo_data

    loader = Loader(filepaths=[
        f"{fixture.data_dir}/votes.json",
        # Loading these helps generate mod_out_statement_ids
        f"{fixture.data_dir}/comments.json",
        f"{fixture.data_dir}/conversation.json",
    ])

    _, _, mod_out_statement_ids, _ = process_statements(statement_data=loader.comments_data)

    cluster_run_1 = run_clustering(
        votes=loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
    )

    max_group_count = 5
    padded_cluster_centers = helpers.pad_centroid_list_to_length(cluster_run_1.cluster_centers, max_group_count)

    cluster_run_2 = run_clustering(
        votes=loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
        init_centers=padded_cluster_centers,
    )

    # same number of clusters
    assert len(cluster_run_1.cluster_centers) == len(cluster_run_2.cluster_centers) # type:ignore

    # projected statements and cluster IDs
    assert_frame_equal(cluster_run_1.projected_participants, cluster_run_2.projected_participants)
    # components/eigenvectors
    assert cluster_run_1.components.tolist() == cluster_run_2.components.tolist()
    # statement centers/means
    assert cluster_run_1.means.tolist() == cluster_run_2.means.tolist()
    assert cluster_run_1.cluster_centers.tolist() == cluster_run_2.cluster_centers.tolist()