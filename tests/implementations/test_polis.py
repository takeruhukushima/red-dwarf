import pytest
from tests.fixtures import polis_convo_data
from reddwarf.implementations.polis import run_clustering
from reddwarf.polis import PolisClient
from reddwarf.utils import polismath


# This test will only match polismath for sub-100 participant convos.
# For testing our code agaisnt real data with against larger conversations,
# we'll need to implement base clustering.
@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_run_clustering(polis_convo_data):
    math_data, data_path, _ = polis_convo_data
    client = PolisClient()
    client.load_data(filepaths=[f"{data_path}/votes.json"])

    # TODO: Derive these ourselves without math_data.
    statement_ids_moderated_out = math_data["mod-out"]
    participant_ids_active = math_data["in-conv"]

    projected_ptpts, comps, _, center = run_clustering(
        votes=client.data_loader.votes_data,
        # TODO: Make this ourselves.
        mod_out=statement_ids_moderated_out,
    )

    assert comps[0] == pytest.approx(math_data["pca"]["comps"][0])
    assert comps[1] == pytest.approx(math_data["pca"]["comps"][1])
    assert center == pytest.approx(math_data["pca"]["center"])

    # Since projections are for base clusters, need to do some work to match its ordering.

    # Filter out for active users.
    projected_ptpts = projected_ptpts.loc[participant_ids_active, :]

    # But projections won't be in matching order, so re-order by base_cluster_id, like polismath does.
    participant_to_base_cluster, _ = polismath.create_bidirectional_id_maps(math_data["base-clusters"])
    base_cluster_ids = [participant_to_base_cluster[pid] for pid in projected_ptpts.index]
    projected_ptpts = (projected_ptpts
        .assign(base_cluster_id=base_cluster_ids)
        .sort_values(by="base_cluster_id")
    )

    assert projected_ptpts["x"].values == pytest.approx(math_data["base-clusters"]["x"])
    assert projected_ptpts["y"].values == pytest.approx(math_data["base-clusters"]["y"])