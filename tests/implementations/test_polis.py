import pytest
from tests.fixtures import polis_convo_data
from reddwarf.implementations.polis import run_clustering
from reddwarf.polis import PolisClient


@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_run_clustering(polis_convo_data):
    math_data, data_path, _ = polis_convo_data
    client = PolisClient()
    client.load_data(filepaths=[f"{data_path}/votes.json"])

    projected_ptpts, comps, _, center = run_clustering(
        votes=client.data_loader.votes_data,
        # TODO: Make this ourselves.
        mod_out=math_data["mod-out"],
    )

    assert comps[0] == pytest.approx(math_data["pca"]["comps"][0])
    assert comps[1] == pytest.approx(math_data["pca"]["comps"][1])
    assert center == pytest.approx(math_data["pca"]["center"])

    # Only include hardcoded active users, since we don't filter them yet.
    # TODO: Generate these from our own code.
    active_pids = [1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 12, 11, 13, 14, 15, 19, 20, 21, 22, 23, 27, 33]
    projected_ptpts = projected_ptpts.loc[active_pids, :]

    assert projected_ptpts["x"].values == pytest.approx(math_data["base-clusters"]["x"])
    assert projected_ptpts["y"].values == pytest.approx(math_data["base-clusters"]["y"])