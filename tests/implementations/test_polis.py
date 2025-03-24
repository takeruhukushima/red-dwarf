import pytest
from tests.fixtures import polis_convo_data
from reddwarf.implementations.polis import run_clustering
from reddwarf.utils.matrix import generate_raw_matrix
from reddwarf.polis import PolisClient

import json

@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_run_clustering(polis_convo_data):
    math_data, data_path, _ = polis_convo_data
    client = PolisClient()
    client.load_data(filepaths=[f"{data_path}/votes.json"])

    projected_ptpts, _, comps, center = run_clustering(
        votes=client.data_loader.votes_data,
        mod_out=math_data["mod-out"],
    )
    print(json.dumps(projected_ptpts.values.tolist(), indent=2))
    print(json.dumps(comps.tolist(), indent=2))
    print(json.dumps((center + 0).tolist(), indent=2))

    assert 1 == 0