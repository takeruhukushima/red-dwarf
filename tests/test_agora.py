from tests.fixtures import small_convo_math_data, medium_convo_math_data
from reddwarf.agora import run_clustering_v1, Conversation
from reddwarf.polis import PolisClient

def test_run_clustering_real_data_small(small_convo_math_data):
    expected_cluster_sizes = [4, 10, 5]
    _, data_path, *_ = small_convo_math_data

    client = PolisClient()
    client.load_data(filepaths=[f"{data_path}/votes.json"])

    convo: Conversation = {
        "id": "dummy",
        "votes": client.data_loader.votes_data, # type:ignore
    }
    results = run_clustering_v1(conversation=convo)

    assert len(expected_cluster_sizes) == len(results["clusters"])

    for cluster_id, expected_cluster_size in enumerate(expected_cluster_sizes):
        actual_cluster_size = len(results["clusters"][cluster_id]["participants"])
        assert actual_cluster_size == expected_cluster_size

def test_run_clustering_real_data_medium(medium_convo_math_data):
    expected_cluster_sizes = [60, 39, 37, 37, 3]
    _, data_path, *_ = medium_convo_math_data

    client = PolisClient()
    client.load_data(filepaths=[f"{data_path}/votes.json"])

    convo: Conversation = {
        "id": "dummy",
        "votes": client.data_loader.votes_data, # type:ignore
    }
    results = run_clustering_v1(conversation=convo)

    assert len(expected_cluster_sizes) == len(results["clusters"])

    for cluster_id, expected_cluster_size in enumerate(expected_cluster_sizes):
        actual_cluster_size = len(results["clusters"][cluster_id]["participants"])
        assert actual_cluster_size == expected_cluster_size