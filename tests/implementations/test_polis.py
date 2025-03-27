import pytest
from tests.fixtures import polis_convo_data
from reddwarf.implementations.polis import run_clustering
from reddwarf.polis import PolisClient
from reddwarf.utils.statements import process_statements


def transform_base_clusters_to_participant_coords(base_clusters):
    """
    Transform base clusters data into a list of dictionaries with participant_id and xy coordinates.

    Args:
        base_clusters (dict): A dictionary containing base clusters data with 'id', 'members', 'x', and 'y' keys.

    Returns:
        list: A list of dictionaries, each containing a participant_id and their xy coordinates.
    """
    # For now, ensure failure if a base-cluster has more than one member, as the test assumes that.
    def get_only_member_or_raise(members):
        if len(members) != 1:
            raise Exception("A base-cluster has more than one member when it cannot")

        return members[0]

    return [
        {
            "participant_id": get_only_member_or_raise(members),
            "xy": [x, y]
        }
        for members, x, y in zip(
            base_clusters['members'],
            base_clusters['x'],
            base_clusters['y']
        )
    ]

# This test will only match polismath for sub-100 participant convos.
# For testing our code agaisnt real data with against larger conversations,
# we'll need to implement base clustering.
@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_run_clustering(polis_convo_data):
    # We hardcode this because Polis has some bespoke rules that keep these IDs in for clustering.
    # TODO: Try to determine why each pid is kept. Can maybe determine by incrementing through vote history.
    #  5 -> 1 vote @ statement #26 (participant #2's)
    # 10 -> 2 vote @ statements #21 (participant #1's) & #29 (their own, moderated in).
    # 11 -> 1 vote @ statement #29 (participant #10's)
    # 14 -> 1 vote @ statement #27 (participant #6's)
    keep_participant_ids = [ 5, 10, 11, 14 ]
    math_data, data_path, _ = polis_convo_data
    # Transpose base cluster coords into participant_ids
    expected_projected_ptpts = transform_base_clusters_to_participant_coords(math_data["base-clusters"])

    client = PolisClient()
    client.load_data(filepaths=[
        f"{data_path}/votes.json",
        # Loading these helps generate mod_out_statement_ids
        f"{data_path}/comments.json",
        f"{data_path}/conversation.json",
    ])

    _, _, mod_out_statement_ids, _ = process_statements(statement_data=client.data_loader.comments_data)

    projected_ptpts, comps, _, center = run_clustering(
        votes=client.data_loader.votes_data,
        mod_out_statement_ids=mod_out_statement_ids,
        keep_participant_ids=keep_participant_ids,
    )

    assert comps[0] == pytest.approx(math_data["pca"]["comps"][0])
    assert comps[1] == pytest.approx(math_data["pca"]["comps"][1])
    assert center == pytest.approx(math_data["pca"]["center"])

    # Ensure we have as many expected coords as calculated coords.
    assert len(projected_ptpts.index) == len(expected_projected_ptpts)

    for projection in expected_projected_ptpts:
        expected_xy = projection["xy"]
        calculated_xy = projected_ptpts.loc[projection["participant_id"], ["x", "y"]].values

        assert calculated_xy == pytest.approx(expected_xy)
