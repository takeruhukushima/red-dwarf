from pandas._testing import assert_almost_equal
import pytest
from reddwarf.data_loader import Loader
from tests.fixtures import polis_convo_data
from reddwarf.utils.consensus import select_consensus_statements
from reddwarf.utils.matrix import generate_raw_matrix
from reddwarf.utils.statements import process_statements


@pytest.mark.parametrize(
    "polis_convo_data",
    ["small-no-meta", "small-with-meta", "medium-no-meta", "medium-with-meta"],
    indirect=True,
)
def test_select_consensus_statements_real_data(polis_convo_data):
    fixture = polis_convo_data

    loader = Loader(
        filepaths=[
            f"{fixture.data_dir}/votes.json",
            f"{fixture.data_dir}/comments.json",
            f"{fixture.data_dir}/conversation.json",
        ]
    )

    raw_vote_matrix = generate_raw_matrix(votes=loader.votes_data)

    _, _, mod_out_statement_ids, _ = process_statements(
        statement_data=loader.comments_data,
    )

    calculated = select_consensus_statements(
        # NOTE: Not filtered down to clusterable_pids like we normally do.
        vote_matrix=raw_vote_matrix,
        mod_out_statement_ids=mod_out_statement_ids,
    )

    expected = fixture.math_data["consensus"]

    # Ensure same number of statements for agree/disagree
    assert len(calculated["agree"]) == len(expected["agree"])
    assert len(calculated["disagree"]) == len(expected["disagree"])

    # Ensure same statement IDs
    get_tids = lambda data, direction: [st["tid"] for st in data[direction]]
    assert get_tids(calculated, "agree") == get_tids(expected, "agree")
    assert get_tids(calculated, "disagree") == get_tids(expected, "disagree")

    # Ensure same z-score test values
    get_scores = lambda data, direction: [st["p-test"] for st in data[direction]]
    assert_almost_equal(get_scores(calculated, "agree"), get_scores(expected, "agree"))
    assert_almost_equal(
        get_scores(calculated, "disagree"), get_scores(expected, "disagree")
    )
