from reddwarf import utils

import pytest
from pandas.testing import assert_frame_equal
import pandas as pd
from collections import namedtuple
from random import shuffle
from numpy import nan

Vote = namedtuple("Vote", ["participant_id", "statement_id", "vote", "modified"])

statement_count = lambda vote_matrix: vote_matrix.shape[1]
participant_count = lambda vote_matrix: vote_matrix.shape[0]

@pytest.fixture
def simple_timestamped_votes():
    # TODO: Make this work with participant and statement indexes that don't start at zero.
    votes = [
        Vote(0, 0,  0, 1730000000),
        Vote(0, 1,  1, 1730000000),
        Vote(2, 0, -1, 1730000000),
        Vote(1, 2,  1, 1735000000),
        Vote(1, 0, -1, 1735700000),
        Vote(1, 1,  1, 1735700000),
    ]
    return [v._asdict() for v in votes]

@pytest.fixture
def simple_votes(simple_timestamped_votes):
    # Bare minimum data without modified date.
    return [
        { k: v for k, v in d.items() if k != 'modified' }
        for d in simple_timestamped_votes
    ]

@pytest.fixture
def simple_vote_matrix():
    real_vote_matrix = pd.DataFrame(
        [
            [ 0,    1, None],
            [-1,    1,    1],
            [-1, None, None],
        ],
        columns=[0, 1, 2], # statement_ids
        index=[0, 1, 2], # participant_ids
        dtype=float,
    )
    # Reproduce property of pivot columns.
    real_vote_matrix.columns.name = "statement_id"
    real_vote_matrix.index.name = "participant_id"
    return real_vote_matrix

def test_generate_raw_matrix_simple(simple_votes, simple_vote_matrix):
    expected = simple_vote_matrix
    vote_matrix = utils.generate_raw_matrix(votes=simple_votes)
    assert_frame_equal(vote_matrix, expected)

@pytest.mark.skip
def test_filter_votes():
    raise

# TODO: Convert these to unit tests of filter_votes()?
def test_generate_raw_matrix_cutoff_from_timestamp(simple_timestamped_votes):
    shuffled_votes = simple_timestamped_votes.copy()
    shuffle(shuffled_votes)

    last_vote_timestamp = 1735000000
    vote_matrix = utils.generate_raw_matrix(votes=shuffled_votes, cutoff=last_vote_timestamp)

    vote_before = simple_timestamped_votes[-3]
    assert not pd.isna(vote_matrix.at[vote_before["participant_id"], vote_before["statement_id"]])

    vote_after = simple_timestamped_votes[-2]
    assert pd.isna(vote_matrix.at[vote_after["participant_id"], vote_after["statement_id"]])

def test_generate_raw_matrix_cutoff_from_start(simple_timestamped_votes):
    shuffled_votes = simple_timestamped_votes.copy()
    shuffle(shuffled_votes)

    vote_matrix = utils.generate_raw_matrix(votes=shuffled_votes, cutoff=4)

    vote_before = simple_timestamped_votes[3]
    assert not pd.isna(vote_matrix.at[vote_before["participant_id"], vote_before["statement_id"]])

    vote_after = simple_timestamped_votes[4]
    assert pd.isna(vote_matrix.at[vote_after["participant_id"], vote_after["statement_id"]])

def test_generate_raw_matrix_cutoff_from_end(simple_timestamped_votes):
    shuffled_votes = simple_timestamped_votes.copy()
    shuffle(shuffled_votes)

    vote_matrix = utils.generate_raw_matrix(votes=shuffled_votes, cutoff=-2)

    third_last_vote = simple_timestamped_votes[-3]
    assert not pd.isna(vote_matrix.at[third_last_vote["participant_id"], third_last_vote["statement_id"]])

    second_last_vote = simple_timestamped_votes[-2]
    assert pd.isna(vote_matrix.at[second_last_vote["participant_id"], second_last_vote["statement_id"]])

def test_get_unvoted_statement_ids(simple_vote_matrix):
    # Null out existing statement column.
    simple_vote_matrix[1] = nan
    # Null out new statement column.
    simple_vote_matrix[3] = None
    unused_statement_ids = simple_vote_matrix.pipe(utils.get_unvoted_statement_ids)
    assert unused_statement_ids == [1, 3]

def test_filter_matrix_no_filtering(simple_vote_matrix):
    initial_matrix = simple_vote_matrix
    all_statement_ids = initial_matrix.columns
    filtered_matrix = utils.filter_matrix(
        vote_matrix=initial_matrix,
        active_statement_ids=all_statement_ids,
        min_user_vote_threshold=0,
    )

    # No filtering
    assert participant_count(initial_matrix) == participant_count(filtered_matrix)
    assert statement_count(initial_matrix) == statement_count(filtered_matrix)

def test_filter_matrix_min_user_vote(simple_vote_matrix):
    initial_matrix = simple_vote_matrix
    all_statement_ids = initial_matrix.columns
    filtered_matrix = utils.filter_matrix(
        vote_matrix=initial_matrix,
        active_statement_ids=all_statement_ids,
        min_user_vote_threshold=2,
    )

    below_threshold_participant_id = 2
    assert below_threshold_participant_id in initial_matrix.index
    assert below_threshold_participant_id not in filtered_matrix.index

    # No statements filtered
    assert statement_count(initial_matrix) == statement_count(filtered_matrix)

def test_filter_matrix_min_user_vote_bypass(simple_vote_matrix):
    initial_matrix = simple_vote_matrix
    all_statement_ids = initial_matrix.columns
    filtered_matrix = utils.filter_matrix(
        vote_matrix=initial_matrix,
        active_statement_ids=all_statement_ids,
        min_user_vote_threshold=2,
        keep_participant_ids=[2],
    )

    below_threshold_participant_id = 2
    assert below_threshold_participant_id in initial_matrix.index
    assert below_threshold_participant_id in filtered_matrix.index

    # No statements filtered
    assert statement_count(initial_matrix) == statement_count(filtered_matrix)

def test_filter_matrix_mod_out_statement(simple_vote_matrix):
    initial_matrix = simple_vote_matrix
    all_statement_ids_but_one = initial_matrix.columns[:-1]
    filtered_matrix = utils.filter_matrix(
        vote_matrix=initial_matrix,
        active_statement_ids=all_statement_ids_but_one,
        min_user_vote_threshold=0,
    )

    assert participant_count(initial_matrix) == participant_count(filtered_matrix)

    assert statement_count(initial_matrix) == 3
    assert statement_count(filtered_matrix) == 2

def test_filter_matrix_mod_out_statement_with_zero(simple_vote_matrix):
    initial_matrix = simple_vote_matrix
    # Set statement 1 to have no votes.
    initial_matrix[1] = None
    all_statement_ids = initial_matrix.columns
    filtered_matrix = utils.filter_matrix(
        vote_matrix=initial_matrix,
        active_statement_ids=all_statement_ids,
        min_user_vote_threshold=0,
        unvoted_filter_type="zero",
    )
    assert statement_count(initial_matrix) == statement_count(filtered_matrix)

    assert (filtered_matrix[1] == 0).all()

def test_impute_missing_votes():
    # Manually calculated
    expected_matrix = pd.DataFrame(
        [
            [ 0,    1,    0,  1/3],
            [-1,    0,    0,   -1],
            [-1,  0.5,    0,    1],
            [-1,  0.5,    0,    1],
        ],
        columns=[0, 1, 2, 3], # statement_ids
        index=[0, 1, 2, 3], # participant_ids
        dtype=float,
    )

    initial_matrix = pd.DataFrame(
        [
            [ 0,    1, None, None],
            [-1,    0, None,   -1],
            [-1, None, None,    1],
            [-1, None,    0,    1],
        ],
        columns=[0, 1, 2, 3], # statement_ids
        index=[0, 1, 2, 3], # participant_ids
        dtype=float,
    )
    imputed_matrix = utils.impute_missing_votes(vote_matrix=initial_matrix)
    assert_frame_equal(imputed_matrix, expected_matrix)

def test_impute_missing_votes_no_vote_statement_error():
    initial_matrix = pd.DataFrame(
        [
            [ 0,    1, None, None],
            [-1,    0, None,   -1],
            [-1, None, None,    1],
            [-1, None, None,    1],
        ],
        columns=[0, 1, 2, 3], # statement_ids
        index=[0, 1, 2, 3], # participant_ids
        dtype=float,
    )
    with pytest.raises(ValueError):
        utils.impute_missing_votes(vote_matrix=initial_matrix)

@pytest.mark.skip
def test_run_pca():
    raise

@pytest.mark.skip
def test_run_kmeans():
    raise

@pytest.mark.skip
def test_find_optimal_k():
    raise

@pytest.mark.skip
def test_scale_projected_data():
    raise
