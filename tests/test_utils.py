import pytest
from collections import namedtuple
import pandas as pd
from reddwarf.utils import generate_raw_matrix
from pandas.testing import assert_frame_equal
from random import shuffle
from numpy import nan

Vote = namedtuple("Vote", ["participant_id", "statement_id", "vote", "modified"])

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
    vote_matrix = generate_raw_matrix(votes=simple_votes)
    assert_frame_equal(vote_matrix, expected)

def test_generate_raw_matrix_cutoff_from_timestamp(simple_timestamped_votes):
    shuffled_votes = simple_timestamped_votes.copy()
    shuffle(shuffled_votes)

    last_vote_timestamp = 1735000000
    vote_matrix = generate_raw_matrix(votes=shuffled_votes, cutoff=last_vote_timestamp)

    vote_before = simple_timestamped_votes[-3]
    assert not pd.isna(vote_matrix.at[vote_before["participant_id"], vote_before["statement_id"]])

    vote_after = simple_timestamped_votes[-2]
    assert pd.isna(vote_matrix.at[vote_after["participant_id"], vote_after["statement_id"]])

def test_generate_raw_matrix_cutoff_from_start(simple_timestamped_votes):
    shuffled_votes = simple_timestamped_votes.copy()
    shuffle(shuffled_votes)

    vote_matrix = generate_raw_matrix(votes=shuffled_votes, cutoff=4)

    vote_before = simple_timestamped_votes[3]
    assert not pd.isna(vote_matrix.at[vote_before["participant_id"], vote_before["statement_id"]])

    vote_after = simple_timestamped_votes[4]
    assert pd.isna(vote_matrix.at[vote_after["participant_id"], vote_after["statement_id"]])

def test_generate_raw_matrix_cutoff_from_end(simple_timestamped_votes):
    shuffled_votes = simple_timestamped_votes.copy()
    shuffle(shuffled_votes)

    vote_matrix = generate_raw_matrix(votes=shuffled_votes, cutoff=-2)

    third_last_vote = simple_timestamped_votes[-3]
    assert not pd.isna(vote_matrix.at[third_last_vote["participant_id"], third_last_vote["statement_id"]])

    second_last_vote = simple_timestamped_votes[-2]
    assert pd.isna(vote_matrix.at[second_last_vote["participant_id"], second_last_vote["statement_id"]])
