import pytest
from reddwarf.data_loader import Loader

SMALL_CONVO_ID = "9knpdktubt"

def test_load_data_from_api_comments():
    loader = Loader(conversation_id=SMALL_CONVO_ID)
    assert len(loader.comments_data) > 0

    first_comment = loader.comments_data[0]
    expected_keys = [
        'txt',
        'tid',
        'created',
        'tweet_id',
        'quote_src_url',
        'is_seed',
        'is_meta',
        'lang',
        'pid',
        'velocity',
        'mod',
        'active',
        'agree_count',
        'disagree_count',
        'pass_count',
        'count',
        'conversation_id'
    ]
    assert sorted(first_comment.keys()) == sorted(expected_keys)

def test_load_data_from_api_votes():
    loader = Loader(conversation_id=SMALL_CONVO_ID)
    assert len(loader.votes_data) > 0

    first_vote = loader.votes_data[0]
    expected_keys = [
        'pid',
        'tid',
        'vote',
        'weight_x_32767',
        'modified',
        'conversation_id'
    ]
    assert sorted(expected_keys) == sorted(first_vote.keys())

def test_load_data_from_api_math():
    loader = Loader(conversation_id=SMALL_CONVO_ID)
    assert len(loader.math_data) > 0

    # TODO: Test for presences of sub-keys.
    expected_keys = [
        'comment-priorities',
        'user-vote-counts',
        'meta-tids',
        'pca',
        'group-clusters',
        'n',
        'consensus',
        'n-cmts',
        'repness',
        'group-aware-consensus',
        'mod-in',
        'votes-base',
        'base-clusters',
        'mod-out',
        'group-votes',
        'lastModTimestamp',
        'in-conv',
        'tids',
        'lastVoteTimestamp',
        'math_tick',
    ]
    assert sorted(loader.math_data.keys()) == sorted(expected_keys)

def test_load_data_from_api_and_dump_files(tmp_path):
    votes_path = tmp_path / "votes.json"
    comments_path = tmp_path / "comments.json"
    math_path = tmp_path / "math.json"

    loader = Loader(conversation_id=SMALL_CONVO_ID, output_dir=str(tmp_path))

    assert votes_path.exists() == True
    assert comments_path.exists() == True
    assert math_path.exists() == True
