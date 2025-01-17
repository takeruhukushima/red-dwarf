import pytest
from reddwarf.data_loader import Loader
import requests_cache
requests_cache.install_cache('test_cache')

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

@pytest.mark.skip(reason="not ready yet")
def test_load_data_from_api_votes():
    loader = Loader(conversation_id=SMALL_CONVO_ID)
    assert len(loader.votes_data) > 0

def test_load_data_from_api_math():
    loader = Loader(conversation_id=SMALL_CONVO_ID)
    expected_keys = [
        'comment-priorities',
        'user-vote-counts',
        'meta-tids',
        'pca',
        'group-clusters',
        'n', 'consensus',
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
    assert len(loader.math_data) > 0
