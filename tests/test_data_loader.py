import pytest
from reddwarf.data_loader import Loader

SMALL_CONVO_ID = "9knpdktubt"

def test_load_data_from_api_conversation():
    loader = Loader(conversation_id=SMALL_CONVO_ID)
    assert len(loader.conversation_data) > 0

    expected_keys = [
        'topic', 'description', 'is_anon', 'is_active', 'is_draft',
        'is_public', 'email_domain', 'owner', 'participant_count',
        'created', 'strict_moderation', 'profanity_filter', 'spam_filter',
        'context', 'modified', 'owner_sees_participation_stats', 'course_id',
        'link_url', 'upvotes', 'parent_url', 'vis_type', 'write_type',
        'bgcolor', 'help_type', 'socialbtn_type', 'style_btn',
        'auth_needed_to_vote', 'auth_needed_to_write', 'auth_opt_fb',
        'auth_opt_tw', 'auth_opt_allow_3rdparty', 'help_bgcolor',
        'help_color', 'is_data_open', 'is_curated', 'dataset_explanation',
        'write_hint_type', 'subscribe_type', 'org_id', 'need_suzinvite',
        'use_xid_whitelist', 'prioritize_seed', 'importance_enabled',
        'site_id', 'auth_opt_fb_computed', 'auth_opt_tw_computed', 'translations',
        'ownername', 'is_mod', 'is_owner', 'conversation_id',
    ]
    assert sorted(loader.conversation_data) == sorted(expected_keys)

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
    convo_path = tmp_path / "conversation.json"
    votes_path = tmp_path / "votes.json"
    comments_path = tmp_path / "comments.json"
    math_path = tmp_path / "math-pca2.json"

    Loader(conversation_id=SMALL_CONVO_ID, output_dir=str(tmp_path))

    assert convo_path.exists() == True
    assert votes_path.exists() == True
    assert comments_path.exists() == True
    assert math_path.exists() == True
