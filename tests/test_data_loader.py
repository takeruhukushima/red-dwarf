import pytest
from reddwarf.data_loader import Loader

SMALL_CONVO_ID = "9knpdktubt"

def test_load_data_from_api_comments():
    loader = Loader(conversation_id=SMALL_CONVO_ID)
    print(loader.comments_data)
    assert len(loader.comments_data) > 0

@pytest.mark.skip(reason="not ready yet")
def test_load_data_from_api_votes():
    loader = Loader(conversation_id=SMALL_CONVO_ID)
    assert len(loader.votes_data) > 0

@pytest.mark.skip(reason="not ready yet")
def test_load_data_from_api_math():
    loader = Loader(conversation_id=SMALL_CONVO_ID)
    assert len(loader.math_data) > 0
