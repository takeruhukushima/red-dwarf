import pytest
from reddwarf.utils import polismath
from tests.fixtures import polis_convo_data

# TODO: Figure out if in-conv is always equivalent. If so, remove this test.
@pytest.mark.parametrize("polis_convo_data", ["small", "small-no-meta", "small-with-meta", "medium"], indirect=True)
def test_extract_data_from_polismath(polis_convo_data):
    math_data, _, _ = polis_convo_data
    expected = math_data["in-conv"]

    actual_participant_ids, *_ = polismath.extract_data_from_polismath(math_data)

    assert sorted(expected) == actual_participant_ids