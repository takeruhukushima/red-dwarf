import pytest
from tests.fixtures import polis_convo_data
from reddwarf.data_loader import Loader
from reddwarf.utils.statements import process_statements


@pytest.mark.parametrize(
    "polis_convo_data",
    ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"],
    indirect=True,
)
def test_meta_tids(polis_convo_data):
    fixture = polis_convo_data
    expected_data = fixture.math_data["meta-tids"]

    loader = Loader(filepaths=[f"{fixture.data_dir}/comments.json"])
    _, _, _, meta_statement_ids = process_statements(loader.comments_data)

    assert meta_statement_ids == sorted(expected_data)


@pytest.mark.parametrize(
    "polis_convo_data",
    ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"],
    indirect=True,
)
def test_statements_mod_out(polis_convo_data):
    fixture = polis_convo_data
    expected_data = fixture.math_data["mod-out"]

    loader = Loader(filepaths=[f"{fixture.data_dir}/comments.json"])
    _, _, mod_out_statement_ids, _ = process_statements(loader.comments_data)

    assert mod_out_statement_ids == sorted(expected_data)


@pytest.mark.parametrize(
    "polis_convo_data",
    ["small-no-meta", "small-with-meta", "medium-with-meta", "medium-no-meta"],
    indirect=True,
)
def test_statements_mod_in(polis_convo_data):
    fixture = polis_convo_data
    expected_data = fixture.math_data["mod-in"]

    loader = Loader(filepaths=[f"{fixture.data_dir}/comments.json"])
    _, mod_in_statement_ids, _, _ = process_statements(loader.comments_data)

    assert mod_in_statement_ids == sorted(expected_data)
