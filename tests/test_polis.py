import json
import pytest
from reddwarf.polis import PolisClient

def test_user_vote_counts():
    # Load the expected data from the JSON file
    with open('sample_data/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['user-vote-counts']
        # Convert keys from string to int. Not sure why API returns them as strings.
        expected_data = {int(k): v for k,v in expected_data.items()}

    # Instantiate the PolisClient and load raw data
    client = PolisClient()
    client.load_data('sample_data/votes.json')

    # Call the method and assert the result matches the expected data
    assert client.get_user_vote_counts() == expected_data

def test_meta_tids():
    with open('sample_data/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['meta-tids']

    client = PolisClient()
    client.load_data('sample_data/comments.json')

    assert client.get_meta_tids() == expected_data

def test_mod_in():
    with open('sample_data/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['mod-in']

    client = PolisClient()
    client.load_data('sample_data/comments.json')

    assert client.get_mod_in() == sorted(expected_data)

def test_mod_out():
    with open('sample_data/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['mod-out']

    client = PolisClient()
    client.load_data('sample_data/comments.json')

    assert sorted(client.get_mod_out()) == sorted(expected_data)
