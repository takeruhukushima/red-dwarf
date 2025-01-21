import json
from reddwarf.polis_pandas import PolisClient

def test_user_vote_counts():
    # Load the expected data from the JSON file
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['user-vote-counts']
        # Convert keys from string to int. Not sure why API returns them as strings.
        expected_data = {int(k): v for k,v in expected_data.items()}

    # Instantiate the PolisClient and load raw data
    client = PolisClient()
    client.load_data(filepath='sample_data/below-100-ptpts/votes.json')

    # Call the method and assert the result matches the expected data
    assert client.get_user_vote_counts() == expected_data

def test_meta_tids():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['meta-tids']

    client = PolisClient()
    client.load_data(filepath='sample_data/below-100-ptpts/comments.json')

    assert client.get_meta_tids() == sorted(expected_data)

def test_mod_in():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['mod-in']

    client = PolisClient()
    client.load_data(filepath='sample_data/below-100-ptpts/comments.json')

    assert client.get_mod_in() == sorted(expected_data)

def test_mod_out():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['mod-out']

    client = PolisClient()
    client.load_data(filepath='sample_data/below-100-ptpts/comments.json')

    assert sorted(client.get_mod_out()) == sorted(expected_data)

def test_last_vote_timestamp():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['lastVoteTimestamp']

    client = PolisClient()
    client.load_data(filepath='sample_data/below-100-ptpts/votes.json')

    assert client.get_last_vote_timestamp() == expected_data

SHAPE_AXIS = { 'row': 0, 'column': 1 }

def test_participant_count():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['n']

    client = PolisClient()
    client.load_data(filepath='sample_data/below-100-ptpts/votes.json')
    client.get_matrix()

    assert client.participant_count == expected_data

def test_statement_count():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['n-cmts']

    client = PolisClient()
    client.load_data(filepath='sample_data/below-100-ptpts/votes.json')
    client.get_matrix()

    assert client.statement_count == expected_data

def test_impute_missing_values():
    client = PolisClient(is_strict_moderation=False)
    client.load_data(filepath='sample_data/below-100-ptpts/votes.json')
    client.load_data(filepath='sample_data/below-100-ptpts/comments.json')
    matrix_with_missing = client.get_matrix(is_filtered=True)
    client.impute_missing_votes()
    matrix_without_missing = client.get_matrix(is_filtered=True)

    assert matrix_with_missing.isnull().values.sum() > 0
    assert matrix_without_missing.isnull().values.sum() == 0
    assert matrix_with_missing.shape == matrix_without_missing.shape

def test_filtered_participants_grouped():
    with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
        expected_data = json.load(file)['in-conv']

    client = PolisClient(is_strict_moderation=False)
    client.load_data(filepath='sample_data/below-100-ptpts/votes.json')
    client.load_data(filepath='sample_data/below-100-ptpts/comments.json')

    unaligned_matrix = client.get_matrix(is_filtered=True)
    assert sorted(unaligned_matrix.index.to_list()) != sorted(expected_data)

    client.matrix = None
    client.keep_participant_ids = [ 5, 10, 11, 14 ]
    aligned_matrix = client.get_matrix(is_filtered=True)
    assert sorted(aligned_matrix.index.to_list()) == sorted(expected_data)

def test_infer_moderation_type_from_api():
    client = PolisClient()
    assert client.is_strict_moderation is None
    client.load_data(conversation_id="9knpdktubt")
    assert client.is_strict_moderation is not None

def test_load_data_from_report_id():
    client = PolisClient()
    client.load_data(report_id="r5hr48j8y8mpcffk7crmk")


# def test_group_cluster_count():
#     with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
#         expected_data = json.load(file)['group-clusters']

#     client = PolisClient()
#     client.load_data('sample_data/below-100-ptpts/votes.json')

#     assert len(client.get_group_clusters()) == len(expected_data)

# def test_pca_base_cluster_count():
#     with open('sample_data/below-100-ptpts/math-pca2.json', 'r') as file:
#         expected_data = json.load(file)['base-clusters']

#     client = PolisClient()
#     client.load_data('sample_data/below-100-ptpts/votes.json')

#     assert len(client.base_clusters.get('x', [])) == len(expected_data['x'])
#     assert len(client.base_clusters.get('y', [])) == len(expected_data['y'])
