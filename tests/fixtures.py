import pytest
import json

@pytest.fixture
def polis_convo_data(request):
    if request.param == "small":
        path = "tests/fixtures/below-100-ptpts"
    elif request.param == "medium":
        path = "tests/fixtures/above-100-ptpts"
    else:
        raise ValueError("No directory set for loading fixture data")

    filename = "math-pca2.json"
    with open(f"{path}/{filename}", 'r') as f:
        data = json.load(f)

    return data, path, filename