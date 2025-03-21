import pytest
import json

@pytest.fixture
def small_convo_math_data():
    path = "tests/fixtures/below-100-ptpts"
    filename = "math-pca2.json"
    with open(f"{path}/{filename}", 'r') as f:
        data = json.load(f)

    return data, path, filename

@pytest.fixture
def medium_convo_math_data():
    path = "tests/fixtures/above-100-ptpts"
    filename = "math-pca2.json"
    with open(f"{path}/{filename}", 'r') as f:
        data = json.load(f)

    return data, path, filename
