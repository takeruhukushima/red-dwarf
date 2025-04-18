import pytest
from reddwarf.data_presenter import print_repness

def test_print_repness_minimal():
    mock_repness = {
        "0": [
            {"tid": 0, "n-success": 9, "n-trials": 10, "repful-for": "agree"},
        ],
        "1": [
            {"tid": 1, "n-success": 9, "n-trials": 10, "repful-for": "disagree"},
        ]
    }

    mock_statements = [
        {"statement_id": 0, "txt": "This is the first statement"},
        {"statement_id": 1, "txt": "This is the second statement."},
    ]

    print_repness(repness=mock_repness, statements_data=mock_statements)

@pytest.mark.skip
def test_print_repness_real_data():
    raise NotImplementedError