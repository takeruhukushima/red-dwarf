import numpy as np
import pytest
from tests.fixtures import small_convo_math_data
from reddwarf.utils import stats

def test_priority_metric(small_convo_math_data):
    math_data, _, _ = small_convo_math_data
    votes_base = math_data["votes-base"]
    for statement_id, votes in votes_base.items():
        expected_priority = math_data["comment-priorities"][statement_id]

        is_meta = int(statement_id) in math_data["meta-tids"]
        n_agree = (np.asarray(votes["A"]) == 1).sum()
        n_disagree = (np.asarray(votes["D"]) == 1).sum()
        n_total = (np.asarray(votes["S"]) == 1).sum()
        comment_extremity = math_data["pca"]["comment-extremity"][int(statement_id)]

        calculated_priority = stats.priority_metric(
            is_meta=is_meta,
            n_agree=n_agree,
            n_disagree=n_disagree,
            n_total=n_total,
            extremity=comment_extremity,
        )
        assert expected_priority == pytest.approx(calculated_priority)

def test_priority_metric_for_meta_default():
    meta_priority_default = 7
    expected_priority = meta_priority_default**2

    calculated_priority = stats.priority_metric(
        is_meta=True,
        n_agree=10,
        n_disagree=0,
        n_total=25,
        extremity=0,
    )

    assert calculated_priority == expected_priority

def test_priority_metric_for_meta_override():
    meta_priority_override = 10
    expected_priority = meta_priority_override**2

    calculated_priority = stats.priority_metric(
        is_meta=True,
        n_agree=10,
        n_disagree=0,
        n_total=25,
        extremity=1.0,
        meta_priority=meta_priority_override,
    )
    assert calculated_priority == expected_priority

def test_priority_metric_limits_no_extremity_all_passing():
    comment_extremity = 0
    expected_priority = 0

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=0,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority)

def test_priority_metric_limits_no_extremity_all_disagree():
    comment_extremity = 0
    expected_priority = 0

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=0,
        n_disagree=10000,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority, abs=0.001)

def test_priority_metric_limits_no_extremity_all_agree():
    comment_extremity = 0
    expected_priority = 1

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=10000,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority, abs=0.001)

def test_priority_metric_limits_no_extremity_split_full_engagement():
    comment_extremity = 0
    expected_priority = 1/4

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=5000,
        n_disagree=5000,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority, abs=0.001)

def test_priority_metric_limits_high_extremity_all_passed():
    comment_extremity = 4.0
    expected_priority = 0

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=0,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority)

def test_priority_metric_limits_high_extremity_all_agree():
    comment_extremity = 4.0
    expected_priority = (comment_extremity+1)**2

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=10000,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority, abs=0.01)

def test_priority_metric_limits_high_extremity_all_disagree():
    comment_extremity = 4.0
    expected_priority = 0

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=0,
        n_disagree=10000,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_priority, abs=0.001)