import numpy as np
import pytest
from tests.fixtures import polis_convo_data
from reddwarf.utils import stats

from reddwarf.utils import stats, polismath, matrix
from reddwarf.data_loader import Loader

def test_importance_metric_no_votes():
    expected_importance = [ 1/4,   2/4,   1,     2,      4   ]
    comment_extremity =   [(1-1), (2-1), (4-1), (8-1), (16-1)]
    # extremity values    [ 0,     1,     3,     7,     15   ]

    calculated_importance = stats.importance_metric(
        n_agree=0,
        n_disagree=0,
        n_total=0,
        extremity=comment_extremity,
    )

    assert expected_importance == calculated_importance.tolist()

def test_importance_metric_limits_no_extremity_all_agree():
    comment_extremity = 0
    expected_importance = 1

    calculated_importance = stats.importance_metric(
        n_agree=10000,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert expected_importance == pytest.approx(calculated_importance, abs=0.001)

def test_importance_metric_limits_no_extremity_all_disagree():
    comment_extremity = 0
    expected_importance = 0

    calculated_importance = stats.importance_metric(
        n_agree=0,
        n_disagree=10000,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert expected_importance == pytest.approx(calculated_importance, abs=0.001)

def test_importance_metric_limits_no_extremity_split_full_engagement():
    comment_extremity = 0
    expected_importance = 1/4

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=5000,
        n_disagree=5000,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert calculated_priority == pytest.approx(expected_importance, abs=0.001)

def test_importance_metric_limits_no_extremity_all_pass():
    comment_extremity = 0
    expected_importance = 0

    calculated_importance = stats.importance_metric(
        n_agree=0,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert expected_importance == pytest.approx(calculated_importance, abs=0.001)

def test_importance_metric_limits_high_extremity_all_agree():
    comment_extremity = 4.0
    expected_importance = comment_extremity+1

    calculated_importance = stats.importance_metric(
        n_agree=10000,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert expected_importance == pytest.approx(calculated_importance, abs=0.001)

def test_importance_metric_limits_high_extremity_all_disagree():
    comment_extremity = 4.0
    expected_importance = 0

    calculated_importance = stats.importance_metric(
        n_agree=0,
        n_disagree=10000,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert expected_importance == pytest.approx(calculated_importance, abs=0.001)

def test_importance_metric_limits_high_extremity_all_pass():
    comment_extremity = 4.0
    expected_importance = 0

    calculated_importance = stats.importance_metric(
        n_agree=0,
        n_disagree=0,
        n_total=10000,
        extremity=comment_extremity,
    )
    assert expected_importance == pytest.approx(calculated_importance, abs=0.001)

# TODO: Use this test to more fully show trends and effects.
@pytest.mark.skip()
def test_importance_metric_array():
    expected_importances = [0, 7**2]

    calculated_priority = stats.importance_metric(
        n_agree=   [    0,     0],
        n_disagree=[10000, 10000],
        n_total=   [10000, 10000],
        extremity= [  4.0,   4.0],
    )
    assert calculated_priority == pytest.approx(expected_importances, abs=0.001)

def test_importance_metric_smaller_full_agree_pseudo_count():
    # Should approach 1 at higher volume of votes
    pseudo_counts =        [1,     10]
    # Approaches slower with higher pseudo-count.
    expected_importances = [0.9804, 0.84027778]

    calculated_priority = stats.importance_metric(
        n_agree=   [100, 100],
        n_disagree=[  0,   0],
        n_total=   [100, 100],
        extremity= [  0,   0],
        pseudo_count=pseudo_counts,
    )
    assert calculated_priority == pytest.approx(expected_importances, abs=0.001)

def test_priority_metric_no_votes():
    prio = lambda n: (81/16)*(4**n)
    # expected_values = [ 5.0625,  20.25,   81,      324,     1296   ]
    expected_priority = [ prio(0), prio(1), prio(2), prio(3), prio(4)]
    comment_extremity = [ 0,       1,       (4-1),   (8-1),   (16-1) ]
    # extremity values  [ 0,       1,        3,       7,       15    ]

    calculated_priority = stats.priority_metric(
        is_meta=False,
        n_agree=0,
        n_disagree=0,
        n_total=0,
        extremity=comment_extremity,
    )

    assert expected_priority == calculated_priority.tolist()

# TODO: Investigate why "medium-with-meta" and "medium-no-meta" don't pass.
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta"], indirect=True)
def test_priority_metric_real_data(polis_convo_data):
    math_data, _, _, _ = polis_convo_data
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

# TODO: Use this test to more fully show trends and effects.
def test_priority_metric_array():
    expected_priorities = [0, 7**2]

    calculated_priority = stats.priority_metric(
        is_meta=   [False,  True],
        n_agree=   [    0,     0],
        n_disagree=[10000, 10000],
        n_total=   [10000, 10000],
        extremity= [  4.0,   4.0],
    )
    assert calculated_priority == pytest.approx(expected_priorities, abs=0.001)

# TODO: Investigate why "medium-with-meta" doesn't work. (59/60 mismatched)
@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-no-meta"], indirect=True)
def test_group_aware_consensus_real_data(polis_convo_data):
    math_data, path, _, _ = polis_convo_data
    loader = Loader(filepaths=[
        f'{path}/votes.json',
        f'{path}/comments.json',
        f'{path}/conversation.json',
    ])
    VOTES = loader.votes_data
    # STATEMENTS = loader.comments_data

    # _, _, mod_out, _ = stmnts.process_statements(statement_data=STATEMENTS)
    # print(mod_out)

    vote_matrix = matrix.generate_raw_matrix(votes=VOTES)
    # TODO: Why do moderated out statements not plug into comment stats? BUG?
    # vote_matrix = matrix.simple_filter_matrix(
    #     vote_matrix=vote_matrix,
    #     mod_out_statement_ids=mod_out,
    # )

    # Get list of all active participant ids, since Polis has some edge-cases
    # that keep specific participants, and we need to keep them from being filtered out.
    all_clustered_participant_ids, cluster_labels = polismath.extract_data_from_polismath(math_data)
    vote_matrix = vote_matrix.loc[all_clustered_participant_ids, :]

    # Generate stats all groups and all statements.
    _, gac_df = stats.calculate_comment_statistics_dataframes(
        vote_matrix=vote_matrix,
        cluster_labels=cluster_labels,
    )

    calculated_gac = {
        str(pid): float(row[0])
        for pid, row in gac_df.iterrows()
    }

    assert calculated_gac == pytest.approx(math_data["group-aware-consensus"])