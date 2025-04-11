import pandas as pd
from numpy.testing import assert_allclose, assert_array_almost_equal
import pytest
from tests.fixtures import polis_convo_data
from reddwarf.utils import pca as PcaUtils
from reddwarf.utils import matrix as MatrixUtils
from reddwarf.data_loader import Loader
from tests import helpers


def test_run_pca_toy():
    # Pre-calculated
    expected_projections = [
        [ 1.238168, -0.173377],
        [-0.998131, -0.268031],
        [-0.123951,  0.205794],
        [-0.116086,  0.235614],
    ]
    expected_eigenvectors = [
        [0.87418057, -0.48400607, -0.0393249],
        [0.47382435,  0.86790524, -0.1491005],
    ]
    expected_eigenvalues = [ 0.85272226, 0.06658833 ]

    initial_matrix = pd.DataFrame(
        [
            [ 1, 0,  0],
            [-1, 1,  0.1],
            [ 0, 1,  0.1],
            [ 0, 1, -0.1],
        ],
        columns=pd.Index([0, 1, 2]), # statement_ids
        index=pd.Index([0, 1, 2, 3]), # participant_ids
        dtype=float,
    )
    actual_projected_participants, _, pca = PcaUtils.run_pca(vote_matrix=initial_matrix)
    actual_eigenvectors = pca.components_
    actual_eigenvalues = pca.explained_variance_
    assert_allclose(actual_projected_participants.values, expected_projections, rtol=10**-5)
    assert_allclose(actual_eigenvectors, expected_eigenvectors, rtol=10**-5)
    assert_allclose(actual_eigenvalues, expected_eigenvalues, rtol=10**-5)

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta"], indirect=True)
def test_run_pca_real_data_below_100(polis_convo_data):
    math_data, data_path, *_, _ = polis_convo_data
    math_data = helpers.flip_signs_by_key(nested_dict=math_data, keys=["pca.center", "pca.comment-projection", "base-clusters.x", "base-clusters.y", "group-clusters[*].center"])
    expected_pca = math_data["pca"]

    # Just fetch the moderated out statements from polismath (no need to recalculate here)
    mod_out_statement_ids = math_data["mod-out"]

    loader = Loader(filepaths=[f"{data_path}/votes.json"])
    real_vote_matrix = MatrixUtils.generate_raw_matrix(votes=loader.votes_data)

    real_vote_matrix = MatrixUtils.simple_filter_matrix(
        vote_matrix=real_vote_matrix,
        mod_out_statement_ids=mod_out_statement_ids,
    )

    actual_projected_participants, actual_projected_statements, actual_pca = PcaUtils.run_pca(vote_matrix=real_vote_matrix)

    assert actual_pca.components_[0] == pytest.approx(expected_pca["comps"][0])
    assert actual_pca.components_[1] == pytest.approx(expected_pca["comps"][1])
    assert actual_pca.mean_ == pytest.approx(expected_pca["center"])

    expected_participant_projections = helpers.transform_base_clusters_to_participant_coords(base_clusters=math_data["base-clusters"])
    for expected in expected_participant_projections:
        pid = expected["participant_id"]
        assert_array_almost_equal(actual_projected_participants.loc[pid, ["x","y"]].values, expected["xy"])

    # print(actual_projected_statements.values.transpose())
    # print(expected_pca["comment-projection"])
    # print(PcaUtils.pca_project_cmnts(actual_pca.components_, actual_pca.mean_).transpose())
    # print(PcaUtils.pca_project_cmnts(expected_pca["comps"], np.asarray(expected_pca["center"])).transpose())
    assert_array_almost_equal(actual_projected_statements.values.transpose(), expected_pca["comment-projection"])

@pytest.mark.parametrize("polis_convo_data", ["medium-with-meta"], indirect=True)
def test_run_pca_real_data_above_100(polis_convo_data):
    math_data, data_path, *_, _ = polis_convo_data
    # Some signs are flipped for the "medium-with-meta" fixture data, because signs are arbitrary in PCA.
    # If we initialize differently later on, it should flip and match.
    math_data = helpers.flip_signs_by_key(nested_dict=math_data, keys=["pca.comps[0]"])
    expected_pca = math_data["pca"]

    # Just fetch the moderated out statements from polismath (no need to recalculate here)
    mod_out_statement_ids = math_data["mod-out"]

    loader = Loader(filepaths=[f"{data_path}/votes.json"])
    real_vote_matrix = MatrixUtils.generate_raw_matrix(votes=loader.votes_data)

    real_vote_matrix = MatrixUtils.simple_filter_matrix(
        vote_matrix=real_vote_matrix,
        mod_out_statement_ids=mod_out_statement_ids,
    )

    _, _, actual_pca = PcaUtils.run_pca(vote_matrix=real_vote_matrix)

    assert actual_pca.components_[0] == pytest.approx(expected_pca["comps"][0])
    assert actual_pca.components_[1] == pytest.approx(expected_pca["comps"][1])
    assert actual_pca.mean_ == pytest.approx(expected_pca["center"])

# TODO: Find a good place to run integration tests against real remote datasets.
@pytest.mark.skip
def test_run_pca_real_data_testing():
    # loader = Loader(polis_id="r6iwc8tb22am6hpd8f7c5") # 20 voters, 0/146 meta WORKS
    # loader = Loader(polis_id="r5hr48j8y8mpcffk7crmk") # 27 voters, 11/63 meta WORKS strict=no
    # loader = Loader(polis_id="r7dr5tzke7pbpbajynkv8") #  69 voters, 74/133 meta NOPE 2/133 mismatch strict=yes
    # loader = Loader(polis_id="r8jhyfp54cyanhu26cz3v") #  90 voters, 0/51 meta WORKS strict=no
    # loader = Loader(polis_id="r2dtdbwbsrzu8bj8wmpmc") #  87 voters, 3/92 meta WORKS strict=no
    # loader = Loader(polis_id="r9cnkypdddkmnefz8k2bn") # 113 voters, 0/40 meta WORKS strict=no
    # loader = Loader(polis_id="r4zurrpweay6khmsaab4e") # 108 voters, 0/82 meta WORKS strict=yes
    # loader = Loader(polis_id="r5ackfcj5dmawnt2de7x7") # 109 voters, 16/95 meta NOPE DIFF SHAPES? bug?
    # loader = Loader(polis_id="r6bpmcmizi2kyvhzkhfr7") # 23 voters, 0/9 meta WORKS strict=yes
    # loader = Loader(polis_id="r4tfrhexm3dhawcfav6dy") # 116 voters, 0/170 meta WORKS
    # loader = Loader(polis_id="r7f89nk32sjuknecjbrxp") # 135 voters, 0/52 meta WORKS
    # loader = Loader(polis_id="r2xzujbewezpjdc9pmzhd") # 135 voters, 3/48 meta WORKS
    # loader = Loader(polis_id="r8bzudrhs8j6petppicrj") # 140 voters, 0/46 meta WORKS
    # loader = Loader(polis_id="r4zdxrdscmukmkakmbz3k") # 145 voters, 0/117 meta WORKS
    loader = Loader(polis_id="r45ru4zfmutun54ttskne") # 336 voters, 15/148 meta NOPE 76/148 mismatch
    # loader = Loader(polis_id="") # x voters, x/x meta ...
    real_vote_matrix = MatrixUtils.generate_raw_matrix(votes=loader.votes_data)
    math_data = loader.math_data
    expected_pca = math_data["pca"]

    real_vote_matrix = MatrixUtils.simple_filter_matrix(
        vote_matrix=real_vote_matrix,
        mod_out_statement_ids=math_data["mod-out"],
    )

    _, _, actual_pca = PcaUtils.run_pca(vote_matrix=real_vote_matrix)
    # powerit_pca = PcaUtils.powerit_pca(real_vote_matrix.values)

    # We test absolute because PCA methods don't always give the same sign, and can flip.
    # TODO: Try to remove this.
    assert actual_pca.components_[0] == pytest.approx(expected_pca["comps"][0])
    assert actual_pca.components_[1] == pytest.approx(expected_pca["comps"][1])

    assert actual_pca.mean_ == pytest.approx(expected_pca["center"])

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-with-meta"], indirect=True)
def test_with_proj_and_extremity(polis_convo_data):
    math_data, _, _, _ = polis_convo_data
    # Invert to correct for flipped signs in polismath.
    math_data = helpers.flip_signs_by_key(nested_dict=math_data, keys=["pca.center", "pca.comment-projection", "base-clusters.x", "base-clusters.y", "group-clusters[*].center"])
    expected_pca = math_data["pca"]

    minimal_pca = {
        "center": expected_pca["center"],
        "comps": expected_pca["comps"],
    }

    calculated_pca = PcaUtils.with_proj_and_extremity(pca=minimal_pca)

    assert expected_pca["comment-projection"] == calculated_pca["comment-projection"]
    assert expected_pca["comment-extremity"] == calculated_pca["comment-extremity"]