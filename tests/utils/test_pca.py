import pandas as pd
from numpy.testing import assert_allclose, assert_array_almost_equal
import pytest
from reddwarf.utils.statements import process_statements
from tests.fixtures import polis_convo_data
from reddwarf.utils.reducer.pca import run_pca
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
    actual_projected_participants, _, pca = run_pca(vote_matrix=initial_matrix)
    actual_eigenvectors = pca.components_
    actual_eigenvalues = pca.explained_variance_
    assert_allclose(actual_projected_participants, expected_projections, rtol=10**-5)
    assert_allclose(actual_eigenvectors, expected_eigenvectors, rtol=10**-5)
    assert_allclose(actual_eigenvalues, expected_eigenvalues, rtol=10**-5)

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta", "medium-no-meta", "medium-with-meta"], indirect=True)
def test_matching_explained_variance(polis_convo_data):
    """
    This is the most robust way to comfirm our PCA calculations are working,
    since variance isn't affect by arbitrary but expected sign-flips of
    components, nor potential unexpected sign-flip bugs in polismath from
    historic agree/disagree behavior changes.
    """
    fixture = polis_convo_data

    loader = Loader(filepaths=[
        f"{fixture.data_dir}/votes.json",
        f"{fixture.data_dir}/comments.json",
        f"{fixture.data_dir}/math-pca2.json",
    ])

    expected_pca = loader.math_data["pca"]

    # Generate PCA object and explained variance from scratch.
    _, _, mod_out_statement_ids, _ = process_statements(loader.comments_data)
    real_vote_matrix = MatrixUtils.generate_raw_matrix(votes=loader.votes_data)
    real_vote_matrix = MatrixUtils.simple_filter_matrix(
        vote_matrix=real_vote_matrix,
        mod_out_statement_ids=mod_out_statement_ids,
    )
    _, _, actual_pca = run_pca(vote_matrix=real_vote_matrix)
    actual_variance = actual_pca.explained_variance_

    # Rebuild explained variance from polismath data.
    expected_variance = helpers.calculate_explained_variance(
        sparse_vote_matrix=real_vote_matrix.values,
        means=expected_pca["center"],
        components=expected_pca["comps"],
    )

    assert actual_variance == pytest.approx(expected_variance)

@pytest.mark.parametrize("polis_convo_data", ["small-with-mod-out-votes"], indirect=True)
def test_explained_variance_modification_fails(polis_convo_data):
    fixture = polis_convo_data

    loader = Loader(filepaths=[
        f"{fixture.data_dir}/votes.json",
        f"{fixture.data_dir}/comments.json",
        f"{fixture.data_dir}/math-pca2.json",
    ])

    expected_pca = loader.math_data["pca"]

    # Generate PCA object and explained variance from scratch.
    _, _, mod_out_statement_ids, _ = process_statements(loader.comments_data)

    # HERE: We pop off one mod-out statement, so it stays in during our
    # calculation. We happen to know that the final statement ID we pop off has
    # votes. (It wouldn't matter if it didn't.)
    assert len(mod_out_statement_ids) > 0
    mod_out_statement_ids.pop()

    real_vote_matrix = MatrixUtils.generate_raw_matrix(votes=loader.votes_data)
    real_vote_matrix = MatrixUtils.simple_filter_matrix(
        vote_matrix=real_vote_matrix,
        mod_out_statement_ids=mod_out_statement_ids,
    )
    _, _, actual_pca = run_pca(vote_matrix=real_vote_matrix)
    actual_variance = actual_pca.explained_variance_

    # Rebuild explained variance from polismath data.
    expected_variance = helpers.calculate_explained_variance(
        sparse_vote_matrix=real_vote_matrix.values,
        means=expected_pca["center"],
        components=expected_pca["comps"],
    )

    assert actual_variance != pytest.approx(expected_variance)

@pytest.mark.parametrize("polis_convo_data", ["small-no-meta", "small-with-meta"], indirect=True)
def test_run_pca_real_data_below_100(polis_convo_data):
    fixture = polis_convo_data
    math_data = helpers.flip_signs_by_key(nested_dict=fixture.math_data, keys=["pca.center", "pca.comment-projection", "base-clusters.x", "base-clusters.y", "group-clusters[*].center"])
    expected_pca = math_data["pca"]

    # Just fetch the moderated out statements from polismath (no need to recalculate here)
    mod_out_statement_ids = math_data["mod-out"]

    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    real_vote_matrix = MatrixUtils.generate_raw_matrix(votes=loader.votes_data)

    real_vote_matrix = MatrixUtils.simple_filter_matrix(
        vote_matrix=real_vote_matrix,
        mod_out_statement_ids=mod_out_statement_ids,
    )

    X_participants, X_statements, actual_pca = run_pca(vote_matrix=real_vote_matrix)

    assert actual_pca.components_[0] == pytest.approx(expected_pca["comps"][0])
    assert actual_pca.components_[1] == pytest.approx(expected_pca["comps"][1])
    assert actual_pca.mean_ == pytest.approx(expected_pca["center"])

    expected_participant_projections = helpers.transform_base_clusters_to_participant_coords(base_clusters=math_data["base-clusters"])
    expected_df = pd.DataFrame.from_records(expected_participant_projections)
    expected_df[['x', 'y']] = pd.DataFrame(expected_df['xy'].tolist(), index=expected_df.index)
    expected_df = expected_df.drop(columns=['xy']).set_index('participant_id').sort_index()

    # Only include participants that are clustered (aka "in conversation" calculation)
    actual_df = pd.DataFrame(X_participants, index=real_vote_matrix.index, columns=pd.Index(["x", "y"]))
    actual_df = actual_df.loc[math_data["in-conv"]].sort_index()

    pd.testing.assert_frame_equal(expected_df, actual_df, rtol=1e-5)

    assert_array_almost_equal(X_statements.transpose(), expected_pca["comment-projection"])

# TODO: Accomodate sign flips in "medium-no-meta".
@pytest.mark.parametrize("polis_convo_data", ["medium-with-meta"], indirect=True)
def test_run_pca_real_data_above_100(polis_convo_data):
    fixture = polis_convo_data
    # Some signs are flipped for the "medium-with-meta" fixture data, because signs are arbitrary in PCA.
    # If we initialize differently later on, it should flip and match.
    math_data = helpers.flip_signs_by_key(nested_dict=fixture.math_data, keys=["pca.comps[0]", "pca.center"])
    expected_pca = math_data["pca"]

    # Just fetch the moderated out statements from polismath (no need to recalculate here)
    mod_out_statement_ids = math_data["mod-out"]

    loader = Loader(filepaths=[f"{fixture.data_dir}/votes.json"])
    real_vote_matrix = MatrixUtils.generate_raw_matrix(votes=loader.votes_data)

    real_vote_matrix = MatrixUtils.simple_filter_matrix(
        vote_matrix=real_vote_matrix,
        mod_out_statement_ids=mod_out_statement_ids,
    )

    _, _, actual_pca = run_pca(vote_matrix=real_vote_matrix)

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

    _, _, actual_pca = run_pca(vote_matrix=real_vote_matrix)
    # powerit_pca = PcaUtils.powerit_pca(real_vote_matrix.values)

    # We test absolute because PCA methods don't always give the same sign, and can flip.
    # TODO: Try to remove this.
    assert actual_pca.components_[0] == pytest.approx(expected_pca["comps"][0])
    assert actual_pca.components_[1] == pytest.approx(expected_pca["comps"][1])

    assert actual_pca.mean_ == pytest.approx(expected_pca["center"])