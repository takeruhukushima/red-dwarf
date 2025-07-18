import pytest
import numpy as np
from reddwarf.utils.clusterer.base import run_clusterer
from tests.fixtures import polis_convo_data
from hdbscan import HDBSCAN

@pytest.mark.parametrize("polis_convo_data", ["medium"], indirect=True)
def test_run_clusterer_hdbscan(polis_convo_data):
    fixture = polis_convo_data

    X_math_data = np.asarray([
        fixture.math_data["base-clusters"]["x"],
        fixture.math_data["base-clusters"]["y"],
    ]).transpose()

    clusterer_model = run_clusterer(
        X_participants_clusterable=X_math_data,
        clusterer="hdbscan",
        random_state=42,
    )

    assert isinstance(clusterer_model, HDBSCAN)
    assert hasattr(clusterer_model, "labels_")

    n_samples = X_math_data.shape[0]
    assert len(clusterer_model.labels_) == n_samples

    # Ensure at least some are clustered.
    label_counts = dict(zip(*np.unique(clusterer_model.labels_, return_counts=True)))
    assert label_counts[-1] < len(clusterer_model.labels_)

@pytest.mark.parametrize("polis_convo_data", ["small"], indirect=True)
def test_run_clusterer_unknown(polis_convo_data):
    fixture = polis_convo_data

    X_math_data = np.asarray([
        fixture.math_data["base-clusters"]["x"],
        fixture.math_data["base-clusters"]["y"],
    ]).transpose()

    with pytest.raises(NotImplementedError):
        clusterer_model = run_clusterer(
            X_participants_clusterable=X_math_data,
            clusterer="unknown",
            random_state=42,
        )