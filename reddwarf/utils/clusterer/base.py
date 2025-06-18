from typing import Optional

from numpy.typing import NDArray

from reddwarf.sklearn.cluster import PolisKMeans
from reddwarf.utils.clusterer.kmeans import find_best_kmeans


def run_clusterer(
    X_participants_clusterable: NDArray,
    clusterer="kmeans",
    force_group_count=None,
    max_group_count=5,
    **clusterer_kwargs,
) -> Optional[PolisKMeans]:
    match clusterer:
        case "kmeans":
            if force_group_count:
                k_bounds = [force_group_count, force_group_count]
            else:
                k_bounds = [2, max_group_count]

            _, _, kmeans = find_best_kmeans(
                X_to_cluster=X_participants_clusterable,
                k_bounds=k_bounds,
                # Force polis strategy of initiating cluster centers. See: PolisKMeans.
                init="polis",
                **clusterer_kwargs,
            )

            return kmeans

        case "hdbscan":
            raise NotImplementedError("clusterer type hdbscan not yet implemented")
        case _:
            raise NotImplementedError("clusterer type unknown")
