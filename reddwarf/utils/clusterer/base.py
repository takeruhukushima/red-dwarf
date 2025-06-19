from typing import Optional, Union, TypeAlias, TYPE_CHECKING

from numpy.typing import NDArray


from reddwarf.sklearn.cluster import PolisKMeans
from reddwarf.utils.clusterer.kmeans import find_best_kmeans

if TYPE_CHECKING:
    from sklearn.cluster import HDBSCAN
    from reddwarf.sklearn.cluster import PolisKMeans

ReducerModel: TypeAlias = Union["HDBSCAN", "PolisKMeans"]


def run_clusterer(
    X_participants_clusterable: NDArray,
    clusterer="kmeans",
    force_group_count=None,
    max_group_count=5,
    init_centers=None,
    **clusterer_kwargs,
) -> Optional[ReducerModel]:
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
                init_centers=init_centers,
                # TODO: Support passing in arbitrary clusterer_kwargs.
            )

            return kmeans

        case "hdbscan":
            from sklearn.cluster import HDBSCAN

            hdb = HDBSCAN(**clusterer_kwargs)
            hdb.fit(X_participants_clusterable)

            return hdb
        case _:
            raise NotImplementedError("clusterer type unknown")
