import pandas as pd
import numpy as np
from reddwarf.sklearn.model_selection import GridSearchNonCV
from reddwarf.sklearn.cluster import PolisKMeans
from sklearn.metrics import silhouette_score
from typing import List, Tuple, Optional

# TODO: Start passing init_centers based on /math/pca2 endpoint data,
# and see how often we get the same clusters.
def run_kmeans(
        dataframe: pd.DataFrame,
        n_clusters: int = 2,
        init="k-means++",
        # TODO: Improve this type. 3d?
        init_centers: Optional[List] = None,
        random_state: Optional[int] = None,
) -> PolisKMeans:
    """
    Runs K-Means clustering on a 2D DataFrame of xy points, for a specific K,
    and returns labels for each row and cluster centers. Optionally accepts
    guesses on cluster centers, and a random_state to reproducibility.

    Args:
        dataframe (pd.DataFrame): A dataframe with two columns (assumed `x` and `y`).
        n_clusters (int): How many clusters k to assume.
        init (string): The cluster initialization strategy. See `PolisKMeans` docs.
        init_centers (List): A list of xy coordinates to use as initial center guesses. See `PolisKMeans` docs.
        random_state (int): Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.

    Returns:
        kmeans (PolisKMeans): The estimator object returned from PolisKMeans.
    """
    # TODO: Set random_state to a value eventually, so calculation is deterministic.
    kmeans = PolisKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        init=init,
        init_centers=init_centers,
    ).fit(dataframe)

    return kmeans

def find_optimal_k(
        projected_data: pd.DataFrame,
        max_group_count: int = 5,
        init="k-means++",
        init_centers: Optional[List] = None,
        random_state: Optional[int] = None,
        debug: bool = False,
) -> Tuple[int, float, PolisKMeans | None]:
    """
    Use silhouette scores to find the best number of clusters k to assume to fit the data.

    Args:
        projected_data (pd.DataFrame): A dataframe with two columns (assumed `x` and `y`).
        max_group_count (int): The max K number of groups to test for. (Default: 5)
        init_centers (List): A list of xy coordinates to use as initial center guesses.
        random_state (int): Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
        debug (bool): Whether to print debug output. (Default: False)

    Returns:
        optimal_k (int): Ideal number of clusters.
        optimal_silhouette_score (float): Silhouette score for this K value.
        optimal_kmeans (PolisKMeans | None): The optimal fitted estimator returned from PolisKMeans.
    """
    param_grid = {"n_clusters": list(range(2, max_group_count+1))}

    search = GridSearchNonCV(
        estimator=PolisKMeans(
            random_state=random_state,
            init=init,
            init_centers=init_centers,
        ),
        param_grid=param_grid,
        scoring=lambda estimator, X: silhouette_score(X, estimator.fit_predict(X)),
    )

    search.fit(projected_data)
    best_k = search.best_params_['n_clusters']
    best_silhouette_score = search.best_score_
    best_kmeans = search.best_estimator_

    return best_k, best_silhouette_score, best_kmeans

def init_clusters(data: np.ndarray, k: int):
    """
    Initializes k cluster centers from distinct rows in a 2D NumPy array.
    Each row is assumed to be an [x, y] coordinate.

    Ref: https://github.com/compdemocracy/polis/blob/61b8a18b962d0278e70bb92960e643ea937a4d6a/math/src/polismath/math/clusters.clj#L55-L65
    """
    # Remove duplicate rows
    unique_rows = np.unique(data, axis=0)

    # Take the first k unique rows as cluster centers
    centers = unique_rows[:k]

    return centers