import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from concave_hull import concave_hull_indexes
from typing import List, Optional
import numpy as np
import seaborn as sns
import pandas as pd

def generate_figure(
        coord_data,
        coord_labels,
        cluster_labels: Optional[List[int]] = None,
        flip_x: bool = False,
        flip_y: bool = False,
) -> None:
    """
    Generates a matplotlib scatterplot with optional bounded clusters.

    The plot is drawn from a dataframe of xy values, each point labelled by index `participant_id`.
    When a list of cluster labels are supplied (corresponding to each row), concave hulls are drawn around them.

    Signs of PCA projection coordinates are arbitrary, and can flip without
    meaning. Inverting axes can help compare results with Polis platform
    visualizations.

    Args:
        coord_dataframe (pd.DataFrame): A dataframe of coordinates with columns named `x` and `y`, indexed by `participant_id`.
        cluster_labels (List[int]): A list of group labels, one for each row in `coord_dataframe`.
        flip_x (bool): Flip the presentation of the X-axis so it descends left-to-right
        flip_y (bool): Flip the presentation of the Y-axis so it descends top-to-bottom

    Returns:
        None.
    """
    plt.figure(figsize=(7, 5), dpi=80)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    if flip_x:
        plt.gca().invert_xaxis()
    if flip_y:
        plt.gca().invert_yaxis()

    # Label points with participant_id
    for label, xy in zip(coord_labels, coord_data):
        plt.annotate(str(label),
            (float(xy[0]), float(xy[1])),
            xytext=(2, 2),
            color="gray",
            textcoords='offset points')

    scatter_kwargs = defaultdict()
    scatter_kwargs["x"] = coord_data[:, 0]
    scatter_kwargs["y"] = coord_data[:, 1]
    scatter_kwargs["s"] = 10       # point size
    scatter_kwargs["alpha"] = 0.8  # point transparency
    if cluster_labels is not None:
        # Ref: https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
        scatter_kwargs["cmap"] = "Set1"      # color map
        scatter_kwargs["c"] = cluster_labels # color indexes

        print("Calculating convex hulls around clusters...")
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            label_mask = cluster_labels == label
            cluster_points = coord_data[label_mask]
            print(f"Hull {str(label)}, bounding {len(cluster_points)} points")

            if len(cluster_points) < 3:
                # TODO: Accomodate 2 points like Polis platform does.
                print("Cannot create concave hull for less than 3 points. Skipping...")
                continue

            hull_point_indices = concave_hull_indexes(cluster_points, concavity=4.0)
            hull_points = cluster_points[hull_point_indices]

            polygon = patches.Polygon(
                hull_points,
                fill=True,
                color="gray",
                alpha=0.3,
                edgecolor=None,
            )
            plt.gca().add_patch(polygon)

    scatter = plt.scatter(**scatter_kwargs)

    # Add a legend if labels are provided
    if cluster_labels is not None:
        plt.colorbar(scatter, label="Cluster", ticks=cluster_labels)

    plt.show()

    return None

class DataPresenter():
    def __init__(self, client):
        self.client = client

    def render_optimal_cluster_figure(self):
        print(f"Optimal clusters for K={self.client.optimal_k}")
        print("Plotting PCA embeddings with K-means, K="+str(np.max(self.client.optimal_cluster_labels)+1))
        self.generate_figure(coord_dataframe=self.client.projected_data, cluster_labels=self.client.optimal_cluster_labels)

    def generate_figure(self, coord_dataframe, cluster_labels=None):
        coord_data = coord_dataframe.loc[:, ["x", "y"]].values
        coord_labels = coord_dataframe.index
        generate_figure(coord_data=coord_data, coord_labels=coord_labels, cluster_labels=cluster_labels)

    def generate_vote_heatmap(self, vote_df):
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_theme(font_scale=.7)
        sns.set_color_codes()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(vote_df, center=0, cmap="RdYlBu", ax=ax)
        plt.show()
