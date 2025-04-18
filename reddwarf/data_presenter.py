import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from concave_hull import concave_hull_indexes
from typing import List, Optional
from reddwarf.types.polis import PolisRepness
import numpy as np
import seaborn as sns

from reddwarf.implementations.polis import PolisClusteringResult

GROUP_LABEL_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H"]

def generate_figure_polis(result: PolisClusteringResult, show_guesses=False, flip_x=True, flip_y=False):
    cluster_labels = result.projected_participants["cluster_id"].values

    coord_data = result.projected_participants.loc[:, ["x", "y"]].values
    # Add the init center guesses to the bottom of the coord stack. Internally, they
    # will be give a fake "-1" colored label that won't be used to draw clusters.
    # This is for illustration purpose to see the centroid guesses.
    if show_guesses:
        coord_data = np.vstack([
            coord_data,
            np.asarray(result.kmeans.init_centers_used_ if result.kmeans else []),
        ])

    generate_figure(
        coord_data=coord_data,
        coord_labels=[f"p{pid}" for pid in result.projected_participants.index],
        cluster_labels=cluster_labels,
        # Always needs flipping to look like Polis interface.
        flip_x=flip_x,
        # Sometimes needs flipping to look like Polis interface.
        flip_y=flip_y,
    )

def generate_figure(
        coord_data,
        coord_labels = None,
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
        coord_data (pd.DataFrame): A dataframe of coordinates with columns named `x` and `y`, indexed by `participant_id`.
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

    # Label points when coordinate labels are provided.
    if coord_labels is not None:
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

    # Wrap clusters in hulls when cluster labels are provided.
    if cluster_labels is not None:
        # Ref: https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
        scatter_kwargs["cmap"] = "Set1"      # color map

        # Pad cluster_labels to match the number of points
        UNGROUPED_LABEL = -1
        if len(cluster_labels) < len(coord_data):
            pad_length = len(coord_data) - len(cluster_labels)
            cluster_labels = np.concatenate([cluster_labels, [UNGROUPED_LABEL] * pad_length])

        scatter_kwargs["c"] = cluster_labels # color indexes

        print("Calculating convex hulls around clusters...")
        # Subset to allow unlabelled points to just be plotted
        unique_labels = np.unique(cluster_labels)
        for label in unique_labels:
            if label == UNGROUPED_LABEL:
                continue # skip hulls when ungrouped label was padded in

            label_mask = cluster_labels == label
            cluster_points = coord_data[label_mask]

            print(f"Hull {label}, bounding {len(cluster_points)} points")

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
        cbar = plt.colorbar(scatter, label="Cluster", ticks=cluster_labels)

        UNGROUPED_LABEL_NAME = "[Center Guess]"
        tick_labels = [UNGROUPED_LABEL_NAME if lbl == -1 else GROUP_LABEL_NAMES[lbl] for lbl in cluster_labels]
        cbar.ax.set_yticklabels(tick_labels)

    plt.show()

    return None

def print_repness(
    repness: PolisRepness,
    statements_data: list[dict],
) -> None:
    """
    Helper function to format printed output of Polis repness object.

    Arguments:
        repness (dict): Repness dict that matches structure from polismath API
        statements_data (list[dict]): Statement data with keys `statement_id` and `txt`

    Returns:
        None
    """
    for gid, repful_statements in repness.items():
        gid = int(gid)
        group_label = GROUP_LABEL_NAMES[gid]
        print("GROUP {group_label}".format(group_label=group_label))

        for rep_st in repful_statements:
            st_data = [st for st in statements_data if st["statement_id"] == rep_st["tid"]][0]
            print(f"* {st_data['txt']}")
            if rep_st["repful-for"] == "agree":
                tmpl = "   {percent}% of those in group {group_label} who voted on statement {statement_id} agreed."
                print(tmpl.format(
                    group_label=group_label,
                    statement_id=rep_st["tid"],
                    percent=int((rep_st["n-success"]/rep_st["n-trials"])*100),
                ))
            else:
                tmpl = "   {percent}% of those in group {group_label} who voted on statement {statement_id} disagreed."
                print(tmpl.format(
                    percent=int((rep_st["n-success"]/rep_st["n-trials"])*100),
                    group_label=group_label,
                    statement_id=rep_st["tid"],
                ))
            print("")

        print("")

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
