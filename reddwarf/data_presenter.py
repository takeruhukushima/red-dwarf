import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from concave_hull import concave_hull_indexes
import numpy as np

class DataPresenter():
    def __init__(self, client=None):
        self.client = client

    def render_optimal_cluster_figure(self):
        print(f"Optimal clusters for K={self.client.optimal_k}")
        print("Plotting PCA embeddings with K-means, K="+str(np.max(self.client.optimal_cluster_labels)+1))
        self.generate_figure(coord_dataframe=self.client.projected_data, labels=self.client.optimal_cluster_labels)

    def generate_figure(self, coord_dataframe, labels=None):
        plt.figure(figsize=(7, 5), dpi=80)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        plt.gca().invert_xaxis()

        # Label points with participant_id if no labels set.
        for participant_id, row in coord_dataframe.iterrows():
            plt.annotate(participant_id,
                (row["x"], row["y"]),
                xytext=(2, 2),
                color="gray",
                textcoords='offset points')

        scatter_kwargs = defaultdict()
        scatter_kwargs["x"] = coord_dataframe.loc[:,"x"]
        scatter_kwargs["y"] = coord_dataframe.loc[:,"y"]
        scatter_kwargs["s"] = 10       # point size
        scatter_kwargs["alpha"] = 0.8  # point transparency
        if labels is not None:
            # Ref: https://matplotlib.org/stable/users/explain/colors/colormaps.html#qualitative
            scatter_kwargs["cmap"] = "Set1"    # color map
            scatter_kwargs["c"] = labels        # color indexes

            print("Calculating convex hulls around clusters...")
            unique_labels = set(labels)
            for label in unique_labels:
                points = coord_dataframe[labels == label]
                print(f"Hull {str(label)}, bounding {len(points)} points")
                if len(points) < 3:
                    # TODO: Accomodate 2 points like Polis platform does.
                    print("Cannot create concave hull for less than 3 points. Skipping...")
                    continue
                vertex_indices = concave_hull_indexes(points, concavity=4.0)
                hull_points = points.iloc[vertex_indices, :]
                polygon = patches.Polygon(
                    hull_points,
                    fill=True,
                    color="gray",
                    alpha=0.3,
                    edgecolor=None,
                )
                plt.gca().add_patch(polygon)
        plt.scatter(**scatter_kwargs)
        plt.show()
