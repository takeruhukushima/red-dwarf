from reddwarf.implementations import base


# This is to not break things.
# TODO: Adde deprecation warning.
def run_clustering(**kwargs) -> base.PolisClusteringResult:
    return run_pipeline(**kwargs)


def run_pipeline(**kwargs) -> base.PolisClusteringResult:
    return base.run_pipeline(reducer="pca", clusterer="kmeans", **kwargs)
