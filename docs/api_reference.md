# API Reference

## `reddwarf.implementations.polis`

### ::: reddwarf.implementations.polis.run_clustering

    options:
        show_root_heading: true

## `reddwarf.implementations.agora`

### ::: reddwarf.implementations.agora.run_clustering_v1

    options:
        show_root_heading: true

## `reddwarf.sklearn`

Various custom Scikit-Learn estimators to mimick aspects of Polis, suitable for
use in Scikit-Learn workflows, pipelines, and APIs.

### ::: reddwarf.sklearn.cluster.PolisKMeans

    options:
        show_root_heading: true
        docstring_style: numpy

### ::: reddwarf.sklearn.cluster.PolisKMeansDownsampler

    options:
        show_root_heading: true

### ::: reddwarf.sklearn.model_selection.GridSearchNonCV

    options:
        show_root_heading: true

### ::: reddwarf.sklearn.transformers.SparsityAwareScaler

    options:
        show_root_heading: true

## `reddwarf.utils.matrix`

### ::: reddwarf.utils.matrix.generate_raw_matrix

    options:
        show_root_heading: true

### ::: reddwarf.utils.matrix.simple_filter_matrix

    options:
        show_root_heading: true

### ::: reddwarf.utils.matrix.get_clusterable_participant_ids

    options:
        show_root_heading: true

## `reddwarf.utils.pca`

### ::: reddwarf.utils.pca.run_reducer

    options:
        show_root_heading: true

## `reddwarf.utils.clustering`

### ::: reddwarf.utils.clustering.find_optimal_k

    options:
        show_root_heading: true

### ::: reddwarf.utils.clustering.run_kmeans

    options:
        show_root_heading: true

## `reddwarf.utils.consensus`

### ::: reddwarf.utils.consensus.select_consensus_statements

    options:
        show_root_heading: true

## `reddwarf.utils.stats`

### ::: reddwarf.utils.stats.select_representative_statements

    options:
        show_root_heading: true

### ::: reddwarf.utils.stats.calculate_comment_statistics

    options:
        show_root_heading: true

### ::: reddwarf.utils.stats.calculate_comment_statistics_dataframes

    options:
        show_root_heading: true

## `reddwarf.utils`

(These are in the process of being either moved or deprecated.)

### ::: reddwarf.utils.filter_votes

    options:
        show_root_heading: true

### ::: reddwarf.utils.filter_matrix

    options:
        show_root_heading: true

### ::: reddwarf.utils.get_unvoted_statement_ids

    options:
        show_root_heading: true

## `reddwarf.data_presenter`

### ::: reddwarf.data_presenter.generate_figure

    options:
        show_root_heading: true

## Types

### ::: reddwarf.implementations.polis.PolisClusteringResult

    options:
        show_root_heading: true

### ::: reddwarf.types.agora.Conversation

    options:
        show_root_heading: true

### ::: reddwarf.types.agora.Vote

    options:
        show_root_heading: true

### ::: reddwarf.types.agora.VoteValueEnum

    options:
        show_root_heading: true

### ::: reddwarf.types.agora.Identifier

    options:
        show_root_heading: true

### ::: reddwarf.types.agora.ClusteringOptions

    options:
        show_root_heading: true

### ::: reddwarf.types.agora.ClusteringResult

    options:
        show_root_heading: true

### ::: reddwarf.types.agora.Cluster

    options:
        show_root_heading: true

### ::: reddwarf.types.agora.ClusteredParticipant

    options:
        show_root_heading: true
