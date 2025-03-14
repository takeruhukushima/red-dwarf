from reddwarf.polis import PolisClient
from reddwarf.data_presenter import DataPresenter
import pandas as pd

from reddwarf import utils


CONVOS = {
    # Topic: What were the most significant developments in tech and politics in 2018?
    # 5 groups, 65 ptpts (56 grouped), 43 comments (open)
    "tech-politics-2018": {
        "report_id": "r2dfw8eambusb8buvecjt",
        "convo_id": "6jrufhr6dp",
    },
    # Topic: How should we operate vehicle-for-hire, e.g. Uber, Lyft and taxis in Toronto?
    # 2 groups, 47 ptpts (36 grouped), 69 comments (open)
    "rideshare-toronto": {
        "report_id": "r8xhmkwp6shm9yfermteh",
        "convo_id": "7vampckwrh",
    },
    # Topic: Help us pick rules for our AI chatbot! 7/7
    # 2 groups, 1_127 ptpts (1_094 grouped), 1_418 comments (closed)
    "anthropic-ccai": {
        "report_id": "r3rwrinr5udrzwkvxtdkj",
        "convo_id": "3akt5cdsfk",
    },
    # Topic: How should we use open source tools in governmment?
    # Test convo using xids and avatar images.
    "xid-testing": {
        "convo_id": "4kjz5rrrfe",
    },
}

if True:
    # testing representativeness calculations
    report_id = CONVOS["tech-politics-2018"]["report_id"]
    print(f"Loading data from report: https://pol.is/report/{report_id}")

    client = PolisClient()
    client.load_data(report_id=report_id)

    USE_POLISMATH_CLUSTERING = True
    if USE_POLISMATH_CLUSTERING:
        math_data = client.data_loader.math_data
        group_clusters_with_pids = utils.expand_group_clusters_with_participants(
            group_clusters=math_data["group-clusters"],
            base_clusters=math_data["base-clusters"],
        )
        # Get list of all active participant ids, since Polis has some edge-cases
        # that keep specific participants, and we need to keep them from being filtered out.
        all_participant_ids = utils.get_all_participant_ids(group_clusters_with_pids)
        client.keep_participant_ids = all_participant_ids

    # Generate vote matrix and run clustering
    vote_matrix = client.get_matrix(is_filtered=True)
    client.run_pca()
    client.scale_projected_data()

    if USE_POLISMATH_CLUSTERING:
        cluster_labels = utils.generate_cluster_labels(group_clusters_with_pids)
        client.optimal_cluster_labels = cluster_labels
    else:
        client.find_optimal_k()  # Find optimal number of clusters
        cluster_labels = client.optimal_cluster_labels

    stats_by_group = utils.calculate_comment_statistics_by_group(
        vote_matrix=vote_matrix,
        cluster_labels=cluster_labels,
    )

    # Figuring out select-rep-comments flow
    # See: https://github.com/compdemocracy/polis/blob/7bf9eccc287586e51d96fdf519ae6da98e0f4a70/math/src/polismath/math/repness.clj#L209C7-L209C26
    polis_repness = {}
    for gid, stats_df in enumerate(stats_by_group):
        print(f"Group ID: {gid}")

        group_data = {}
        group_data["sufficient"] = []
        def create_filter_mask(row):
            return utils.is_passes_by_test(row["pat"], row["rat"], row["pdt"], row["rdt"])
        group_data["sufficient"] = stats_df[stats_df.apply(create_filter_mask, axis=1)]
        group_data["sufficient"] = pd.DataFrame([
            utils.finalize_cmt_stats(row)
            for _, row in group_data["sufficient"].reset_index().iterrows()
        ], index=group_data["sufficient"].index)
        if len(group_data["sufficient"]) > 0:
            repness_metric = lambda row: row["repness"] * row["repness-test"] * row["p-success"] * row["p-test"]
            group_data["sufficient"] = group_data["sufficient"].assign(sort_order=repness_metric).sort_values(by="sort_order", ascending=False)

        # Track the best, even if doesn't meet sufficient minimum, to have at least one.
        # TODO: Merge this wil above iteration
        if len(group_data["sufficient"]) == 0:
            group_data["best"] = {}
            for _, row in stats_df.reset_index().iterrows():
                if utils.beats_best_by_test(row["rat"], row["rdt"], group_data["best"].get("repness-test", None)):
                    group_data["best"] = utils.finalize_cmt_stats(row)

        # Track the best-agree, to bring to top if exists.
        group_data["best-agree"] = None
        for _, row in stats_df.reset_index().iterrows():
            if utils.beats_best_agr(row["na"], row["nd"], row["ra"], row["rat"], row["pa"], row["pat"], row["ns"], group_data["best-agree"]):
                group_data["best-agree"] = row

        # Start building repness key
        best_agree = group_data.get("best-agree")
        best = group_data.get("best")
        if best_agree is not None:
            best_agree = utils.finalize_cmt_stats(best_agree)
            best_agree.update({"n-agree": best_agree["n-success"], "best-agree": True})
            best_head = [best_agree]
        elif best is not None:
            best_head = [best]
        else:
            best_head = []

        if len(group_data["sufficient"]) > 0:
            group_data["sufficient"] = group_data["sufficient"].drop(columns="sort_order").astype({"tid": "int32", "n-success": "int32", "n-trials": "int32"})
        if len(best_head) > 0:
            selected = best_head + [dict(row) for _, row in group_data["sufficient"].iterrows() if row["tid"] != best_head[0]["tid"]]
        else:
            selected = [dict(row) for _, row in group_data["sufficient"].iterrows()]
        # sorted() does the work of agrees-before-disagrees in polismath
        polis_repness[str(gid)] = sorted(selected[:5], key=lambda x: x["repful-for"])
    import json
    print(json.dumps(polis_repness, indent=2))

    presenter = DataPresenter(client=client)
    presenter.render_optimal_cluster_figure()

if False:
    # test agora method
    from reddwarf.agora import run_clustering
    from reddwarf.types.agora import Conversation

    report_id = CONVOS["tech-politics-2018"]["report_id"]
    print(f"Loading data from report: https://pol.is/report/{report_id}")

    client = PolisClient()
    client.load_data(report_id=report_id)

    convo: Conversation = {
        "id": "demo",
        "votes": client.data_loader.votes_data,
    }
    results = run_clustering(conversation=convo)

    from pprint import pprint
    pprint(results)

if False:
    # Render a figure with best hulls.
    report_id = CONVOS["tech-politics-2018"]["report_id"]
    print(f"Loading data from report: https://pol.is/report/{report_id}")

    client = PolisClient()
    client.load_data(report_id=report_id)
    # To see the consequences of not having pass/neutral/zero votes
    DO_STRIP_PASS=False
    if DO_STRIP_PASS:
        client.votes = []
        client.load_votes_data(data=[v for v in client.data_loader.votes_data if v["vote"] != 0])
    client.get_matrix(is_filtered=True)
    client.run_pca()
    client.scale_projected_data()
    client.find_optimal_k()

    presenter = DataPresenter(client=client)
    presenter.render_optimal_cluster_figure()
    # client.generate_figure(coord_dataframe=client.projected_data)

if False:
    # Show convo with duplicate votes.
    # Shareable demo: https://gist.github.com/patcon/9c1a39291cd75b23722a5379d7cfc3cc
    report_id = CONVOS["tech-politics-2018"]["report_id"]
    print(f"Loading data from report: https://pol.is/report/{report_id}")

    client = PolisClient(is_strict_moderation=False)
    client.load_data(report_id=report_id, data_source="csv_export")
    client.get_matrix(is_filtered=True)

if False:
    client = PolisClient()
    client.load_data(conversation_id="9xxwa9jpkm")
    # Reproducing this output: https://github.com/compdemocracy/openData/blob/master/london.youth.policing
    # participant-votes.csv matches, but votes.csv is missing entries.
    # BUG: dates for exports seemingly not matching between matrix build and vote export.
    matrix_raw = client.get_matrix(is_filtered=False, cutoff=1658934741418)
    participants_votes_df = client.build_participants_dataframe(matrix_raw)
    print(participants_votes_df)

# This is a sanity-check for equality of a participants-votes.csv dataframe generated from raw votes vs a downloaded export.
if False:
    # Generate the participants-votes dataframe from raw data
    client = PolisClient(is_strict_moderation=True)
    client.load_data(directory_url="https://raw.githubusercontent.com/compdemocracy/openData/refs/heads/master/london.youth.policing/")
    raw_matrix = client.get_matrix(is_filtered=False)
    # Drop statement columns if no votes.
    raw_matrix = raw_matrix.dropna(axis="columns", how="all")
    # Convert all int columns to strings for easier comparison.
    raw_matrix.columns = raw_matrix.columns.astype(str)
    participants_votes_generated = client.build_participants_dataframe(vote_matrix=raw_matrix)
    participants_votes_generated = participants_votes_generated.join(raw_matrix)
    # Remove rows for participants with zero votes.
    non_voting_participant_ids = participants_votes_generated[participants_votes_generated["n_votes"] == 0].index
    participants_votes_generated = participants_votes_generated.drop(index=non_voting_participant_ids)

    # Generate dataframe from downloaded CSV.
    col_mapper = {
        "participant": "participant_id",
        "group-id": "group_id",
        "n-comments": "n_comments",
        "n-votes": "n_votes",
        "n-agree": "n_agree",
        "n-disagree": "n_disagree",
    }
    participants_votes_downloaded = (
        pd
            .read_csv("https://raw.githubusercontent.com/compdemocracy/openData/refs/heads/master/london.youth.policing/participants-votes.csv")
            .rename(columns=col_mapper)
            # Override group_id for now, until we can generate and compare.
            .assign(group_id=None)
            .set_index("participant_id")
            .sort_index()
    )
    is_dataframes_equal = participants_votes_downloaded.equals(participants_votes_generated)
    print(f"downloaded and generated dataframes are equal? {is_dataframes_equal}")
    print(participants_votes_downloaded.shape)
    print(participants_votes_generated.shape)
    print(participants_votes_downloaded.compare(participants_votes_generated))

if False:
    client = PolisClient()
    # client.load_data(report_id=CONVOS["rideshare-toronto"]["report_id"])
    client.load_data(report_id=CONVOS["tech-politics-2018"]["report_id"])
    # client.load_data(conversation_id=CONVOS["anthropic-ccai"]["convo_id"])
    matrix_raw = client.get_matrix(is_filtered=False)
    client.matrix = None # Flush matrix
    matrix_filtered = client.get_matrix(is_filtered=True)
    matrix_filtered_imputed = client.impute_missing_votes()

    presenter = DataPresenter()
    # presenter.generate_vote_heatmap(matrix_raw)
    presenter.generate_vote_heatmap(matrix_filtered)
    # presenter.generate_vote_heatmap(matrix_filtered_imputed)


if False:
    client = PolisClient()
    client.load_data(conversation_id=CONVOS["xid-testing"]["convo_id"])
    xids = [12334552, 12334553, 12334554, "foobar"]
    mappings = client.data_loader.fetch_xid_to_pid_mappings(xids)
    for xid, pid in mappings.items():
        print(f"{pid=} => {xid=}")
