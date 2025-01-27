from reddwarf.polis_pandas import PolisClient
from reddwarf.data_presenter import DataPresenter


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

if False:
    client = PolisClient()
    # client.load_data(report_id=CONVOS["rideshare-toronto"]["report_id"])
    client.load_data(report_id=CONVOS["tech-politics-2018"]["report_id"])
    client.get_matrix(is_filtered=True)
    client.run_pca()
    client.scale_projected_data()
    client.find_optimal_k()

    presenter = DataPresenter(client=client)
    presenter.render_optimal_cluster_figure()
    # client.generate_figure(coord_dataframe=client.projected_data)

if True:
    client = PolisClient()
    client.load_data(conversation_id="9xxwa9jpkm")
    # Reproducing this output: https://github.com/compdemocracy/openData/blob/master/london.youth.policing
    # participant-votes.csv matches, but votes.csv is missing entries.
    # BUG: dates for exports seemingly not matching between matrix build and vote export.
    matrix_raw = client.get_matrix(is_filtered=False, cutoff=1658934741418)
    participants_votes_df = client.build_participants_dataframe(matrix_raw)
    print(participants_votes_df)

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
