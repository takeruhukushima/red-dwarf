from reddwarf.polis_pandas import PolisClient
from reddwarf.data_presenter import DataPresenter

if False:
    client = PolisClient()
    # client.load_data(report_id="r8xhmkwp6shm9yfermteh")
    client.load_data(report_id="r2dfw8eambusb8buvecjt")
    client.get_matrix(is_filtered=True)
    client.run_pca()
    client.scale_projected_data()
    client.find_optimal_k()

    presenter = DataPresenter(client=client)
    presenter.render_optimal_cluster_figure()
    # client.generate_figure(coord_dataframe=client.projected_data)

if True:
    client = PolisClient()
    # client.load_data(report_id="r8xhmkwp6shm9yfermteh")
    client.load_data(report_id="r2dfw8eambusb8buvecjt")
    matrix_raw = client.get_matrix(is_filtered=False)
    client.matrix = None
    matrix_filtered = client.get_matrix(is_filtered=True)
    matrix_filtered_imputed = client.impute_missing_votes()

    presenter = DataPresenter()
    # presenter.generate_vote_heatmap(matrix_raw)
    presenter.generate_vote_heatmap(matrix_filtered)
    # presenter.generate_vote_heatmap(matrix_filtered_imputed)


if False:
    client = PolisClient()
    client.load_data(conversation_id="4kjz5rrrfe")
    xids = [12334552, 12334553, 12334554, "foobar"]
    mappings = client.data_loader.fetch_xid_to_pid_mappings(xids)
    for xid, pid in mappings.items():
        print(f"{pid=} => {xid=}")
