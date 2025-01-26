from reddwarf.polis_pandas import PolisClient
from reddwarf.data_presenter import DataPresenter

if True:
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
