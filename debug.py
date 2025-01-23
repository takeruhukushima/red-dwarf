from reddwarf.polis_pandas import PolisClient

if True:
    client = PolisClient()
    client.load_data(report_id="r8xhmkwp6shm9yfermteh")
    client.get_matrix(is_filtered=False)
    client.impute_missing_votes()
    client.run_pca()
    client.scale_pca_polis()
    client.generate_figure()
