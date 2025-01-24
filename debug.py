from reddwarf.polis_pandas import PolisClient

if True:
    client = PolisClient()
    # client.load_data(report_id="r8xhmkwp6shm9yfermteh")
    client.load_data(report_id="r2dfw8eambusb8buvecjt")
    filtered_matrix = client.get_matrix(is_filtered=True)
    client.impute_missing_votes()
    client.run_pca()
    print(len(client.projected_data))
    print(client.projected_data.values)
    client.generate_figure()

    # client.scale_pca_polis()
    # print(client.eigenvectors)
    # client.generate_figure()
