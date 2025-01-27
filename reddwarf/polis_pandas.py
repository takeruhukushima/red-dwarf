import pandas as pl # For sake of clean diffs
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from reddwarf.data_loader import Loader
from reddwarf.models import ModeratedEnum

class PolisClient():
    def __init__(self, is_strict_moderation=None) -> None:
        self.data_loader = None
        self.n_components = 2
        # Ref: https://hyp.is/8zUyWM5fEe-uIO-J34vbkg/gwern.net/doc/sociology/2021-small.pdf
        self.min_votes = 7
        self.votes = []
        self.statements_df = None
        self.participants_df = None
        # We sometimes want to keep some participant IDs that would otherwise be removed
        # (e.g., for not meeting vote threshold), to reproduce bugs in Polis codebase algorithms.
        self.keep_participant_ids = []
        # Ref: https://hyp.is/MV0Iws5fEe-k9BdY6UR1VQ/gwern.net/doc/sociology/2021-small.pdf
        self.matrix = None
        # Ref: https://gist.github.com/patcon/fd9079a5fbcd533160f8ae211e975307#file-math-pca2-json-L2
        # Ref: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L217-L225
        self.user_vote_counts = defaultdict(int)
        self.meta_tids = []
        self.mod_in = []
        self.mod_out = []
        self.last_vote_timestamp = 0
        self.group_clusters = []
        self.base_clusters = {}
        # TODO: Make accessor methods for these?
        self.statement_count = None
        self.participant_count = None
        self.is_strict_moderation = is_strict_moderation
        # Also know as PCA coords, PCA components, or embeddings.
        self.eigenvectors = []
        # Also know as PCA explained variance.
        self.eigenvalues = []
        self.optimal_k = None
        self.optimal_cluster_labels = None
        self.optimal_silhouette = None

    def get_participant_row_mask(self):
        raise NotImplementedError

    def get_is_strict_moderation(self):
        if self.is_strict_moderation == None:
            raise ValueError('is_strict_moderation cannot be auto-detected, and must be set manually')

        return self.is_strict_moderation

    def get_active_statement_ids(self):
        active_statements = self.statements_df[self.statements_df["is_shown"]]
        active_statement_ids = active_statements.index
        return active_statement_ids

    def get_unvoted_statement_ids(self):
        null_column_mask = self.matrix.isnull().all()
        null_column_ids = self.matrix.columns[null_column_mask].tolist()

        return null_column_ids

    def apply_masks(self, participant_rows=True, statement_cols=True):
        raise NotImplementedError

    def add_votes_batch(self, votes_df):
        """Add multiple votes from a DataFrame"""
        # TODO: Use tuples instead of named columns later, for perf.
        for _, row in votes_df.iterrows():
            self.add_vote(row)

    def add_vote(self, vote_row):
        """Add a single vote to the system"""
        # If this is a new vote (not an update)
        self.user_vote_counts[vote_row['participant_id']] += 1

        if self.last_vote_timestamp < int(vote_row['modified']):
            self.last_vote_timestamp = int(vote_row['modified'])

        self.votes.append({
            'participant_id': vote_row['participant_id'],
            'statement_id': vote_row['statement_id'],
            'vote': vote_row['vote'],
            'modified': vote_row['modified'],
        })
        # Matrix is now stale
        self.matrix = None
        self.statement_count = None
        self.participant_count = None

    def get_user_vote_counts(self):
        return self.user_vote_counts

    def get_meta_tids(self):
        return self.meta_tids

    def get_mod_in(self):
        return self.mod_in

    def get_mod_out(self):
        return self.mod_out

    def get_last_vote_timestamp(self):
        return self.last_vote_timestamp

    def get_group_clusters(self):
        return self.group_clusters

    def generate_raw_matrix(self, cutoff=None):
        if cutoff:
            # TODO: This should already be sorted earlier. confirm.
            date_sorted_votes = sorted([v for  v in self.votes], key=lambda x: x['modified'])
            # date_sorted_votes = self.votes
            if cutoff > 1_300_000_000:
                cutoff_timestamp = cutoff
                votes = [v for v in date_sorted_votes if v['modified'] <= cutoff_timestamp]
            else:
                cutoff_index = cutoff
                votes = date_sorted_votes[:cutoff_index]
        else:
            votes = self.votes

        raw_matrix = pl.DataFrame.from_dict(votes)
        raw_matrix = raw_matrix.pivot(
            values="vote",
            index="participant_id",
            columns="statement_id",
        )

        participant_count = raw_matrix.index.max() + 1
        comment_count = raw_matrix.columns.max() + 1
        raw_matrix = raw_matrix.reindex(
            index=range(participant_count),
            columns=range(comment_count),
            fill_value=np.nan,
        )

        return raw_matrix

    def get_matrix(self, is_filtered=False, cutoff=None):
        # Only generate matrix when needed.
        if self.matrix is None:
            raw_matrix = self.generate_raw_matrix(cutoff=cutoff)
            self.statement_count = raw_matrix.notna().any().sum()
            self.participant_count = len(raw_matrix)

            self.matrix = raw_matrix

        if is_filtered:
            self.filter_matrix()

        return self.matrix

    def build_participants_dataframe(self, vote_matrix):
        # TODO: Drop statements from participants-votes when moderated out and no votes.
        def get_comment_count(pid):
            return (self
                .statements_df["participant_id"]
                .value_counts()
                .get(pid, 0)
            )

        def get_xid(pid):
            # TODO: Implemment this
            return None

        def get_group_id(pid):
            # TODO: Implement this.
            return None

        participants_df = (pl.DataFrame()
            .assign(participant_id=vote_matrix.index)
            .set_index('participant_id')
            .assign(xid=[get_xid(pid) for pid in vote_matrix.index])
            .assign(group_id=[get_group_id(pid) for pid in vote_matrix.index])
            .assign(n_comments=[get_comment_count(pid) for pid in vote_matrix.index])
            .assign(n_votes=vote_matrix.count(axis="columns"))
            .assign(n_agree=vote_matrix.apply(lambda row: row.eq(1).sum(), axis="columns"))
            .assign(n_disagree=vote_matrix.apply(lambda row: row.eq(-1).sum(), axis="columns"))
        )
        return participants_df

    def filter_matrix(self):
        # Filter out moderated statements.
        self.matrix = self.matrix.filter(self.get_active_statement_ids(), axis='columns')
        # Filter out participants with less than 7 votes (keeping IDs we're forced to)
        # Ref: https://hyp.is/JbNMus5gEe-cQpfc6eVIlg/gwern.net/doc/sociology/2021-small.pdf
        participant_ids_meeting_vote_thresh = self.matrix[self.matrix.count(axis="columns") >= self.min_votes].index.to_list()
        participant_ids_in = participant_ids_meeting_vote_thresh + self.keep_participant_ids
        participant_ids_in_unique = list(set(participant_ids_in))
        self.matrix = self.matrix.filter(participant_ids_in_unique, axis='rows')
        # This is otherwise the more efficient way, but we want to keep some
        # to troubleshoot bugs in upsteam Polis math.
        # self.matrix = self.matrix.dropna(thresh=self.min_votes, axis='rows')

        # TODO: What about statements with no votes? E.g., 53 in oprah. Filter out? zero?
        # Test this on a conversation where it will actually change statement count.
        unvoted_filter_type = 'drop' # `drop` or `zero`
        if unvoted_filter_type == 'zero':
            self.matrix[self.get_unvoted_statement_ids()] = 0
        elif unvoted_filter_type == 'drop':
            self.matrix = self.matrix.drop(self.get_unvoted_statement_ids(), axis='columns')
        else:
            raise ValueError('unvoted_filter_type must be `drop` or `zero`')

    # Ref: https://hyp.is/8zUyWM5fEe-uIO-J34vbkg/gwern.net/doc/sociology/2021-small.pdf
    def impute_missing_votes(self):
        mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputed_matrix = pl.DataFrame(
            mean_imputer.fit_transform(self.matrix),
            columns=self.matrix.columns,
            index=self.matrix.index,
        )
        return imputed_matrix

    def run_pca(self):
        imputed_matrix = self.impute_missing_votes()

        pca = PCA(n_components=self.n_components) ## pca is apparently different, it wants
        pca.fit(imputed_matrix) ## .T transposes the matrix (flips it)

        self.eigenvectors = pca.components_
        self.eigenvalues = pca.explained_variance_

        # Project participant vote data onto 2D using eigenvectors.
        self.projected_data = pca.transform(imputed_matrix)
        self.projected_data = pl.DataFrame(self.projected_data, index=imputed_matrix.index, columns=["x", "y"])
        self.projected_data.index.name = "participant_id"

    def scale_projected_data(self):
        total_active_comment_count = self.matrix.shape[1]
        participant_vote_counts = self.matrix.count(axis="columns")
        # Ref: https://hyp.is/x6nhItMMEe-v1KtYFgpOiA/gwern.net/doc/sociology/2021-small.pdf
        # Ref: https://github.com/compdemocracy/polis/blob/15aa65c9ca9e37ecf57e2786d7d81a4bd4ad37ef/math/src/polismath/math/pca.clj#L155-L156
        participant_scaling_coeffs = np.sqrt(total_active_comment_count / participant_vote_counts).values
        # See: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        # Reshape scaling_coeffs list to match the shape of projected_data matrix
        participant_scaling_coeffs = np.reshape(participant_scaling_coeffs, (-1, 1))

        self.projected_data = self.projected_data * participant_scaling_coeffs

    # Not working yet.
    def build_base_clusters(self):
        # TODO: Is this participant_count the correct one?
        n_clusters=min(self.base_cluster_count, self.participant_count)
        # Ref: Polis math values, base-iters
        print(self.eigenvectors)
        cluster_labels, cluster_centers = self.run_kmeans(self.eigenvectors, n_clusters=n_clusters)
        # ptpt_to_cluster_mapping = dict(zip(self.matrix.index, range(len(kmeans.labels_))))
        # cluster_to_ptpt_mapping = {v: k for k, v in ptpt_to_cluster_mapping.items()}
        participant_df = pl.DataFrame.from_dict(
            {
                'participant_id': self.matrix.index,
                'cluster_id': cluster_labels.tolist(),
            },
        )
        cluster_df = participant_df.groupby('cluster_id')['participant_id'].apply(list).reset_index(name="participant_ids")
        self.base_clusters['id'] = list(range(100))
        self.base_clusters['members'] = cluster_df["participant_ids"].values.tolist()
        self.base_clusters['x'] = [xy[0] for xy in cluster_centers.tolist()]
        self.base_clusters['y'] = [xy[1] for xy in cluster_centers.tolist()]

    # Not working yet.
    def load_base_clusters_from_math(self):
        self.base_clusters = self.data_loader.math_data["base-clusters"]

    # TODO: Start passing init_centers based on /math/pca2 endpoint data,
    # and see how often we get the same clusters.
    def run_kmeans(self, dataframe, n_clusters=2, init_centers=None):
        if init_centers:
            # Pass an array of xy coords to see kmeans guesses.
            init = init_centers[:n_clusters]
        else:
            # Use the default strategy in sklearn.
            init = "k-means++"
        kmeans = KMeans(n_clusters=n_clusters, random_state=None, init=init, n_init="auto").fit(dataframe)
        return kmeans.labels_, kmeans.cluster_centers_

    def find_optimal_k(self):
        K_RANGE = range(2, 6)
        K_best = 0 # Best K so far.
        silhouette_best = -np.inf

        for K in K_RANGE:
            cluster_labels, _ = self.run_kmeans(dataframe=self.projected_data, n_clusters=K)
            silhouette_K = silhouette_score(self.projected_data, cluster_labels)
            print(f"{K=}, {silhouette_K=}")
            if silhouette_K >= silhouette_best:
                K_best = K
                silhouette_best = silhouette_K
                clusters_K_best = cluster_labels

        self.optimal_k = K_best
        self.optimal_silhouette = silhouette_best
        self.optimal_cluster_labels = clusters_K_best

    def load_data(self, filepaths=[], conversation_id=None, report_id=None):
        if conversation_id or report_id or filepaths:
            self.data_loader = Loader(conversation_id=conversation_id, report_id=report_id, filepaths=filepaths)

            # Infer moderation type from API conversation_data when available.
            if self.data_loader.conversation_data:
                self.is_strict_moderation = self.data_loader.conversation_data["strict_moderation"]

            if self.data_loader.comments_data:
                self.load_comments_data(data=self.data_loader.comments_data)
            if self.data_loader.votes_data:
                self.load_votes_data(data=self.data_loader.votes_data)

    def load_votes_data(self, data=None):
        votes_df = pl.DataFrame.from_records(data).astype({'modified': 'int64'})
        self.add_votes_batch(votes_df)

    def load_comments_data(self, data=None):
        statements_df = (pl.DataFrame
            .from_records(data)
            .set_index('statement_id')
            .sort_index()
        )

        def is_shown(statement):
            if self.get_is_strict_moderation():
                active_mod_states = [ModeratedEnum.APPROVED]
            else:
                active_mod_states = [ModeratedEnum.APPROVED, ModeratedEnum.UNMODERATED]
            is_shown = statement["moderated"] in active_mod_states
            return is_shown

        statements_df["is_shown"] = statements_df.apply(is_shown, axis="columns")
        for i, row in statements_df.iterrows():
            # TODO: Why does is_meta make this mod-in.
            # Maybe I don't understand what mod-in does...
            # Note: mod-in/mod-out doesn't seem to be actually used in the front-end, so a bug here wouldn't matter.
            # Ref: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L825-L842
            if row['is_meta'] or row['moderated'] == 1:
                self.mod_in.append(i)

            if row['is_meta'] or row['moderated'] == -1:
                self.mod_out.append(i)

            # Ref: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L843-L850
            if row['is_meta']:
                self.meta_tids.append(i)

        self.statements_df = statements_df
