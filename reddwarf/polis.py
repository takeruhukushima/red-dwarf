import polars as pl
from collections import defaultdict

class PolisClient():
    def __init__(self) -> None:
        self.n_components = 2
        # Ref: https://hyp.is/8zUyWM5fEe-uIO-J34vbkg/gwern.net/doc/sociology/2021-small.pdf
        self.min_votes = 7
        self.votes = []
        # Ref: https://hyp.is/MV0Iws5fEe-k9BdY6UR1VQ/gwern.net/doc/sociology/2021-small.pdf
        self.vote_matrix = None
        # Ref: https://gist.github.com/patcon/fd9079a5fbcd533160f8ae211e975307#file-math-pca2-json-L2
        # Ref: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L217-L225
        self.user_vote_counts = defaultdict(int)
        self.meta_tids = []
        self.mod_in = []
        self.mod_out = []
        self.last_vote_timestamp = 0
        self.group_clusters = []
        self.base_clusters = {}

    def impute_missing_votes(self):
        # Ref: https://hyp.is/8zUyWM5fEe-uIO-J34vbkg/gwern.net/doc/sociology/2021-small.pdf
        # Use "mean" strategy of sklearn SimpleImputer.
        # See: https://scikit-learn.org/1.6/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer
        raise NotImplementedError

    def get_participant_row_mask(self):
        raise NotImplementedError

    def get_statement_col_mask(self):
        raise NotImplementedError

    def apply_masks(self, participant_rows=True, statement_cols=True):
        raise NotImplementedError

    def add_votes_batch(self, votes_df):
        """Add multiple votes from a DataFrame"""
        # TODO: Use tuples instead of named columns later, for perf.
        for row in votes_df.iter_rows(named=True):
            self.add_vote(row)

    def add_vote(self, vote_row):
        """Add a single vote to the system"""
        # If this is a new vote (not an update)
        self.user_vote_counts[vote_row['pid']] += 1

        if self.last_vote_timestamp < int(vote_row['modified']):
            self.last_vote_timestamp = int(vote_row['modified'])

        self.votes.append({
            'participant_id': vote_row['pid'],
            'statement_id': vote_row['tid'],
            'vote': vote_row['vote'],
        })
        # Matrix is now stale
        self.matrix = None

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

    def get_matrix(self):
        if self.matrix is None:
            # Only generate matrix when needed.
            self.matrix = pl.from_dicts(self.votes)
            self.matrix = self.matrix.pivot(values="vote", index="participant_id", on="statement_id")

        return self.matrix

    def load_data(self, filepath):
        if filepath.endswith("votes.json"):
            self.load_votes_data(filepath)
        elif filepath.endswith("comments.json"):
            self.load_comments_data(filepath)
        else:
            raise ValueError("Unknown file type")

    def load_votes_data(self, filepath):
        votes_df = pl.read_json(filepath)
        self.add_votes_batch(votes_df)

    def load_comments_data(self, filepath):
        comments_df = pl.read_json(filepath)
        for row in comments_df.iter_rows(named=True):
            # TODO: Why does is_meta make this mod-in.
            # Maybe I don't understand what mod-in does...
            # Note: mod-in/mod-out doesn't seem to be actually used in the front-end, so a bug here wouldn't matter.
            # Ref: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L825-L842
            if row['is_meta'] or row['mod'] == 1:
                self.mod_in.append(row['tid'])

            if row['is_meta'] or row['mod'] == -1:
                self.mod_out.append(row['tid'])

            # Ref: https://github.com/compdemocracy/polis/blob/6d04f4d144adf9640fe49b8fbaac38943dc11b9a/math/src/polismath/math/conversation.clj#L843-L850
            if row['is_meta']:
                self.meta_tids.append(row['tid'])

