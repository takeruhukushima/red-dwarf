import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import List, Dict, Optional

def impute_missing_votes(vote_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing votes in a voting matrix using column-wise mean.

    Reference:
        Small, C. (2021). "Polis: Scaling Deliberation by Mapping High Dimensional Opinion Spaces."
        Specific highlight: https://hyp.is/8zUyWM5fEe-uIO-J34vbkg/gwern.net/doc/sociology/2021-small.pdf

    Args:
    vote_matrix (pd.DataFrame):  A vote matrix DataFrame with NaN values where: \
                                    (1) rows are voters, \
                                    (2) columns are statements, and \
                                    (3) values are votes.
    Returns:
    imputed_matrix (pd.DataFrame): The same vote matrix DataFrame imputing NaN values with column mean.
    """
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_matrix = pd.DataFrame(
        mean_imputer.fit_transform(vote_matrix),
        columns=vote_matrix.columns,
        index=vote_matrix.index,
    )
    return imputed_matrix

def generate_raw_matrix(
        votes: List[Dict],
        cutoff: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generates a raw vote matrix from a list of vote records.

    If a cutoff is provided, votes are filtered based on either:
    - An `int` representing unix timestamp (ms), keeping only votes before or at that time.
        - Any int above 13_000_000_000 is considered a timestamp.
    - Any other positive or negative `int` is considered an index, reflecting where to trim the time-sorted vote list.
        - positive: filters in votes that many indices from start
        - negative: filters out votes that many indices from end

    Args:
        votes (List[Dict]): A date-sorted list of vote records, where each record is a dictionary containing: \
            - "participant_id": The ID of the voter. \
            - "statement_id": The ID of the statement being voted on. \
            - "vote": The recorded vote value. \
            - "modified": A unix timestamp object representing when the vote was made.

    Returns:
        raw_matrix (pd.DataFrame): A full raw vote matrix DataFrame with NaN values where: \
                                        (1) rows are voters, \
                                        (2) columns are statements, and \
                                        (3) values are votes. \
                                    This includes even voters that have no votes, and statements on which no votes were placed.
    """
    if cutoff:
        # TODO: Add tests to confirm votes list is already date-sorted for each data_source.
        # TODO: Detect datetime object as arg instead.
        if cutoff > 1_300_000_000:
            cutoff_timestamp = cutoff
            votes = [v for v in votes if v['modified'] <= cutoff_timestamp]
        else:
            cutoff_index = cutoff
            votes = votes[:cutoff_index]
    else:
        votes = votes

    raw_matrix = pd.DataFrame.from_dict(votes)
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
