import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

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
