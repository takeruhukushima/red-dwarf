import pandas as pd
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Tuple, Optional
from types import SimpleNamespace
from scipy.stats import norm
from reddwarf.utils.matrix import VoteMatrix
from reddwarf.types.polis import (
    PolisRepness,
    PolisRepnessStatement,
)


def one_prop_test(
    succ: ArrayLike,
    n: ArrayLike,
) -> np.float64 | NDArray[np.float64]:
    # Convert inputs to numpy arrays (if they aren't already)
    succ = np.asarray(succ) + 1
    n = np.asarray(n) + 1

    # Compute the test statistic
    return 2 * np.sqrt(n) * ((succ / n) - 0.5)

# Calculate representativeness two-prop test
def two_prop_test(
    succ_in: ArrayLike,
    succ_out: ArrayLike,
    n_in: ArrayLike,
    n_out: ArrayLike,
) -> np.float64 | NDArray[np.float64]:
    """Two-prop test ported from Polis. Accepts numpy arrays for bulk processing."""
    # Ported with adaptation from Polis
    # See: https://github.com/compdemocracy/polis/blob/90bcb43e67dad660629e0888fedc0d32379f375d/math/src/polismath/math/stats.clj#L18-L33

    # Laplace smoothing (add 1 to each count)
    succ_in = np.asarray(succ_in) + 1
    succ_out = np.asarray(succ_out) +  1
    n_in = np.asarray(n_in) +  1
    n_out = np.asarray(n_out) +  1

    # Compute proportions
    pi1 = succ_in / n_in
    pi2 = succ_out / n_out
    pi_hat = (succ_in + succ_out) / (n_in + n_out)

    # Handle edge case when pi_hat == 1
    # TODO: Handle when divide by zero.
    if np.any(pi_hat == 1):
        # XXX: Not technically correct; limit-based solution needed.
        return np.where(pi_hat == 1, 0, (pi1 - pi2) / np.sqrt(pi_hat * (1 - pi_hat) * (1 / n_in + 1 / n_out)))

    # Compute the test statistic
    denominator = np.sqrt(pi_hat * (1 - pi_hat) * (1 / n_in + 1 / n_out))

    return (pi1 - pi2) / denominator

def is_significant(z_val: float, confidence: float = 0.90) -> bool:
    """Test whether z-statistic is significant at 90% confidence (one-tailed, right-side)."""
    critical_value = norm.ppf(confidence)  # 90% confidence level, one-tailed
    return z_val > critical_value

def is_statement_significant(row: pd.Series, confidence=0.90) -> bool:
    "Decide whether we should count a statement in a group as being representative."
    pat, rat, pdt, rdt = [row[col] for col in ["pat", "rat", "pdt", "rdt"]]
    is_agreement_significant = is_significant(pat, confidence) and is_significant(rat, confidence)
    is_disagreement_significant = is_significant(pdt, confidence) and is_significant(rdt, confidence)

    return is_agreement_significant or is_disagreement_significant

def beats_best_by_repness_test(
    this_row: pd.Series,
    best_row: pd.Series | None,
) -> bool:
    """
    Returns True if a given comment/group stat has a more representative z-score than
    current_best_z. Used for ensuring at least one representative comment for every group,
    even if none remain after more thorough filters.
    """
    if best_row is not None:
        this_repness_test = max(this_row["rat"], this_row["rdt"])
        best_repness_test = max(best_row["rat"], best_row["rdt"])

        return this_repness_test > best_repness_test
    else:
        return True


def beats_best_of_agrees(
    this_row: pd.Series,
    best_row: pd.Series | None,
    confidence: float = 0.90,
) -> bool:
    """
    Like beats_best_by_repness_test, but only considers agrees. Additionally, doesn't
    focus solely on repness, but also on raw probability of agreement, so as to
    ensure that there is some representation of what people in the group agree on.
    """
    # Explicitly don't allow something that hasn't been voted on at all
    if this_row["na"] == 0 and this_row["nd"] == 0:
        return False

    if best_row is not None:
        if best_row["ra"] > 1.0:
            # If we have a current_best by representativeness estimate, use the more robust measurement
            repness_metric_agr = lambda row: row["ra"] * row["rat"] * row["pa"] * row["pat"]
            return repness_metric_agr(this_row) > repness_metric_agr(best_row)
        else:
            # If we have current_best, but only by probability estimate, just shoot for something generally agreed upon
            prob_metric = lambda row: row["pa"] * row["pat"]
            return prob_metric(this_row) > prob_metric(best_row)

    # Otherwise, accept if either repness or probability look generally good.
    # TODO: Hardcode significance at 90%, so lways one statement for group?
    return (is_significant(this_row["pat"], confidence) or
        (this_row["ra"] > 1.0 and this_row["pa"] > 0.5))

# For making it more legible to get index of agree/disagree votes in numpy array.
votes = SimpleNamespace(A=0, D=1)

def count_votes(
    values: ArrayLike,
    vote_value: Optional[int] = None,
) -> np.int64 | NDArray[np.int64]:
    values = np.asarray(values)
    if vote_value:
        # Count votes that match value.
        return np.sum(values == vote_value, axis=0)
    else:
        # Count any non-missing values.
        return np.sum(np.isfinite(values), axis=0)

def count_disagree(values: ArrayLike) -> np.int64 | NDArray[np.int64]:
    return count_votes(values, -1)

def count_agree(values: ArrayLike) -> np.int64 | NDArray[np.int64]:
    return count_votes(values, 1)

def count_all_votes(values: ArrayLike) -> np.int64 | NDArray[np.int64]:
    return count_votes(values)

def probability(count, total, pseudo_count=1):
    """Probability with Laplace smoothing"""
    return (pseudo_count + count ) / (2*pseudo_count + total)

def calculate_comment_statistics(
    vote_matrix: VoteMatrix,
    cluster_labels: list[int] | NDArray[np.integer],
    pseudo_count: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates comparative statement statistics across all votes and groups, using only efficient numpy operations.

    The representativeness metric is defined as:
    R_v(g,c) = P_v(g,c) / P_v(~g,c)

    Where:
    - P_v(g,c) is probability of vote v on comment c in group g
    - P_v(~g,c) is probability of vote v on comment c in all groups except g

    And:
    - N(g,c) is the total number of non-missing votes on comment c in group g
    - N_v(g,c) is the total number of vote v on comment c in group g

    Args:
        vote_matrix (VoteMatrix): ...
        cluster_labels (list[int]): ...

    Returns:
        N_g_c (np.ndarray[int]): numpy matrix with counts of non-missing votes on comments/groups
        N_v_g_c (np.ndarray[int]): numpy matrix with counts of vote types on comments/groups
        P_v_g_c (np.ndarray[float]): numpy matrix with probabilities of vote types on comments/groups
        R_v_g_c (np.ndarray[float]): numpy matrix with representativeness of vote types on comments/groups
        P_v_g_c_test (np.ndarray[float]): test z-scores for probability of votes/comments/groups
        R_v_g_c_test (np.ndarray[float]): test z-scores for representativeness of votes/comments/groups
    """
    # Get the vote matrix values
    X = vote_matrix.values

    group_count = len(set(cluster_labels))
    statement_ids = vote_matrix.columns

    # Set up all the variables to be populated.
    N_g_c = np.empty([group_count, len(statement_ids)], dtype='int32')
    N_v_g_c = np.empty([len(votes.__dict__), group_count, len(statement_ids)], dtype='int32')
    P_v_g_c = np.empty([len(votes.__dict__), group_count, len(statement_ids)])
    R_v_g_c = np.empty([len(votes.__dict__), group_count, len(statement_ids)])
    P_v_g_c_test = np.empty([len(votes.__dict__), group_count, len(statement_ids)])
    R_v_g_c_test = np.empty([len(votes.__dict__), group_count, len(statement_ids)])

    for gid in range(group_count):
        # Create mask for the participants in target group
        in_group_mask = (np.asarray(cluster_labels) == gid)
        X_in_group = X[in_group_mask]

        # Count any votes [-1, 0, 1] for all statements/features at once

        # NON-GROUP STATS

        # For in-group
        n_agree_in_group    = N_v_g_c[votes.A, gid, :] = count_agree(X_in_group)    # na
        n_disagree_in_group = N_v_g_c[votes.D, gid, :] = count_disagree(X_in_group) # nd
        n_votes_in_group    = N_g_c[gid, :] = count_all_votes(X_in_group)           # ns

        # Calculate probabilities
        p_agree_in_group    = P_v_g_c[votes.A, gid, :] = probability(n_agree_in_group, n_votes_in_group, pseudo_count)    # pa
        p_disagree_in_group = P_v_g_c[votes.D, gid, :] = probability(n_disagree_in_group, n_votes_in_group, pseudo_count) # pd

        # Calculate probability test z-scores
        P_v_g_c_test[votes.A, gid, :] = one_prop_test(n_agree_in_group, n_votes_in_group)    # pat
        P_v_g_c_test[votes.D, gid, :] = one_prop_test(n_disagree_in_group, n_votes_in_group) # pdt

        # GROUP COMPARISON STATS

        out_group_mask = ~in_group_mask
        X_out_group = X[out_group_mask]

        # For out-group
        n_agree_out_group = count_agree(X_out_group)
        n_disagree_out_group = count_disagree(X_out_group)
        n_votes_out_group = count_all_votes(X_out_group)

        # Calculate out-group probabilities
        p_agree_out_group    = probability(n_agree_out_group, n_votes_out_group, pseudo_count)
        p_disagree_out_group = probability(n_disagree_out_group, n_votes_out_group, pseudo_count)

        # Calculate representativeness
        R_v_g_c[votes.A, gid, :] = p_agree_in_group / p_agree_out_group       # ra
        R_v_g_c[votes.D, gid, :] = p_disagree_in_group / p_disagree_out_group # rd

        # Calculate representativeness test z-scores
        R_v_g_c_test[votes.A, gid, :] = two_prop_test(n_agree_in_group, n_agree_out_group, n_votes_in_group, n_votes_out_group)       # rat
        R_v_g_c_test[votes.D, gid, :] = two_prop_test(n_disagree_in_group, n_disagree_out_group, n_votes_in_group, n_votes_out_group) # rdt

    return (
        N_g_c, # ns
        N_v_g_c, # na / nd
        P_v_g_c, # pa / pd
        R_v_g_c, # ra / rd
        P_v_g_c_test, # pat / pdt
        R_v_g_c_test, # rat / rdt
    )

def finalize_cmt_stats(statement: pd.Series) -> PolisRepnessStatement:
    """
    Convert verbose internal statistics into more concise agree/disagree format.

    The most representative of agree/disagree is used.

    Args:
        statement (pd.Series): A dataframe row with statement stats in verbose format.

    Returns:
        PolisRepnessStatement: A dict with keys matching expected Polis format.
    """
    if statement["rat"] > statement["rdt"]:
        vals = [statement[k] for k in ["na", "ns", "pa", "pat", "ra", "rat"]] + ["agree"]
    else:
        vals = [statement[k] for k in ["nd", "ns", "pd", "pdt", "rd", "rdt"]] + ["disagree"]

    return {
        "tid":          int(statement["statement_id"]),
        "n-success":    int(vals[0]),
        "n-trials":     int(vals[1]),
        "p-success":    float(vals[2]),
        "p-test":       float(vals[3]),
        "repness":      float(vals[4]),
        "repness-test": float(vals[5]),
        "repful-for":   vals[6], # type:ignore
    }

def calculate_comment_statistics_by_group(
    vote_matrix: VoteMatrix,
    cluster_labels: list[int] | NDArray[np.integer],
    pseudo_count: int = 1,
) -> list[pd.DataFrame]:
    """
    Calculates comparative statement statistics across all votes and groups, generating dataframes.

    Args:
        vote_matrix (VoteMatrix): The vote matrix where rows are voters, columns are statements,
                                  and values are votes (1 for agree, -1 for disagree, 0 for pass).
        cluster_labels (np.ndarray): Array of cluster labels for each participant row in the vote matrix.
        pseudo_count (int): Smoothing parameter to avoid division by zero. Default is 1.

    Returns:
        pd.DataFrame: DataFrame containing verbose statistics for each statement.
    """
    N_g_c, N_v_g_c, P_v_g_c, R_v_g_c, P_v_g_c_test, R_v_g_c_test = calculate_comment_statistics(
        vote_matrix=vote_matrix,
        cluster_labels=cluster_labels,
        pseudo_count=pseudo_count,
    )

    group_count = len(set(cluster_labels))
    group_stats = [pd.DataFrame()] * group_count
    for group_id in range(group_count):
        group_stats[group_id] = pd.DataFrame({
            'na':  N_v_g_c[votes.A, group_id, :],      # number agree votes
            'nd':  N_v_g_c[votes.D, group_id, :],      # number disagree votes
            'ns':  N_g_c[group_id, :],                 # number seen/total/non-missing votes
            'pa':  P_v_g_c[votes.A, group_id, :],      # probability agree
            'pd':  P_v_g_c[votes.D, group_id, :],      # probability disagree
            'pat': P_v_g_c_test[votes.A, group_id, :], # probability agree test z-score
            'pdt': P_v_g_c_test[votes.D, group_id, :], # probability disagree test z-score
            'ra':  R_v_g_c[votes.A, group_id, :],      # repness of agree (representativeness)
            'rd':  R_v_g_c[votes.D, group_id, :],      # repness of disagree (representativeness)
            'rat': R_v_g_c_test[votes.A, group_id, :], # repress of agree test z-score
            'rdt': R_v_g_c_test[votes.D, group_id, :], # repress of disagree test z-score
        }, index=vote_matrix.columns)

    return group_stats

def repness_metric(df: pd.DataFrame) -> pd.Series:
    metric = df["repness"] * df["repness-test"] * df["p-success"] * df["p-test"]
    return metric

# Figuring out select-rep-comments flow
# See: https://github.com/compdemocracy/polis/blob/7bf9eccc287586e51d96fdf519ae6da98e0f4a70/math/src/polismath/math/repness.clj#L209C7-L209C26
# TODO: omg please clean this up.
def select_representative_statements(
    grouped_stats_df: list[pd.DataFrame],
    pick_max: int = 5,
    confidence: float = 0.90,
) -> PolisRepness:
    """
    Selects statistically representative statements from each group cluster.

    This is expected to match the Polis outputs when all defaults are set.

    Args:
        grouped_stats_df (list[pd.DataFrame]): Dataframes of statement statistics, keyed to group ID
        pick_max (int): The max number of statements that will be returned per group
        confidence (float): A decimal percentage representing confidence interval

    Returns:
        PolisRepness: A dict object with lists of statements keyed to groups, matching Polis format.
    """
    repness = {}
    for gid, stats_df in enumerate(grouped_stats_df):
        stats_df = stats_df.reset_index()

        best_agree = None
        # Track the best-agree, to bring to top if exists.
        for _, row in stats_df.iterrows():
            if beats_best_of_agrees(row, best_agree, confidence):
                best_agree = row

        sig_filter = lambda row: is_statement_significant(row, confidence)
        sufficient_statements_row_mask = stats_df.apply(sig_filter, axis="columns")
        sufficient_statements = stats_df[sufficient_statements_row_mask].reset_index()

        # Track the best, even if doesn't meet sufficient minimum, to have at least one.
        best_overall = None
        if len(sufficient_statements) == 0:
            for _, row in stats_df.iterrows():
                if beats_best_by_repness_test(row, best_overall):
                    best_overall = row
        else:
            # Finalize statements into output format.
            # TODO: Figure out how to finalize only at end in output. Change repness_metric?
            sufficient_statements = (
                pd.DataFrame([
                    finalize_cmt_stats(row)
                        for _, row in sufficient_statements.iterrows()
                ])
                    # Create a column to sort repnress, then remove.
                    .assign(repness_metric=repness_metric)
                    .sort_values(by="repness_metric", ascending=False)
                    .drop(columns="repness_metric")
            )

        if best_agree is not None:
            best_agree = finalize_cmt_stats(best_agree)
            best_agree.update({"n-agree": best_agree["n-success"], "best-agree": True})
            best_head = [best_agree]
        elif best_overall is not None:
            best_overall = finalize_cmt_stats(best_overall)
            best_head = [best_overall]
        else:
            best_head = []

        selected = best_head
        selected = selected + [
            row.to_dict() for _, row in sufficient_statements.iterrows()
            if best_head
                # Skip any statements already in best_head
                and best_head[0]["tid"] != row["tid"]
        ]
        selected = selected[:pick_max]
        # Does the work of agrees-before-disagrees sort in polismath, since "a" before "d".
        selected = sorted(selected, key=lambda row: row["repful-for"])
        repness[str(gid)] = selected

    return repness # type:ignore