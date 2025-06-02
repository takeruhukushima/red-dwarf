from numpy import integer as npInt
from typing import TypeAlias, Literal, Optional
from typing_extensions import NotRequired, TypedDict


UnixTimestampMillisec: TypeAlias = int
IncrementingId: TypeAlias = int
BaseClusterId: TypeAlias = IncrementingId
GroupId: TypeAlias = IncrementingId
ParticipantId: TypeAlias = IncrementingId
StatementId: TypeAlias = IncrementingId

PolisCommentPriorities: TypeAlias = dict[str, float]  # str[StatementId]
PolisUserVoteCounts: TypeAlias = dict[str, int]  # str[ParticipantId]


class PolisGroupCluster(TypedDict):
    id: GroupId
    members: list[BaseClusterId]
    center: tuple[float, float]


# Custom type
class PolisGroupClusterExpanded(TypedDict):
    id: GroupId
    members: list[ParticipantId]
    center: tuple[float, float]


BaseClusterMembership: TypeAlias = list[ParticipantId]


class PolisBaseClusters(TypedDict):
    # Each outer list will be the same length, and will be 100 items or less.
    id: list[BaseClusterId]
    members: list[BaseClusterMembership]
    x: list[float]
    y: list[float]
    count: list[int]


class PolisRepnessStatement(TypedDict):
    tid: int
    n_success: int
    n_trials: int
    p_success: float
    p_test: float
    repness: float
    repness_test: float
    repful_for: Literal["agree", "disagree"]
    best_agree: NotRequired[bool]
    n_agree: NotRequired[int]


# polismath uses str, but int/GroupId probably makes more sense to use.
PolisRepness: TypeAlias = dict[str | GroupId, list[PolisRepnessStatement]]

PerBaseVoteCounts: TypeAlias = list[int]


class PolisBaseClusterVoteSummary(TypedDict):
    A: PerBaseVoteCounts
    D: PerBaseVoteCounts
    S: PerBaseVoteCounts


class PolisPCA(TypedDict):
    # Each outer list will be the same length, one item for each statement.
    center: list[float]
    comps: tuple[list[float], list[float]]
    comment_projection: tuple[list[float], list[float]]
    comment_extremity: list[float]


class PolisConsensusStatement(TypedDict):
    tid: StatementId
    n_success: int
    n_trials: int
    p_success: float
    p_test: float


class PolisConsensus(TypedDict):
    agree: list[PolisConsensusStatement]
    disagree: list[PolisConsensusStatement]


class PolisStatementVoteSummary(TypedDict):
    A: int
    D: int
    S: int


class PolisGroupVote(TypedDict):
    n_members: int
    votes: dict[str, PolisStatementVoteSummary]  # str[StatementId]
    id: GroupId


PolisGroupVotes: TypeAlias = dict[str, PolisGroupVote]  # str[GroupId]


class PolisMath(TypedDict):
    tids: list[StatementId]
    meta_tids: list[StatementId]
    mod_in: list[StatementId]
    mod_out: list[StatementId]

    in_conv: list[ParticipantId]
    user_vote_counts: PolisUserVoteCounts

    # For drawing graph
    pca: PolisPCA
    group_clusters: list[PolisGroupCluster]

    # Consensus statements
    consensus: PolisConsensus

    # Group statements (representative)
    repness: PolisRepness
    group_votes: PolisGroupVotes

    # Overall statements
    comment_priorities: PolisCommentPriorities
    group_aware_consensus: dict[str, float]  # str[StatementId]

    # Base clusters
    base_clusters: PolisBaseClusters
    votes_base: dict[str, PolisBaseClusterVoteSummary]  # str[StatementId]

    n: int
    n_cmts: int
    lastModTimestamp: None
    lastVoteTimestamp: UnixTimestampMillisec
    math_tick: int
