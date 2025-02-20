from typing import TypedDict, TypeAlias, List, Dict
from enum import Enum

IncrementingId = int

class Statement(TypedDict):  # exported type
     id: IncrementingId

class Participant(TypedDict):  # exported type
    id: IncrementingId

class VoteValueEnum(Enum):  # exported type
    AGREE = "agree"
    DISAGREE = "disagree"
    # NO "PASS". Pass is usually needed for things like "comment_priorities", but we don't need that here.

class Options(TypedDict):  # exported type
    pass
    # define some general options
    # you tell me what you need.
    # Can be the "center", but for the first version _keep it simple_, hard-code something deterministic

class Vote(TypedDict):
    id: IncrementingId
    conversation_id: IncrementingId
    statement_id: IncrementingId
    voted_by_participant_id: IncrementingId

    value: VoteValueEnum

class Conversation(TypedDict):  # exported type
    id: IncrementingId
    opinions: List[Statement]
    participants: List[Participant]
    # This is redundant, only one of them is enouh, _you_ tell me what you need:
    votes: List[Vote]
    # votes_by_participants: Dict[int, Dict[int, VoteValueEnum]] # key is opinion.id]  # key is participant.id
    # votes_by_opinions: Dict[int, Dict[int, VoteValueEnum]] # key is participant.id]  # key is opinion.id
    options: Options


class Cluster(TypedDict):  ## exported type
    members: List[Participant]  # list of participants belonging to this cluster
    # we don't need anything else,
    # because the function caller already has all the information necessary
    # to calculate the statistical information about how a cluster voted on each opinions,
    # and hence the caller can calculate the "representative" core opinions
    # alone without the help of an external library
    # (majority opinions, controversial opinions, etc)

Clusters = List[Cluster] ## exported type, between 2 and 6 clusters, amount selected based on the "best" according to the alg

ClustersByConversations: TypeAlias = Dict[int, Clusters]  # exported type--key is conversation.id

# the function I need
def calculate_clusters(
    *,
    conversations: List[Conversation],
    options: Options,
) -> ClustersByConversations:
    raise NotImplementedError
