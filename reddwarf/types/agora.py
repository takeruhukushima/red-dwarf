from typing import TypedDict, List
from enum import Enum

IncrementingId = int

class Statement(TypedDict):
     id: IncrementingId

class Participant(TypedDict):
    id: IncrementingId

class ClusteredParticipant(Participant):
    x: float
    y: float

class VoteValueEnum(Enum):
    AGREE = "agree"
    DISAGREE = "disagree"
    # Can withhold using "pass" at own discretion.
    PASS = "pass"

class Options(TypedDict):
    pass
    # define some general options
    # you tell me what you need.
    # Can be the "center", but for the first version _keep it simple_, hard-code something deterministic

class Vote(TypedDict):
    id: IncrementingId
    conversation_id: IncrementingId
    statement_id: IncrementingId
    voted_by_participant_id: IncrementingId

    vote: VoteValueEnum

class Conversation(TypedDict):
    id: IncrementingId
    statements: List[Statement]
    participants: List[Participant]
    # votes_by_participants or votes_by_statement can be easily generated from this if required.
    votes: List[Vote]
    options: Options

class Cluster(TypedDict):  ## exported type
    center_x: float
    center_y: float
    participants: List[ClusteredParticipant]
    # we don't need anything else,
    # because the function caller already has all the information necessary
    # to calculate the statistical information about how a cluster voted on each opinions,
    # and hence the caller can calculate the "representative" core opinions
    # alone without the help of an external library
    # (majority opinions, controversial opinions, etc)

ClusteringResult = List[Cluster] ## exported type, between 2 and 6 clusters, amount selected based on the "best" according to the alg

def run_clustering(
    *,
    conversation: Conversation,
    options: Options,
) -> ClusteringResult:
    raise NotImplementedError
