from typing import TypedDict, List, Optional
from enum import IntEnum

IncrementingId = int

class Statement(TypedDict):
     id: IncrementingId

class Participant(TypedDict):
    id: IncrementingId

class ClusteredParticipant(TypedDict):
    id: IncrementingId # participant.id
    x: float
    y: float

class VoteValueEnum(IntEnum):
    AGREE = 1
    DISAGREE = -1
    # Can withhold using "pass" at own discretion.
    PASS = 0

class Vote(TypedDict):
    statement_id: IncrementingId # statement.id
    participant_id: IncrementingId # participant.id

    vote: VoteValueEnum

class Conversation(TypedDict):
    votes: List[Vote]

class Cluster(TypedDict):
    label: int
    participants: List[ClusteredParticipant]

class ClusteringResult(TypedDict):
    clusters: List[Cluster]

class ClusteringOptions(TypedDict):
    min_user_vote_threshold: Optional[int]
    max_clusters: Optional[int]

def run_clustering(conversation: Conversation, options: ClusteringOptions) -> ClusteringResult:
    raise NotImplementedError
