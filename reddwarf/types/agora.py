from typing import TypedDict, List, Optional, Union
from enum import IntEnum

Identifier = Union[int, str]

class Statement(TypedDict):
     id: Identifier

class Participant(TypedDict):
    id: Identifier

class ClusteredParticipant(TypedDict):
    id: Identifier # participant.id
    x: float
    y: float

class VoteValueEnum(IntEnum):
    AGREE = 1
    DISAGREE = -1
    # Can withhold using "pass" at own discretion.
    PASS = 0

class Vote(TypedDict):
    statement_id: Identifier # statement.id
    participant_id: Identifier # participant.id

    vote: VoteValueEnum

class Conversation(TypedDict):
    votes: List[Vote]

class Cluster(TypedDict):
    id: int
    participants: List[ClusteredParticipant]

class ClusteringResult(TypedDict):
    clusters: List[Cluster]

class ClusteringOptions(TypedDict):
    min_user_vote_threshold: Optional[int]
    max_clusters: Optional[int]

def run_clustering(conversation: Conversation, options: ClusteringOptions) -> ClusteringResult:
    raise NotImplementedError
