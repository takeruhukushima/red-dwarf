from typing import TypedDict, TypeAlias

IncrementingId: TypeAlias = int
BaseClusterId: TypeAlias = IncrementingId
GroupId: TypeAlias = IncrementingId
ParticipantId: TypeAlias = IncrementingId

class PolisGroupCluster(TypedDict):
    id: GroupId
    members: list[BaseClusterId]
    center: list[float]

class PolisGroupClusterExpanded(TypedDict):
    id: GroupId
    members: list[ParticipantId]
    center: list[float]

class PolisBaseClusters(TypedDict):
    # Each outer list will be the same length, and will be 100 items or less.
    id: list[BaseClusterId]
    members: list[list[ParticipantId]]
    x: list[float]
    y: list[float]
    count: list[int]