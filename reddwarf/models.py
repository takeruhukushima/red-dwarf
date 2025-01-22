from pydantic import BaseModel, NonNegativeInt, Field, AliasChoices, field_serializer
from typing import Literal, Optional, NewType, Annotated
from datetime import datetime
from enum import IntEnum

class VoteEnum(IntEnum):
    AGREE = 1
    PASS = 0
    DISAGREE = -1

IncrementingId = NewType('IncrementingId', NonNegativeInt)

class Vote(BaseModel):
    @field_serializer('modified')
    def serialize_modified(self, modified: datetime, _info):
        return modified.timestamp() * 1000

    participant_id: IncrementingId = Field(
        validation_alias=AliasChoices('participant_id', 'pid', 'voter-id'),
        serialization_alias="participant_id",
    )
    statement_id: IncrementingId = Field(validation_alias=AliasChoices('statement_id', 'tid', 'comment-id'))
    vote: VoteEnum
    weight_x_32767: Optional[Literal[0]] = None
    modified: datetime = Field(
        validation_alias=AliasChoices('modified', 'timestamp'),
        serialization_alias="modified",
    )
    conversation_id: Optional[str] = None
    datetime: Optional[Annotated[str, Field(exclude=True)]] = None
