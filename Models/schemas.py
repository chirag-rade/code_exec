from enum import Enum
from pydantic import BaseModel, Field, validator
from typing import Any

class AspectorRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    USER_AND_ASSISTANT = "user and assistant"  

class QualityAspect(BaseModel):
    name: str = Field(..., description="The name of the quality aspect.")
    instruction: str = Field(..., description="Instructions & details on how to inspect this quality aspect.")

class AspectorEvaluatorInputSchema(BaseModel):
    quality_aspect: QualityAspect = Field(..., description="The quality aspect to focus on during the evaluation, including its name and specific instructions.")
    role: AspectorRole = Field(..., description="The role perspective from which the quality aspect should be evaluated.")
    conversation: list[str] = Field(..., description="The conversation text to be evaluated.")
    metadata: dict[str, Any] = Field(..., description="Metadata related to the conversation.")
    
    
class FeedbackISC(BaseModel):
    """Evaluation result with ISSUES,SCORE and COMMENT"""

    issues: list[str] = Field(
        description="A concrete list of issues in the conversation. 15 words or less each."
    )
    score: int = Field(
        description="A score representing how good the conversation is in the given quality aspect, 1 is terrible, 5 is exemplary and flawless.",

    )
    comment: str = Field(
        description="A score representing how good the conversation is in the given quality aspect, 1 is terrible, 5 is exemplary and flawless.",

    )
    


class FeedbackBasic(BaseModel):
    response: str = Field(description="The response string returned by the LLM")