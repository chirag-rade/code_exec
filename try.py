class Issue(BaseModel):
    """SINGLE ISSUE"""

    type: str = Field(..., description="Type of the issue identified.")
    reason: str = Field(
        ...,
        description="What is the problem, 1-2 short sentences max. Specifically, what went wrong.",
    )
    cell_positions: list[int] = Field(
        ..., description="List of cell positions that contain the issue."
    )

    fix: str = Field(
        ...,
        description="Suggested fix for the issue, concise but with a clear guidance, not generic. Provide short example like e.g. ... but not a specific example something in between for reference but not for copy paste. No quoted examples. Just reference to what's needed.",
    )


class Intent(BaseModel):
    """User intent."""

    cell_pos: int
    intent: str = Field(
        ...,
        description="User intent, concise, single sentence per user reply. Avoid Assistant intent",
    )


class UserExpertise(BaseModel):
    """User expertise level in a inquiry topic."""

    inquiry_topic: str = Field(..., description="The topic of the user's inquiry.")
    level: str = Field(
        ...,
        description="The user's expertise level in the inquiry topic. Even if assumed. Be concise, 1 sentence max.",
    )


class NotebookWiseFeedback(BaseModel):
    """Scoring of the totality of found issues."""

    user_intents: list[Intent] = Field(
        ...,
        description="Identify user intents. Avoid Assistant intent.",
    )
    user_expertise_level: UserExpertise = Field(
        ...,
        description="Level of expertise and knowledge of the human User in their inquiry topic.",
    )
    scratchpad: str = Field(
        ...,
        description="Place for you to think. Think before issues and score creation. Be concise. Analyze the text to achieve your goal. Always think before looking for issues!",
    )
    issues: list[Issue] = Field(
        ...,
        description="A list of issues in the notebook, each described with where, type, and fix.",
    )
    score: int = Field(
        description="A score representing how good the conversation is, given the total list of mistakes, 1 is far from perfect, 5 is no issues at all.",
        ge=1,
        le=5,
    )


def create_prompt2():

    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                """Name: AI Assistant Perfector.
Profile: You are an expert in perfecting AI Assistants' response content based on the user's expertise level."""
            ),
            HumanMessagePromptTemplate.from_template(
                """
Given the following conversation between Human User and AI Assistant, find issues following the rules described below and rate the total conversation.
Single significant instance of deviation from the rules - score 1 or 2. More issues score<2. No issues=5.

Qualities we care about. Focus on them and only find issues that are directly related to them:
```
You must assume the user just started to learn about the question that is asked, so the 
reply should cover all the points that the user might be new to, and assume the 
user has basic knowledge about the prerequisites. 

This helps us keep the explanation clean, and makes it useful to the user rather 
than throwing all information about the topic to the user.

It is important to identify the query intent to gauge the user knowledge level as well as 
the code complexity to provide the most useful explanation.
```

The task:
```
Please, detect all mismatches between user's expertise level shown and the replies of the Assistant.
If User expertise level is unknown - asumme they are a beginner in that question.
Mismatches might include but not limited to:
    - too much explanation for an expert
    - too little explanation for a beginner
    - Assistant assumes the user is not the beginner in the question asked be it an algo or a technology or something else.

Assume basic knowledge of Python programming by the user and so no need to explain basic things unless asked to.
For example, if the question is about an algorithm in python, assume understnding of Python but a beginner level in algorithms UNLESS USER SHOWS OR STATES A HIGHER OR LOWER LEVEL OF EXPERTISE.

If no issues found, do not create any.
Correctness or accuracy is not your concern and will be handled by other evaluators. Focus only on the serving user's level of expertise in the most helpful manner.
```

Conversation:
CONVERSATION_START
{conversation}
CONVERSATION_END

Now, proceed to completing your task of finding issues and scoring the conversation.
"""
            ),
        ]
    )
    return chat_template