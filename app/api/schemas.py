from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    user_input: str
    api_output: str
    feedback: str
