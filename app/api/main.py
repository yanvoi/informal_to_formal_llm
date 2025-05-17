from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class TextRequest(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/formalize")
def formalize_text(request: TextRequest):
    """
    Formalize the input text.

    Args:
        request (TextRequest): The request body containing the text to formalize.

    Returns:
        dict: A dictionary containing the formalized text.
    """
    return {"formalized_text": request.text.upper()}
