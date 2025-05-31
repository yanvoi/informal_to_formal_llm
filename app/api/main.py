from fastapi import FastAPI, Depends
from .schemas import TextRequest, FeedbackRequest
from .db import Feedback, get_db


app = FastAPI()


@app.get("/")
def read_root():
    """
    Health check endpoint.

    Args:
        None.

    Returns:
        dict: A dictionary containing a hello world message.
    """
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


@app.post("/feedback")
def save_feedback(feedback_req: FeedbackRequest, db=Depends(get_db)):
    """
    Save user feedback to the database.

    Args:
        feedback_req (FeedbackRequest): The request body containing user input, API output, and feedback type.
        db: Database session dependency.

    Returns:
        dict: A dictionary with the status message.
    """
    feedback = Feedback(
        user_input=feedback_req.user_input,
        api_output=feedback_req.api_output,
        feedback=feedback_req.feedback,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return {"status": "success"}
