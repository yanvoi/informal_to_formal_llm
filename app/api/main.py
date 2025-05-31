from dotenv import load_dotenv
from fastapi import FastAPI
from api.llm.model import LLMService
from api.llm.utils import REPO_ID

from .db_mongo import MongoFeedbackDatabase
from .schemas import FeedbackRequest, TextRequest

load_dotenv()
mongo_db = MongoFeedbackDatabase()
app = FastAPI()


def get_llm_service() -> LLMService:
    return app.state.llm_service


@app.on_event("startup")
def startup_event():
    app.state.llm_service = LLMService(REPO_ID)


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
def formalize_text(request: TextRequest, llm_service: LLMService = Depends(get_llm_service)):
    """
    Formalize the input text.

    Args:
        request (TextRequest): The request body containing the text to formalize.

    Returns:
        dict: A dictionary containing the formalized text.
    """
    return {"formalized_text": llm_service.formalize(request.text)}


@app.post("/feedback")
def save_feedback(feedback_req: FeedbackRequest) -> dict:
    """
    Save user feedback to the database.

    Args:
        feedback_req (FeedbackRequest): The request body containing user input, API output, and feedback type.

    Returns:
        dict: A dictionary with the status message.
    """
    mongo_db.add_feedback(feedback_req)
    return {"status": "success"}


@app.get("/feedback", response_model=list[FeedbackRequest])
def get_feedback() -> list[FeedbackRequest]:
    """
    Retrieve all feedback from the database.

    Returns:
        list[FeedbackRequest]: A list of feedback documents.
    """
    return mongo_db.get_all_feedback()
