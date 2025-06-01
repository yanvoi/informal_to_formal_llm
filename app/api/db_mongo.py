import os

from pymongo import MongoClient

from .schemas import FeedbackRequest


class MongoFeedbackDatabase:
    """
    MongoDB implementation for storing and retrieving feedback data.

    Attributes:
        client (MongoClient): The MongoDB client.
        db: The MongoDB database instance.
        collection: The feedback collection.
    """

    def __init__(self):
        mongo_uri = os.getenv("MONGO_URI")
        self.client = MongoClient(mongo_uri)
        self.db = self.client.get_database("informal2formal")
        self.collection = self.db.get_collection("feedback")

    def add_feedback(self, feedback_data: FeedbackRequest) -> None:
        """Insert a feedback document into the MongoDB collection.

        Args:
            feedback_data (FeedbackRequest): Validated feedback data.
        """
        self.collection.insert_one(feedback_data.dict())

    def get_all_feedback(self) -> list[FeedbackRequest]:
        """Retrieve all feedback documents from the MongoDB collection.

        Returns:
            list[FeedbackRequest]: List of feedback documents as Pydantic models.
        """
        docs = self.collection.find({}, {"_id": 0})
        return [FeedbackRequest(**doc) for doc in docs]
