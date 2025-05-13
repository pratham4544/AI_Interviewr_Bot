from pymongo import MongoClient
import os
from dotenv import load_dotenv
from bson import Binary

load_dotenv()

class MongoDBHandler:
    def __init__(self):
        mongo_uri = os.getenv("MONGODB_URI")  # Add this to your .env
        self.client = MongoClient(mongo_uri)
        self.db = self.client["interview_bot"]
        self.collection = self.db["sessions"]

    def save_response(self, session_id, question_index, question, answer_text, evaluation, audio_data,photo_data):
        document = {
            "session_id": session_id,
            "question_index": question_index,
            "question": question,
            "answer_text": answer_text,
            "evaluation": evaluation,
            "audio_data": Binary(audio_data),
            "photo_data": photo_data
            
        }
        
        self.collection.insert_one(document)
