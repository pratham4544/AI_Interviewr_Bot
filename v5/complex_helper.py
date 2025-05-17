import os
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import json
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Tuple
from pymongo import MongoClient
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from groq import Groq
import time
from langchain_core.output_parsers import JsonOutputParser
import threading


load_dotenv()

# Initialize API clients
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Initialize Groq client and LLM
client = Groq(api_key=GROQ_API_KEY)
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0)

class InterviewerBot:
    def __init__(self):
        self.candidate_data = None
        self.questions = []
        self.answers = []
        self.scores = []
        self.follow_up_questions = []
        self.current_question_index = 0
        self.greeting_script = ""
        self.simulation_mode = False
        
    def extract_candidate_info(self) -> Dict[str, Any]:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client['aieta']
        candidates_collection = db['candidates']
        self.candidate_data = candidates_collection.find_one()
        return self.candidate_data
        
    def generate_questions(self):
        prompt_template = '''You are an Interviewer bot with 5+ years of experience taking interviews for Data Science and AI domain. 
        Generate a greeting message and 5 questions based on the candidate information.
        
        Format your response as a JSON object with:
        
        {{
            "interview": {{
                "greeting_script": "A warm greeting to the candidate with giving his name and told about some there work and how you impress with that",
                "questions": [
                    "First question about their experience",
                    "Second question about projects",
                    "Third question about problem-solving",
                    "Fourth question about collaboration",
                    "Fifth question about goals"
                ]
            }}
        }}
        
        candidate info:
        {candidate_data}
        '''
        
        prompt = PromptTemplate(template=prompt_template, input_variables=['candidate_data'])
        chain = (prompt | llm | JsonOutputParser())
        response = chain.invoke({'candidate_data':self.candidate_data})
        return response

    def convert_question_to_speech(self, question: str) -> None:
        response = client.audio.speech.create(
            model='playai-tts',
            voice='Fritz-PlayAI',
            input=question,
            response_format='wav'
        )
        response.write_to_file("speech.wav")
        data, samplerate = sf.read("speech.wav")
        sd.play(data, samplerate)
        sd.wait()

    
    def record_until_enter(self,filename="recording.wav", samplerate=44100):
        print("🎙️ Recording... Press ENTER to stop.")
        duration = 600  # Max duration in seconds (10 mins)
        audio = np.empty((int(duration * samplerate), 1), dtype='int16')

        # Start recording
        recording = sd.rec(audio.shape[0], samplerate=samplerate, channels=1, dtype='int16')

        # Wait for user to press Enter in a separate thread
        stop_flag = {"stop": False}

        def wait_for_enter():
            input()
            stop_flag["stop"] = True

        t = threading.Thread(target=wait_for_enter)
        t.start()

        # Polling loop: wait until Enter is pressed
        i = 0
        while not stop_flag["stop"] and i < duration:
            time.sleep(1)
            i += 1

        sd.stop()
        actual_audio = recording[:i * samplerate]
        write(filename, samplerate, actual_audio)
        print(f"✅ Recording saved to '{filename}'")
        return filename
                


    def convert_speech_to_text(self) -> str:
        with open('recording.wav', 'rb') as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model='whisper-large-v3-turbo',
                response_format='verbose_json'
            )
        return transcription.text

    def evaluate_answer(self, question: str, answer: str) -> Tuple[int, str]:
        prompt_template = '''You are an expert interviewer in Data Science and AI.
        Evaluate the candidate's answer on a scale of 1-10.
        
        Question: {question}
        Answer: {answer}
        
        Format:
        {{
            "evaluation": {{
                "score": <score>,
                "feedback": "<feedback>"
            }}
        }}
        '''
        
        prompt = PromptTemplate(template=prompt_template, input_variables=['question', 'answer'])
        response = (prompt | llm | JsonOutputParser()).invoke({'question': question, 'answer': answer})
        return response
    
    def generate_follow_up_question(self, question: str, answer: str) -> str:
        prompt_template = '''Generate a one follow-up question to help the candidate elaborate also only give me the followup question don't add any other text not at start not at end just pure only one followup question.
        
        Original Question: {question}
        Candidate's Answer: {answer}
        
        '''
        
        prompt = PromptTemplate(template=prompt_template, input_variables=['question', 'answer'])
        response = (prompt | llm).invoke({'question': question, 'answer': answer})
        return response.content.strip()
    
    
    def run_interview(self):
        self.extract_candidate_info()
        self.generate_questions()
        
        if not self.simulation_mode:
            self.convert_question_to_speech(self.greeting_script)
        
        for i in range(len(self.questions)):
            current_question = self.questions[i]
            
            if not self.simulation_mode:
                self.convert_question_to_speech(current_question)
                self.record_candidate_answer()
                answer = self.convert_speech_to_text()
            else:
                answer = input("> ")
            
            score, feedback = self.evaluate_answer(current_question, answer)
            self.scores.append(score)
            self.answers.append(answer)
            
            if score < 6:
                follow_up = self.generate_follow_up_question(current_question, answer)
                self.follow_up_questions.append(follow_up)
                
                if not self.simulation_mode:
                    self.convert_question_to_speech(follow_up)
                    self.record_candidate_answer()
                    follow_up_answer = self.convert_speech_to_text()
                else:
                    follow_up_answer = input("> ")
                
                self.evaluate_answer(follow_up, follow_up_answer)
        
        self.summarize_interview()

    def summarize_interview(self):
        average_score = sum(self.scores) / len(self.scores) if self.scores else 0
        
        prompt_template = '''Provide a brief summary of the candidate's performance.
        
        Questions: {questions}
        Answers: {answers}
        Scores: {scores}
        '''
        
        prompt = PromptTemplate(template=prompt_template, input_variables=['questions', 'answers', 'scores'])
        response = (prompt | llm).invoke({
            'questions': json.dumps(self.questions),
            'answers': json.dumps(self.answers),
            'scores': json.dumps(self.scores)
        })
        
        result = {
            "candidate_id": self.candidate_data.get("_id", ""),
            "candidate_name": self.candidate_data.get("name", ""),
            "questions": self.questions,
            "answers": self.answers,
            "scores": self.scores,
            "follow_up_questions": self.follow_up_questions,
            "average_score": average_score,
            "summary": response.content
        }
        
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client['aieta']
        results_collection = db['interview_results']
        results_collection.insert_one(result)



