import os
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import json
import re
import sys
import time
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Tuple
from pymongo import MongoClient
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize API clients
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Check if required API keys exist
if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY environment variable is not set")
    sys.exit(1)

if not MONGO_URI:
    print("Error: MONGO_URI environment variable is not set")
    sys.exit(1)

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
        self.simulation_mode = False  # Flag for text-only simulation mode
        
    def extract_candidate_info(self) -> Dict[str, Any]:
        """Extract candidate information from MongoDB."""
        print("\n=== STEP: Extract Candidate Information ===")
        try:
            mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # Test the connection
            mongo_client.server_info()
            
            db = mongo_client['aieta']
            candidates_collection = db['candidates']
            candidate_data = candidates_collection.find_one()
            
            # Convert ObjectId to string and handle MongoDB specific types
            if candidate_data:
                if '_id' in candidate_data:
                    candidate_data['_id'] = str(candidate_data['_id'])
                self.candidate_data = candidate_data
                print("Debug - Extracted Candidate Data:", self.candidate_data)
            else:
                print("No candidate data found in MongoDB. Creating default data for testing.")
                # Create default test data if no data exists
                self.candidate_data = {
                    "_id": "test_id",
                    "name": "Test Candidate",
                    "experience": "5 years of experience in data science and machine learning",
                    "skills": "Python, TensorFlow, PyTorch, SQL, Data Analysis"
                }
                print("Debug - Using Default Test Data:", self.candidate_data)
                
            return self.candidate_data
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            print("Creating default test data since MongoDB connection failed.")
            # Create default test data if connection fails
            self.candidate_data = {
                "_id": "test_id",
                "name": "Test Candidate",
                "experience": "5 years of experience in data science and machine learning",
                "skills": "Python, TensorFlow, PyTorch, SQL, Data Analysis"
            }
            print("Debug - Using Default Test Data:", self.candidate_data)
            return self.candidate_data

    def generate_questions(self) -> None:
        """Generate interview questions based on candidate data."""
        print("\n=== STEP: Generate Questions ===")
        
        if not self.candidate_data:
            print("Error: No candidate data available")
            return
        
        print("Debug - Candidate data available:", self.candidate_data)
        
        # Ensure we only pass required fields to the LLM
        candidate_info = {
            "name": self.candidate_data.get("name", "Test Candidate"),
            "experience": self.candidate_data.get("experience", "5 years in data science"),
            "skills": self.candidate_data.get("skills", "Python, Machine Learning, Data Analysis"),
            "job_role": self.candidate_data.get("job_role", "Data Scientist"),
            "education": self.candidate_data.get("education", "Master's in Computer Science")
        }
        
        print("Debug - Prepared candidate info for LLM:", candidate_info)
        
        prompt_template = '''You are an Interviewer bot with 5+ years of experience taking interviews for Data Science and AI domain. 
        Generate a greeting message and 5 questions based on the candidate information.
        
        Format your response as a JSON object with the following structure:
        {{
            "interview": {{
                "greeting_script": "A warm greeting to the candidate",
                "questions": [
                    "First question about their experience with technologies mentioned in their skills",
                    "Second question about a specific project or achievement",
                    "Third question about problem-solving approach",
                    "Fourth question about their experience with team collaboration",
                    "Fifth question about their future goals and aspirations"
                ]
            }}
        }}

        Candidate Information:
        Name: {name}
        Experience: {experience}
        Skills: {skills}
        Job Role: {job_role}
        Education: {education}
        
        Return only the JSON object, nothing else.'''

        try:
            # Create the prompt with the candidate info
            prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=['name', 'experience', 'skills', 'job_role', 'education']
            )
            
            print("\nDebug - Sending request to LLM...")
            response = (prompt | llm).invoke({
                'name': candidate_info['name'],
                'experience': candidate_info['experience'],
                'skills': candidate_info['skills'],
                'job_role': candidate_info['job_role'],
                'education': candidate_info['education']
            })
            
            print("\nDebug - Raw LLM response:", response.content)
            
            # Try to extract JSON from the response
            try:
                # Clean the response to ensure it's valid JSON
                json_str = response.content.strip()
                if '```json' in json_str:
                    json_str = json_str.split('```json')[1].split('```')[0]
                elif '```' in json_str:
                    json_str = json_str.split('```')[1].split('```')[0]
                
                response_dict = json.loads(json_str)
                
                # Extract questions and greeting
                interview_data = response_dict.get("interview", {})
                self.questions = interview_data.get("questions", [])
                self.greeting_script = interview_data.get("greeting_script", "")
                
                print("\nDebug - Parsed Questions:", self.questions)
                print("Debug - Greeting Script:", self.greeting_script)
                
                if not self.questions:
                    print("Warning: No questions were generated. Using default questions.")
                    self.questions = [
                        "Can you tell me about your experience with data science projects?",
                        "What machine learning frameworks are you most comfortable with?",
                        "How do you approach a new data analysis problem?",
                        "Can you describe a challenging project you worked on?",
                        "What are your career goals in the field of data science?"
                    ]
                    
                if not self.greeting_script:
                    self.greeting_script = f"Hello {candidate_info['name']}, thank you for joining us today. Let's get started with the interview."
                
            except json.JSONDecodeError as je:
                print(f"Error decoding JSON response: {je}")
                print("Response content that failed to parse:", response.content)
                raise Exception("Failed to parse LLM response as JSON") from je
                
        except Exception as e:
            print(f"Error generating questions: {e}")
            print("Using default questions due to error.")
            self.questions = [
                "Can you tell me about your experience with data science projects?",
                "What machine learning frameworks are you most comfortable with?",
                "How do you approach a new data analysis problem?",
                "Can you describe a challenging project you worked on?",
                "What are your career goals in the field of data science?"
            ]
            self.greeting_script = "Hello, thank you for joining us today. Let's get started with the interview."

    def convert_question_to_speech(self, question: str) -> None:
        """Convert the current question to speech."""
        print("\n=== STEP: Convert Question to Speech ===")
        
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

    def record_candidate_answer(self) -> None:
        """Record audio input from the candidate with error handling and timeout."""
        print("\n=== STEP: Record Candidate Answer ===")
        try:
            # Audio recording parameters
            freq = 44100  # Sample rate
            duration = 30  # Maximum recording duration in seconds
            
            print(f"Starting recording for {duration} seconds...")
            
            # Record audio
            recording = sd.rec(
                int(duration * freq), 
                samplerate=freq, 
                channels=1,  # Using mono for better compatibility
                dtype='int16'  # 16-bit PCM
            )
            
            # Show recording progress
            with st.spinner(f"Recording... (0/{duration}s)"):
                for i in range(duration):
                    time.sleep(1)
                    st.spinner(f"Recording... ({i+1}/{duration}s)")
            
            # Wait for recording to complete
            sd.wait()
            
            # Save the recording
            write('recording.wav', freq, recording)
            print("Recording saved as 'recording.wav'")
            
            # Verify the file was created and has content
            if not os.path.exists('recording.wav') or os.path.getsize('recording.wav') == 0:
                raise Exception("Recording failed - empty file")
                
        except Exception as e:
            print(f"Error during recording: {str(e)}")
            # Create a silent audio file as fallback
            silent_audio = np.zeros((freq * 2, 1), dtype=np.int16)  # 2 seconds of silence
            write('recording.wav', freq, silent_audio)
            print("Created silent audio file as fallback")

    def convert_speech_to_text(self) -> str:
        """Convert recorded speech to text."""
        print("\n=== STEP: Convert Speech to Text ===")
        
        try:
            with open('recording.wav', 'rb') as file:
                transcription = client.audio.transcriptions.create(
                    file=file,
                    model='whisper-large-v3-turbo',
                    response_format='verbose_json'
                )
            
            return transcription.text
        except Exception as e:
            print(f"Error converting speech to text: {e}")
            return "[Failed to transcribe response]"

    def evaluate_answer(self, question: str, answer: str) -> Tuple[int, str]:
        """Evaluate the candidate's answer and return a score and feedback."""
        print("\n=== STEP: Evaluate Answer ===")
        
        prompt_template = '''You are an expert interviewer in Data Science and AI with 5+ years of experience.
        Evaluate the candidate's answer to the given question. Rate the answer on a scale of 1-10.
        
        Question: {question}
        Answer: {answer}
        
        Format:
        {{
            "evaluation": {{
                "score": <score>,
                "feedback": "<detailed feedback>"
            }}
        }}
        '''
        
        prompt = PromptTemplate(template=prompt_template, input_variables=['question', 'answer'])
        response = (prompt | llm).invoke({'question': question, 'answer': answer})
        
        try:
            response_dict = json.loads(re.search(r'\{.*\}', response.content, re.DOTALL).group())
            score = response_dict["evaluation"]["score"]
            feedback = response_dict["evaluation"]["feedback"]
            return score, feedback
        except Exception as e:
            print(f"Error parsing LLM response for evaluation: {e}")
            print("Raw response:", response.content)
            return 5, "Unable to generate proper feedback"
    
    def generate_follow_up_question(self, question: str, answer: str) -> str:
        """Generate a follow-up question based on the candidate's answer."""
        print("\n=== STEP: Generate Follow-up Question ===")
        
        prompt_template = '''You are an expert interviewer in Data Science and AI with 5+ years of experience.
        The candidate's answer to a previous question was not satisfactory (scored less than 6/10).
        Generate a follow-up question to help the candidate elaborate or clarify their answer.
        
        Original Question: {question}
        Candidate's Answer: {answer}
        
        Follow-up Question:
        '''
        
        prompt = PromptTemplate(template=prompt_template, input_variables=['question', 'answer'])
        response = (prompt | llm).invoke({'question': question, 'answer': answer})
        
        return response.content.strip()
    
    def set_simulation_mode(self, enabled: bool = False):
        """Set the simulation mode for text-only interactions."""
        self.simulation_mode = enabled
        print(f"Simulation mode {'enabled' if enabled else 'disabled'}")
    
    def run_interview(self):
        """Run the interview process."""
        print("\n=== Starting Interview ===")
        
        # Extract candidate info
        self.extract_candidate_info()
        
        # Generate questions
        self.generate_questions()
        
        # Display greeting
        print(f"\nGreeting: {self.greeting_script}")
        if not self.simulation_mode:
            self.convert_question_to_speech(self.greeting_script)
        
        # Run through questions
        i = 0
        while i < len(self.questions):
            current_question = self.questions[i]
            print(f"\nQuestion {i+1}: {current_question}")
            
            # Convert question to speech
            if not self.simulation_mode:
                self.convert_question_to_speech(current_question)
            
            # Get answer (record or simulate)
            if not self.simulation_mode:
                self.record_candidate_answer()
                answer = self.convert_speech_to_text()
            else:
                # In simulation mode, get text input
                print("Simulation mode: Please type your answer:")
                answer = input("> ")
            
            print(f"Answer: {answer}")
            
            # Evaluate answer
            score, feedback = self.evaluate_answer(current_question, answer)
            self.scores.append(score)
            self.answers.append(answer)
            
            print(f"Evaluation Score: {score}/10")
            print(f"Feedback: {feedback}")
            
            # Decide whether to move to next question or ask follow-up
            if score >= 6:
                print("\nGood answer! Moving to next question...")
                i += 1  # Move to next question
            else:
                print("\nAnswer needs improvement. Asking follow-up question...")
                follow_up = self.generate_follow_up_question(current_question, answer)
                self.follow_up_questions.append(follow_up)
                print(f"Follow-up Question: {follow_up}")
                
                # Convert follow-up to speech
                if not self.simulation_mode:
                    self.convert_question_to_speech(follow_up)
                    
                # Get follow-up answer
                if not self.simulation_mode:
                    self.record_candidate_answer()
                    follow_up_answer = self.convert_speech_to_text()
                else:
                    # In simulation mode, get text input
                    print("Simulation mode: Please type your answer to the follow-up:")
                    follow_up_answer = input("> ")
                
                print(f"Follow-up Answer: {follow_up_answer}")
                
                # Re-evaluate with the follow-up answer
                new_score, new_feedback = self.evaluate_answer(follow_up, follow_up_answer)
                print(f"Follow-up Evaluation Score: {new_score}/10")
                print(f"Follow-up Feedback: {new_feedback}")
                
                # Move to next question regardless of follow-up score
                i += 1
        
        print("\n=== Interview Completed ===")
        self.summarize_interview()

    def summarize_interview(self):
        """Summarize the interview results."""
        print("\n=== Interview Summary ===")
        
        average_score = sum(self.scores) / len(self.scores) if self.scores else 0
        print(f"Average Score: {average_score:.1f}/10")
        
        prompt_template = '''You are an expert interviewer in Data Science and AI.
        Based on the interview questions and answers, provide a brief summary of the candidate's performance.
        
        Questions: {questions}
        Answers: {answers}
        Scores: {scores}
        
        Format your response as a brief interview summary including strengths and areas for improvement.
        '''
        
        prompt = PromptTemplate(template=prompt_template, input_variables=['questions', 'answers', 'scores'])
        response = (prompt | llm).invoke({
            'questions': json.dumps(self.questions),
            'answers': json.dumps(self.answers),
            'scores': json.dumps(self.scores)
        })
        
        print("\nInterview Summary:")
        print(response.content)
        
        # Store results in MongoDB
        try:
            mongo_client = MongoClient(MONGO_URI)
            db = mongo_client['aieta']
            results_collection = db['interview_results']
            
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
            
            results_collection.insert_one(result)
            print("\nInterview results saved to database")
            
        except Exception as e:
            print(f"Error saving results to database: {e}")

def main():
    """Main function to run the interview chatbot."""
    interviewer = InterviewerBot()
    
    # Uncomment to enable text-only simulation mode
    # interviewer.set_simulation_mode(True)
    
    interviewer.run_interview()

if __name__ == "__main__":
    main()