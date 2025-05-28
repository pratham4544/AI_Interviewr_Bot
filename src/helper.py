# importing nessary librarires

import os
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np
import streamlit as st
from pymongo import MongoClient
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from groq import Groq
import threading
import numpy as np
from pymongo import MongoClient
from gtts import gTTS
import whisper
from langchain_core.output_parsers import JsonOutputParser
import time
import subprocess
load_dotenv()
import random
from src.prompt import *


# read api key through .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")


# intialtize the LLM models
client = Groq(api_key=GROQ_API_KEY)
llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0)


# 1. Preprocessing Steps

## 1.1 Extract Information of Candidate

def extract_candidate_info():
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client['aieta']
    candidates_collection = db['candidates']
    
    candidate_data = candidates_collection.find_one()
    
    if candidate_data:
        candidate_info = {
            "candidate_id": candidate_data.get('id'),
            "candidate_name": candidate_data.get('personal_information', {}).get('name'),
            "candidate_email": candidate_data.get('personal_information', {}).get('email')
        }
        return candidate_info
    else:
        return None
    
def extract_candidate_info(candidate_id: str):
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client['aieta']
    candidates_collection = db['candidates']
    
    candidate_data = candidates_collection.find_one({"_id": candidate_id})
    return candidate_data or {}
    
def shuffle_candidate_data(candidate_info):
    number = random.randrange(1,11)
    candidate_data = candidate_info[number]
    return candidate_data

## 1.2 Generate Questions Through LLM

def generate_questions(candidate_data, prompt = genearte_questions_prompt):
    prompt_template = genearte_questions_prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=['candidate_data'])
    chain = (prompt | llm | JsonOutputParser())
    response = chain.invoke({'candidate_data': candidate_data})
    questions = response["interview"]["questions"]
    greeting_script = response["interview"]["greeting_script"]
    return response, questions, greeting_script


## 1.3 Load this question into the MongoDB server

def store_interview_template(candidate_data, greeting, questions):

    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client['aieta']
    template_collection = db['interview_templates']
    template_doc = {
        "candidate_id": candidate_data['id'],
        'candidate_email':candidate_data['personal_information']['email'],
        "greeting_script": greeting,
        "questions": questions
    }
    result = template_collection.insert_one(template_doc)
    return print("✅ Stored interview template with ID:", result.inserted_id)


# 2 Running Interview Process

## 2.1 Grab the greetings, questions from MongoDB

def get_stored_interview_template(candidate_id):

    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client['aieta']
    template_collection = db['interview_templates']
    
    template_doc = template_collection.find_one({"candidate_id": str(candidate_id)})
    if template_doc:
        greeting = template_doc.get("greeting_script", "")
        questions = template_doc.get("questions", [])
        return greeting, questions
    else:
        return None, None

## 2.2 Session for interivew using the text as input answer Streamlit UI

def run_interview(candidate_id = 'JOHDOE-20241013151130'):
    st.set_page_config(layout="wide")
    st.title("🧠 AI Interview Room")

    col1, col2 = st.columns(2)


    greeting, questions  = get_stored_interview_template(candidate_id)
    
    if "interview_data" not in st.session_state:
        st.session_state.interview_data = {
            "candidate_id": candidate_id,
            "interactions": []
        }
        st.session_state.question_index = 0
        st.session_state.current_input = ""

    with col1:
        st.subheader("🤖 AI Interviewer")
        st.info(greeting)
        if st.session_state.question_index < len(questions):
            current_question = questions[st.session_state.question_index]
            st.markdown(f"**Q{st.session_state.question_index + 1}:** {current_question}")

    with col2:
        st.subheader("🧑 Candidate")
        if st.session_state.question_index < len(questions):
            current_question = questions[st.session_state.question_index]
            answer_text = st.text_area("Type your answer here:", key=f"answer_{st.session_state.question_index}")
            if st.button("Submit Answer"):
                try:
                    st.write(f"📝 Your Answer: {answer_text}")
                    evaluation = evaluate_answer(current_question, answer_text)
                    st.write(f"📊 Evaluation Score: {evaluation['evaluation']['score']}")
                    st.write(f"💬 Feedback: {evaluation['evaluation']['feedback']}")

                    interaction = {
                        "question": current_question,
                        "answer": answer_text,
                        "score": evaluation["evaluation"]["score"],
                        "feedback": evaluation["evaluation"]["feedback"]
                    }

                    # Follow-up 1
                    if evaluation["evaluation"]["score"] < 6:
                        f1 = generate_follow_up_question(current_question, answer_text)
                        st.markdown(f"**💬 Follow-up 1:** {f1}")
                        follow_up_1_input = st.text_area("Type your answer to follow-up 1:", key=f"followup1_{st.session_state.question_index}")
                        if follow_up_1_input:
                            eval1 = evaluate_answer(f1, follow_up_1_input)
                            interaction["follow_up_1"] = {
                                "question": f1,
                                "answer": follow_up_1_input,
                                "score": eval1["evaluation"]["score"],
                                "feedback": eval1["evaluation"]["feedback"]
                            }

                            # Follow-up 2
                            if eval1["evaluation"]["score"] < 6:
                                f2 = generate_follow_up_question(f1, follow_up_1_input)
                                st.markdown(f"**💬 Follow-up 2:** {f2}")
                                follow_up_2_input = st.text_area("Type your answer to follow-up 2:", key=f"followup2_{st.session_state.question_index}")
                                if follow_up_2_input:
                                    eval2 = evaluate_answer(f2, follow_up_2_input)
                                    interaction["follow_up_2"] = {
                                        "question": f2,
                                        "answer": follow_up_2_input,
                                        "score": eval2["evaluation"]["score"],
                                        "feedback": eval2["evaluation"]["feedback"]
                                    }

                    st.session_state.interview_data["interactions"].append(interaction)
                    st.session_state.question_index += 1
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    if st.session_state.question_index == len(questions):
        st.success("✅ Interview completed!")
        st.json(st.session_state.interview_data)
        

# 3. Helper Functions to Run soomthly streamlit App

def convert_gtts_text_to_speech(question, filename="speech.mp3"):
    if not isinstance(question, str):
        raise ValueError("Input `question` must be a string.")
    tts = gTTS(text=question, lang='en', slow=False)
    tts.save(filename)
    data, samplerate = sf.read(filename)
    sd.play(data, samplerate)
    sd.wait()
    return filename


def record_until_enter(filename="recording.wav", samplerate=44100):
    duration = 600
    audio = np.empty((int(duration * samplerate), 1), dtype='int16')
    recording = sd.rec(audio.shape[0], samplerate=samplerate, channels=1, dtype='int16')

    stop_flag = {"stop": False}
    def wait_for_enter(): input(); stop_flag["stop"] = True
    t = threading.Thread(target=wait_for_enter)
    t.start()

    i = 0
    while not stop_flag["stop"] and i < duration:
        time.sleep(1)
        i += 1

    sd.stop()
    actual_audio = recording[:i * samplerate]
    write(filename, samplerate, actual_audio)
    
    return filename

def convert_whisper_speech_to_text(filepath= "recording.mp3"):
    model = whisper.load_model('small')
    result = model.transcribe(filepath)
    return result["text"]
    

def evaluate_answer(question, answer, prompt=evaluation_prompt):
    prompt_template = prompt
    
    prompt_obj = PromptTemplate(template=prompt_template, input_variables=['question', 'answer'])
    response = (prompt_obj | llm | JsonOutputParser()).invoke({'question': question, 'answer': answer})
    return response

def generate_follow_up_question(question, answer, prompt = followup_questions_prompt):
    prompt_template = prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=['question', 'answer'])
    response = (prompt | llm).invoke({'question': question, 'answer': answer})
    return response.content

def convert_wav_to_mp3(wav_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    mp3_path = os.path.join(output_dir, os.path.splitext(os.path.basename(wav_path))[0] + ".mp3")
    subprocess.run(["ffmpeg", "-y", "-i", wav_path, mp3_path], check=True)
    return mp3_path


def get_candidate_average_score(candidate_id):
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client['aieta']
    collection = db['interviews']
    
    interview = collection.find_one({'candidate_id': candidate_id})
    if not interview:
        return None
    
    scores = [
        interaction['score'] 
        for interaction in interview.get('interactions', []) 
        if 'score' in interaction
    ]
    
    total_scores = sum(scores)
    length_scores = len(scores)
    avg_score= total_scores/ length_scores
    if scores:
        return  avg_score, length_scores*10, total_scores
    else:
        return None
