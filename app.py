import streamlit as st
import os
import tempfile
from streamlit_pdf_viewer import pdf_viewer
from helper import InterviewAssistant, AudioHandler
from dotenv import load_dotenv
from mongo import MongoDBHandler
import uuid

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # unique interview session


# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="AI Interviewer Bot", layout="wide")
st.title("AI Interviewer Bot")

# Initialize session state if not already done
if 'interview_assistant' not in st.session_state:
    st.session_state.interview_assistant = None
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'audio_handler' not in st.session_state:
    st.session_state.audio_handler = None
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = None
if 'answer_text' not in st.session_state:
    st.session_state.answer_text = None
if 'files_processed' not in st.session_state:
    st.session_state.files_processed = False

# Initialize audio handler
if st.session_state.audio_handler is None:
    st.session_state.audio_handler = AudioHandler(
        groq_api_key=os.getenv('GROQ_API_KEY')
    )

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    duration = st.slider("Recording Duration (seconds)", 10, 120, 30)
    
    # Reset button
    if st.button("Reset Interview"):
        st.session_state.interview_assistant = None
        st.session_state.current_question_index = 0
        st.session_state.questions = []
        st.session_state.evaluation_result = None
        st.session_state.answer_text = None
        st.session_state.files_processed = False
        st.rerun()

# Upload buttons
col1, col2 = st.columns(2)
with col1:
    uploaded_file_jd = st.file_uploader(
        "Upload Your Job Description", type=['pdf'], key="jd")
with col2:
    uploaded_file_resume = st.file_uploader(
        "Upload Your Resume", type=['pdf'], key="resume")

# Display uploaded PDFs using pdf_viewer
col1, col2 = st.columns(2)

with col1:
    if uploaded_file_jd is not None:
        jd_bytes = uploaded_file_jd.read()
        pdf_viewer(jd_bytes, scroll_to_page=0)
        # Reset file position after reading
        uploaded_file_jd.seek(0)

with col2:
    if uploaded_file_resume is not None:
        resume_bytes = uploaded_file_resume.read()
        pdf_viewer(resume_bytes)
        # Reset file position after reading
        uploaded_file_resume.seek(0)

# Process uploaded files
if uploaded_file_resume is not None and uploaded_file_jd is not None and not st.session_state.files_processed:
    with st.spinner("Processing files and initializing interview assistant..."):
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_resume:
            temp_resume.write(uploaded_file_resume.getbuffer())
            resume_path = temp_resume.name
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_jd:
            temp_jd.write(uploaded_file_jd.getbuffer())
            jd_path = temp_jd.name
        
        # Initialize interview assistant
        st.session_state.interview_assistant = InterviewAssistant(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
        
        # Load documents
        st.session_state.interview_assistant.load_documents(
            resume_path=resume_path,
            jd_path=jd_path
        )
        
        # Initialize models
        st.session_state.interview_assistant.initialize_models()
        
        # Create vector stores
        st.session_state.interview_assistant.create_vector_stores()
        
        # Generate questions
        st.session_state.questions = st.session_state.interview_assistant.generate_questions()
        
        st.session_state.files_processed = True
        st.success("Files processed successfully. Interview assistant is ready!")

# Display interview interface if files are processed
if st.session_state.files_processed:
    st.header("Interview Session")
    
    # Create tabs for questions
    question_tabs = st.tabs([f"Question {i+1}" for i in range(len(st.session_state.questions))])
    
    # Display current question
    with question_tabs[st.session_state.current_question_index]:
        current_question = st.session_state.questions[st.session_state.current_question_index]
        
        st.subheader(f"Question {st.session_state.current_question_index + 1}")
        st.write(current_question)
        
        # Generate audio for question
        if st.button("Listen to Question"):
            with st.spinner("Generating audio..."):
                speech_path = st.session_state.interview_assistant.convert_question_to_speech(
                    st.session_state.current_question_index
                )
                st.audio(speech_path, format="audio/wav")
        
        # Record answer
        st.subheader("Your Answer")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            audio_value = st.audio_input("Record your answer", key=f"audio_{st.session_state.current_question_index}")
        
        with col2:
            if st.button("Transcribe & Evaluate", key=f"evaluate_{st.session_state.current_question_index}"):
                if audio_value is not None:
                    with st.spinner("Transcribing and evaluating your answer..."):
                        # Save audio to file
                        audio_file_path = f"answer_{st.session_state.current_question_index}.wav"
                        with open(audio_file_path, "wb") as f:
                            audio_bytes = audio_value.read()
                            f.write(audio_bytes)
                        
                        # Transcribe audio
                        st.session_state.answer_text = st.session_state.audio_handler.speech_to_text(audio_file_path)
                        
                        # Evaluate answer
                        st.session_state.evaluation_result = st.session_state.interview_assistant.evaluate_answer(
                            current_question, 
                            st.session_state.answer_text
                        )
                        
                                    # Save to MongoDB
                        mongo_handler = MongoDBHandler()
                        mongo_handler.save_response(
                            session_id=st.session_state.session_id,
                            question_index=st.session_state.current_question_index,
                            question=current_question,
                            answer_text=st.session_state.answer_text,
                            evaluation=st.session_state.evaluation_result,
                            audio_data=audio_bytes
                            )
        
        # Display transcription and evaluation
        if st.session_state.answer_text:
            st.subheader("Transcription")
            st.write(st.session_state.answer_text)
        
        if st.session_state.evaluation_result:
            st.subheader("Evaluation")
            st.write(st.session_state.evaluation_result)
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.current_question_index > 0:
                if st.button("Previous Question"):
                    st.session_state.current_question_index -= 1
                    st.session_state.answer_text = None
                    st.session_state.evaluation_result = None
                    st.rerun()
        
        with col2:
            if st.session_state.current_question_index < len(st.session_state.questions) - 1:
                if st.button("Next Question"):
                    st.session_state.current_question_index += 1
                    st.session_state.answer_text = None
                    st.session_state.evaluation_result = None
                    st.rerun()

else:
    if uploaded_file_resume is None or uploaded_file_jd is None:
        st.info("Please upload both a job description and resume to start the interview process.")