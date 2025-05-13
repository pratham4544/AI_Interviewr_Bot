import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import re
from groq import Groq
import sounddevice as sd
from scipy.io.wavfile import write
import json



class InterviewAssistant:
    """
    A class to handle the interview process including document processing,
    question generation, speech synthesis, and answer evaluation.
    """
    
    # Default interview prompt template
    DEFAULT_PROMPT_TEMPLATE = """
    You are an AI interview assistant. Based on the job description and resume provided below, generate 5 technical interview questions.

    Job Description:
    {jd}

    Resume:
    {resume}

    Format each question like this:
    **Question:** [Your question here]
    """
    
    # Default evaluation prompt template
    DEFAULT_EVAL_TEMPLATE = """
    You are an interview evaluator. Please evaluate the candidate's answer to the question below.
    
    Question: {question}
    
    Answer: {answer}
    
    Evaluate the answer on a scale of 1 to 10 based on:
    1. Correctness of technical information
    2. Clarity of explanation
    3. Relevance to the question
    4. Depth of understanding
    
    Provide a brief explanation for your scores.
    """
    
    def __init__(self, groq_api_key=None, google_api_key=None):
        """Initialize the Interview Assistant with API keys."""
        self.groq_api = groq_api_key or os.getenv('GROQ_API_KEY')
        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        self.llm = None
        self.embeddings = None
        self.jd_data = None
        self.resume_data = None
        self.vector_store_jd = None
        self.vector_store_resume = None
        self.questions = []

        
    def load_pdf(self, file_path):
        """Load and parse a PDF file."""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def load_documents(self, resume_path, jd_path):
        """Load both resume and job description PDFs."""
        self.resume_data = self.load_pdf(resume_path)
        self.jd_data = self.load_pdf(jd_path)
        return self.resume_data, self.jd_data
    
    def initialize_models(self, llm_model="llama-3.1-8b-instant", embedding_model="models/gemini-embedding-exp-03-07"):
        """Initialize LLM and embedding models."""
        # Initialize LLM
        self.llm = ChatGroq(
            model=llm_model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=self.groq_api
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=self.google_api_key
        )
        
        return self.llm, self.embeddings
    
    def create_vector_stores(self):
        """Create vector stores for the job description and resume."""
        if not self.jd_data or not self.resume_data or not self.embeddings:
            raise ValueError("Documents and embeddings must be initialized first")
            
        self.vector_store_jd = FAISS.from_documents(self.jd_data, embedding=self.embeddings)
        self.vector_store_resume = FAISS.from_documents(self.resume_data, embedding=self.embeddings)
        
        return self.vector_store_jd, self.vector_store_resume
    
    def generate_questions(self, custom_prompt=None):
        """Generate interview questions based on the job description and resume."""
        if not self.jd_data or not self.resume_data or not self.llm:
            raise ValueError("Documents and LLM must be initialized first")
        
        # Extract text content from the document objects
        jd_text = "\n".join([doc.page_content for doc in self.jd_data])
        resume_text = "\n".join([doc.page_content for doc in self.resume_data])
        
        # Use default or custom prompt template
        prompt_template = custom_prompt or self.DEFAULT_PROMPT_TEMPLATE
        prompt = PromptTemplate(template=prompt_template, input_variables=['jd', 'resume'])
        chain = prompt | self.llm
        
        # Invoke the chain with the actual text content
        response = chain.invoke({"jd": jd_text, "resume": resume_text})
        
        # Extract questions from the response
        extract_questions = response.content
        self.questions = re.findall(r'\*\*Question:\*\* (.*?)(?:\n|$)', extract_questions)
        
        return self.questions
    
    def convert_question_to_speech(self, question_index=0):
        """Convert a question to speech using Groq TTS API."""
        if not self.questions:
            raise ValueError("No questions have been generated yet")
            
        if question_index >= len(self.questions):
            raise IndexError(f"Question index out of range. Only {len(self.questions)} questions available")
        
        client = Groq(api_key=self.groq_api)
        
        speech_file_path = f"question_{question_index}.wav"
        question_text = self.questions[question_index]
        
        response = client.audio.speech.create(
            model="playai-tts",
            voice="Fritz-PlayAI",
            input=question_text,
            response_format="wav"
        )
        
        response.write_to_file(speech_file_path)
        return speech_file_path
    
    def record_answer(self, duration=30, file_name=None):
        """Record audio answer from user."""
        freq = 44100
        
        print(f"Recording your answer for {duration} seconds...")
        # Record audio
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
        sd.wait()
        
        recording_path = file_name or "recording.wav"
        write(recording_path, freq, recording)
        
        print("Recording complete!")
        return recording_path
    
    def transcribe_audio(self, audio_path):
        """Convert recorded audio to text using Groq API."""
        client = Groq(api_key=self.groq_api)
        
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3-turbo",
                prompt="Interview answer transcription",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                language="en",
                temperature=0.0
            )
        
        return transcription.text
    
    def evaluate_answer(self, question, answer, custom_eval_prompt=None):
        """Evaluate the answer to the question."""
        if not self.llm:
            raise ValueError("LLM must be initialized first")
            
        # Use default or custom evaluation prompt
        eval_template = custom_eval_prompt or self.DEFAULT_EVAL_TEMPLATE
        prompt = PromptTemplate(template=eval_template, input_variables=['question', 'answer'])
        chain = prompt | self.llm
        
        response_eval = chain.invoke({"question": question, "answer": answer})
        return response_eval.content
    
    def run_interview_session(self, question_index=0):
        """Run a complete interview session for a specific question."""
        if not self.questions:
            raise ValueError("No questions have been generated yet")
            
        if question_index >= len(self.questions):
            raise IndexError(f"Question index out of range. Only {len(self.questions)} questions available")
        
        # Get the question
        question = self.questions[question_index]
        print(f"\nQuestion {question_index + 1}: {question}")
        
        # Convert to speech
        speech_path = self.convert_question_to_speech(question_index)
        print(f"Question audio saved to {speech_path}")
        
        # Record answer
        recording_path = self.record_answer(file_name=f"answer_{question_index}.wav")
        
        # Transcribe answer
        answer_text = self.transcribe_audio(recording_path)
        print(f"Your answer (transcribed): {answer_text}")
        
        # Evaluate answer
        evaluation = self.evaluate_answer(question, answer_text)
        print("\nEvaluation:")
        print(evaluation)
        
            
        return {
            "question": question,
            "answer_audio": recording_path,
            "answer_text": answer_text,
            "evaluation": evaluation
        }
    
    def run_full_interview(self):
        """Run the full interview process for all questions."""
        if not self.questions:
            raise ValueError("No questions have been generated yet")
        
        results = []
        
        for i, _ in enumerate(self.questions):
            print(f"\n--- Question {i+1}/{len(self.questions)} ---")
            result = self.run_interview_session(i)
            results.append(result)
            
            # Ask if user wants to continue
            if i < len(self.questions) - 1:
                continue_interview = input("\nContinue to next question? (y/n): ")
                if continue_interview.lower() != 'y':
                    break
        
        return results


class InterviewData:
    """
    A class to handle interview data, including loading, processing, and vector storage.
    This is a helper class that can be used independently or within InterviewAssistant.
    """
    
    def __init__(self, embeddings=None):
        """Initialize the Interview Data handler."""
        self.embeddings = embeddings
        self.document_data = {}
        self.vector_stores = {}
    
    def load_document(self, file_path, doc_type):
        """Load a document from a file path and store it with a type identifier."""
        loader = PyPDFLoader(file_path)
        self.document_data[doc_type] = loader.load()
        return self.document_data[doc_type]
    
    def get_document_text(self, doc_type):
        """Get the text content of a document by type."""
        if doc_type not in self.document_data:
            raise ValueError(f"Document type '{doc_type}' not loaded")
            
        return "\n".join([doc.page_content for doc in self.document_data[doc_type]])
    
    def create_vector_store(self, doc_type):
        """Create a vector store for a specific document type."""
        if not self.embeddings:
            raise ValueError("Embeddings model must be initialized first")
            
        if doc_type not in self.document_data:
            raise ValueError(f"Document type '{doc_type}' not loaded")
            
        self.vector_stores[doc_type] = FAISS.from_documents(
            self.document_data[doc_type], 
            embedding=self.embeddings
        )
        
        return self.vector_stores[doc_type]
    
    def query_vector_store(self, doc_type, query, k=5):
        """Query a vector store for similar documents."""
        if doc_type not in self.vector_stores:
            raise ValueError(f"Vector store for '{doc_type}' not created")
            
        return self.vector_stores[doc_type].similarity_search(query, k=k)


class AudioHandler:
    """
    A class to handle audio operations including text-to-speech and speech-to-text.
    This is a helper class that can be used independently or within InterviewAssistant.
    """
    
    def __init__(self, groq_api_key=None):
        """Initialize the Audio Handler with API key."""
        self.groq_api = groq_api_key or os.getenv('GROQ_API_KEY')
        self.client = Groq(api_key=self.groq_api)
    
    def text_to_speech(self, text, output_path=None, voice="Fritz-PlayAI", model="playai-tts"):
        """Convert text to speech using Groq TTS API."""
        output_path = output_path or "speech.wav"
        
        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="wav"
        )
        
        response.write_to_file(output_path)
        return output_path
    
    def record_audio(self, duration=30, output_path=None, sample_rate=44100, channels=2):
        """Record audio from the microphone."""
        output_path = output_path or "recording.wav"
        
        print(f"Recording audio for {duration} seconds...")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
        sd.wait()
        
        write(output_path, sample_rate, recording)
        print("Recording complete!")
        
        return output_path
    
    def speech_to_text(self, audio_path, model="whisper-large-v3-turbo", language="en"):
        """Convert speech to text using Groq API."""
        with open(audio_path, "rb") as file:
            transcription = self.client.audio.transcriptions.create(
                file=file,
                model=model,
                prompt="Speech transcription",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                language=language,
                temperature=0.0
            )
        
        return transcription.text