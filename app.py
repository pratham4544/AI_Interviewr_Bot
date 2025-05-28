import streamlit as st
import os
from pymongo import MongoClient
from dotenv import load_dotenv

from src.helper import extract_candidate_info, generate_questions, store_interview_template

load_dotenv()
client = MongoClient(os.environ['MONGO_URI'])
db = client['aieta']
collection = db['candidates']

st.set_page_config(layout="wide")
st.title('🧠 Rupadi - AI Interviewer Bot')

# Display basic candidate list
st.subheader("Available Candidates:")
for candidate in collection.find({}, {'_id': 1, 'personal_information.first_name': 1}):
    st.write(f"🆔 {candidate['_id']} - 👤 {candidate['personal_information'].get('first_name', 'Unknown')}")

candidate_id = st.text_input('Enter Candidate ID')

if st.button('Submit'):
    if candidate_id:
        st.session_state.candidate_id = candidate_id
        st.write('Extracting Candidate Info')
        candidate_data = extract_candidate_info(candidate_id)
        st.write('Generating Quesitons')
        response, questions, greeting_script = generate_questions(candidate_data)
        st.write('Storing into the database')
        store_interview_template(candidate_data,greeting_script,questions)
        st.success(f"Candidate ID {candidate_id} stored! Now start the interview.")
    else:
        st.error("Please enter a valid Candidate ID.")

if 'candidate_id' in st.session_state:
    st.write(f"Let us start the interview, {st.session_state.get('candidate_name', 'Candidate')}")

    if st.button('Start Interview'):
        st.switch_page("pages/hr_round.py")
