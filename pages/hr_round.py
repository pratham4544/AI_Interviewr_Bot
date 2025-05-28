import streamlit as st
from src.helper import get_stored_interview_template, evaluate_answer, generate_follow_up_question, get_candidate_average_score
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from src.prompt import *

load_dotenv()
client = MongoClient(os.environ['MONGO_URI'])
db = client['aieta']
collection = db['interviews']

st.set_page_config(layout="wide")
st.title("🧠 AI Interview Room")

if 'candidate_id' not in st.session_state:
    st.error("⚠️ No candidate ID found. Please go to the homepage to start.")
    st.stop()

candidate_id = st.session_state.candidate_id
greeting, questions = get_stored_interview_template(candidate_id)

# Initialize session state
if "interview_data" not in st.session_state:
    st.session_state.interview_data = {
        "candidate_id": candidate_id,
        "interactions": []
    }
    st.session_state.question_index = 0
    st.session_state.step = "main"

# UI layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 AI Interviewer")
    st.info(greeting)

    if st.session_state.question_index < len(questions):
        current_question = questions[st.session_state.question_index]
        st.markdown(f"**Q{st.session_state.question_index + 1}:** {current_question}")
    else:
        st.success("✅ Interview completed!")
        collection.insert_one(st.session_state.interview_data)
        st.json(st.session_state.interview_data)
        
        avg_score, length_scores, total_scores = get_candidate_average_score(candidate_id)
        if avg_score >= 6:
            st.success(f'Congratualtions! You Score {avg_score} Get Ready for Next Step of Interview From Total Score {length_scores}')
            st.balloons()
        
        st.stop()

with col2:
    st.subheader("🧑 Candidate")

    current_question = questions[st.session_state.question_index]

    # Main question input
    if st.session_state.step == "main":
        answer_text = st.text_area("Type your answer here:", key=f"answer_{st.session_state.question_index}")
        if st.button("Submit Answer"):
            evaluation = evaluate_answer(current_question, answer_text)
            st.session_state.latest_interaction = {
                "question": current_question,
                "answer": answer_text,
                "score": evaluation["evaluation"]["score"],
                "feedback": evaluation["evaluation"]["feedback"]
            }

            st.session_state.evaluation_score = evaluation["evaluation"]["score"]
            if evaluation["evaluation"]["score"] < 6:
                st.session_state.f1 = generate_follow_up_question(current_question, answer_text)
                st.session_state.step = "followup1"
            else:
                st.session_state.interview_data["interactions"].append(st.session_state.latest_interaction)
                st.session_state.question_index += 1
                st.rerun()

    # Follow-up 1
    elif st.session_state.step == "followup1":
        st.markdown(f"**💬 Follow-up 1:** {st.session_state.f1}")
        f1_input = st.text_area("Answer to Follow-up 1:")
        if st.button("Submit Follow-up 1"):
            eval1 = evaluate_answer(st.session_state.f1, f1_input)
            st.session_state.latest_interaction["follow_up_1"] = {
                "question": st.session_state.f1,
                "answer": f1_input,
                "score": eval1["evaluation"]["score"],
                "feedback": eval1["evaluation"]["feedback"]
            }

            if eval1["evaluation"]["score"] < 6:
                st.session_state.f2 = generate_follow_up_question(st.session_state.f1, f1_input)
                st.session_state.step = "followup2"
            else:
                st.session_state.interview_data["interactions"].append(st.session_state.latest_interaction)
                st.session_state.question_index += 1
                st.session_state.step = "main"
                st.rerun()

    # Follow-up 2
    elif st.session_state.step == "followup2":
        st.markdown(f"**💬 Follow-up 2:** {st.session_state.f2}")
        f2_input = st.text_area("Answer to Follow-up 2:")
        if st.button("Submit Follow-up 2"):
            eval2 = evaluate_answer(st.session_state.f2, f2_input)
            st.session_state.latest_interaction["follow_up_2"] = {
                "question": st.session_state.f2,
                "answer": f2_input,
                "score": eval2["evaluation"]["score"],
                "feedback": eval2["evaluation"]["feedback"]
            }

            st.session_state.interview_data["interactions"].append(st.session_state.latest_interaction)
            st.session_state.question_index += 1
            st.session_state.step = "main"
            st.rerun()
