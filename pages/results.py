import os
import streamlit as st
from pymongo import MongoClient
from src.helper import get_candidate_average_score

# --------------------------
# MongoDB Connection
# --------------------------
MONGO_URI = os.environ.get('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client['aieta']
candidates_collection = db['candidates']
interviews_collection = db['interviews']

# --------------------------
# Streamlit UI
# --------------------------
st.title('📊 Candidate Interview Scores')

# Display Available Candidates
st.subheader("Available Candidates:")
for candidate in candidates_collection.find({}, {'_id': 1, 'personal_information.first_name': 1}):
    name = candidate.get('personal_information', {}).get('first_name', 'Unknown')
    st.write(f"🆔 {candidate['_id']} - 👤 {name}")

# Input Field
candidate_id = st.text_input('🔍 Enter Candidate ID')

# Handle Submit Button
if st.button('Submit'):
    if candidate_id:
        # Fetch score details
        avg_score, total_questions, total_score = get_candidate_average_score(candidate_id)
        
        st.success(f'✅ **Average Score:** {avg_score}\n\n📌 **Total Questions:** {total_questions}\n\n🏆 **Total Score:** {total_score}')
        
        # Fetch interview document
        document = interviews_collection.find_one({"candidate_id": candidate_id})
        
        if document:
            st.subheader(f"📝 Candidate: {document['candidate_id']}")
            st.markdown("---")

            for idx, interaction in enumerate(document.get("interactions", []), 1):
                st.markdown(f"####{interaction.get('question', '')}")
                st.markdown(f"**Answer:** {interaction.get('answer', '')}")
                st.markdown(f"**Score:** {interaction.get('score', '')}")
                
                if feedback := interaction.get("feedback"):
                    st.markdown("**Feedback:**")
                    for fb in feedback:
                        st.markdown(f"- {fb}")

                # Show follow-up responses if available
                for key in ["follow_up_1", "follow_up_2"]:
                    if key in interaction:
                        follow_up = interaction[key]
                        st.markdown(f"#### {key.replace('_', ' ').title()}")
                        st.markdown(f"**Question:** {follow_up.get('question', '')}")
                        st.markdown(f"**Answer:** {follow_up.get('answer', '')}")
                        st.markdown(f"**Score:** {follow_up.get('score', '')}")
                        
