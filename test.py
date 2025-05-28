from src.helper import *
import streamlit as st


st.title('Score')

client = MongoClient(os.environ['MONGO_URI'])
db = client['aieta']
collection = db['candidates']
st.subheader("Available Candidates:")
for candidate in collection.find({}, {'_id': 1, 'personal_information.first_name': 1}):
    st.write(f"🆔 {candidate['_id']} - 👤 {candidate['personal_information'].get('first_name', 'Unknown')}")

candidate_id = st.text_input('Enter Candidate ID')

if st.button('Submit'):
    if candidate_id:

        avg_score, length_scores, total_scores = get_candidate_average_score(candidate_id)
        
        st.write(f'Average Score You Get {avg_score}, \n Total Score is {length_scores}, \n Scores You Get {total_scores}')

        collection = db['interviews']
        document = collection.find_one({"candidate_id": candidate_id})
        if document:
            st.subheader("Candidate ID:")
            st.write(document["candidate_id"])

            st.subheader("Interview Interactions:")
            for i, interaction in enumerate(document.get("interactions", []), 1):
                st.markdown(f"**{interaction.get('question', '')}**")
                st.markdown(f"**Answer:** {interaction.get('answer', '')}")
                st.markdown(f"**Score:** {interaction.get('score', '')}")
                st.markdown("**Feedback:**")
                for fb in interaction.get("feedback", []):
                    st.markdown(f"- {fb}")

                # Check for follow-up questions
                for key in ["follow_up_1", "follow_up_2"]:
                    if key in interaction:
                        follow_up = interaction[key]
                        st.markdown(f"**{key.replace('_', ' ').title()}**")
                        st.markdown(f"**Question:** {follow_up.get('question', '')}")
                        st.markdown(f"**Answer:** {follow_up.get('answer', '')}")
                        st.markdown(f"**Score:** {follow_up.get('score', '')}")
                        st.markdown("**Feedback:**")
                        for fb in follow_up.get("feedback", []):
                            st.markdown(f"- {fb}")
        else:
            st.warning("No document found with that Candidate ID.")
    