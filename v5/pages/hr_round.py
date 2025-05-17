import streamlit as st
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import threading

import streamlit.components.v1 as components

components.html(
    """
    <video id="video" width="640" height="480" autoplay playsinline></video>
    <br/>
    <button id="startBtn">Start Recording</button>
    <button id="stopBtn" disabled>Stop Recording</button>
    <a id="downloadLink" style="display:none;">Download Recording</a>

    <script>
    let mediaRecorder;
    let recordedBlobs;

    const video = document.getElementById('video');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const downloadLink = document.getElementById('downloadLink');

    // Access front camera
    navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' }, 
        audio: true
    }).then(stream => {
        video.srcObject = stream;
        window.stream = stream;
    });

    startBtn.onclick = () => {
        recordedBlobs = [];
        let options = { mimeType: 'video/webm;codecs=vp9' };

        mediaRecorder = new MediaRecorder(window.stream, options);

        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                recordedBlobs.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedBlobs, { type: 'video/webm' });
            const url = window.URL.createObjectURL(blob);
            downloadLink.href = url;
            downloadLink.download = 'recording.webm';
            downloadLink.style.display = 'inline';
            downloadLink.textContent = '⬇️ Download Recording';
        };

        mediaRecorder.start();
        startBtn.disabled = true;
        stopBtn.disabled = false;
    };

    stopBtn.onclick = () => {
        mediaRecorder.stop();
        startBtn.disabled = false;
        stopBtn.disabled = true;
    };
    </script>
    """,
    height=600
)


components.html(
    """
    <script>
    document.addEventListener('visibilitychange', function () {
        if (document.hidden) {
            alert("You switched tabs! This activity is being monitored.");
            // You can also send a signal back to Python using a websocket or API call.
        }
    });
    </script>
    """,
    height=0,
)

# Inject JavaScript to disable copy, paste, and right-click
components.html(
    """
    <script>
    // Disable right-click
    document.addEventListener('contextmenu', event => event.preventDefault());

    // Disable copy
    document.addEventListener('copy', event => event.preventDefault());

    // Disable paste
    document.addEventListener('paste', event => event.preventDefault());

    // Disable keyboard shortcuts like Ctrl+C, Ctrl+V, Ctrl+X
    document.addEventListener('keydown', function(event) {
        if ((event.ctrlKey || event.metaKey) && ['c', 'v', 'x', 'a'].includes(event.key.toLowerCase())) {
            event.preventDefault();
        }
    });
    </script>
    """,
    height=0,
    scrolling=False
)


# Ensure we have access to session state variables from homepage
if 'interviewer_bot' not in st.session_state:
    st.error("Please start from the homepage")
    st.stop()

# Get references to session state objects
interviewer_bot = st.session_state.interviewer_bot
candidate_data = st.session_state.candidate_data
greet_and_question = st.session_state.greet_and_question

# Initialize interview session state if not exists
if 'hr_questions' not in st.session_state:
    # Get questions from the interviewer bot
    st.session_state.hr_questions = greet_and_question['interview']['questions'][:2]  # Only using first 2 questions

if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0

if 'score' not in st.session_state:
    st.session_state.score = {}

if 'interview_complete' not in st.session_state:
    st.session_state.interview_complete = False
    
if 'recording' not in st.session_state:
    st.session_state.recording = False
    
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
    
if 'follow_up_stage' not in st.session_state:
    st.session_state.follow_up_stage = 0  # 0: main question, 1: follow-up 1, 2: follow-up 2

def record_answer():
    """Start recording the candidate's answer"""
    st.session_state.recording = True
    
    # Create placeholder for recording status
    status_placeholder = st.empty()
    status_placeholder.info("🎙️ Recording... Click 'Stop Recording' when finished")
    
    # Create a button to stop recording
    stop_rec_placeholder = st.empty()
    stop_recording = stop_rec_placeholder.button("⏹️ Stop Recording")
    
    if stop_recording:
        st.session_state.recording = False
        status_placeholder.success("✅ Recording completed!")
        stop_rec_placeholder.empty()
        return
    
    # In a real implementation, we would start the audio recording here
    # For now, use a text area as a simulation
    st.session_state.audio_file = "recording.wav"  # This would be the path to the recorded audio file
    
    # For demo purposes, provide a text area to simulate voice input
    st.session_state.current_answer = st.text_area(
        "Simulate your voice answer here:", 
        key=f"answer_{st.session_state.current_question_idx}_{st.session_state.follow_up_stage}"
    )

def real_record_answer():
    """Actually record audio using sounddevice"""
    filename = "recording.wav"
    samplerate = 44100
    
    # Create a placeholder for status messages
    status = st.empty()
    status.info("🎙️ Recording... Click Stop when finished")
    
    # Flag to track if recording should stop
    stop_flag = {"stop": False}
    
    # Function to be run in a separate thread to record audio
    def record_audio():
        duration = 60  # Max duration in seconds
        audio = np.empty((int(duration * samplerate), 1), dtype='int16')
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
            for i in range(int(duration)):
                if stop_flag["stop"]:
                    break
                # Read one second of audio
                data, overflowed = stream.read(samplerate)
                if i < audio.shape[0] // samplerate:
                    audio[i*samplerate:(i+1)*samplerate] = data.reshape(-1, 1)
            
            # Save only the recorded portion
            actual_audio = audio[:i*samplerate]
            write(filename, samplerate, actual_audio)
            st.session_state.audio_file = filename
    
    # Start recording in a separate thread
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    
    # Create a button to stop recording
    if st.button("⏹️ Stop Recording"):
        stop_flag["stop"] = True
        recording_thread.join()  # Wait for recording thread to finish
        status.success("✅ Recording saved")
        return True
    
    return False

def submit_answer():
    """Process the recorded answer"""
    if 'current_answer' not in st.session_state or not st.session_state.current_answer:
        st.warning("Please record or type your answer first")
        return
        
    st.session_state.recording = False
    current_idx = st.session_state.current_question_idx
    follow_up_stage = st.session_state.follow_up_stage
    
    # Get current question
    if follow_up_stage == 0:
        current_question = st.session_state.hr_questions[current_idx]
    elif follow_up_stage == 1:
        current_question = st.session_state.score[st.session_state.hr_questions[current_idx]]["follow_up_question_1"]
    else:  # follow_up_stage == 2
        current_question = st.session_state.score[st.session_state.hr_questions[current_idx]]["follow_up_question_2"]
    
    with st.spinner("Processing your answer..."):
        # In a real implementation:
        # 1. If audio was recorded, convert to text using interviewer_bot.convert_speech_to_text()
        # 2. Otherwise use the text input directly
        answer_text = st.session_state.current_answer
        
        # If we had a real audio file and real recording:
        # if st.session_state.audio_file:
        #     answer_text = interviewer_bot.convert_speech_to_text()
        
        # Evaluate the answer
        evaluation = interviewer_bot.evaluate_answer(current_question, answer_text)
        
        # Update score dictionary based on follow-up stage
        if follow_up_stage == 0:
            if st.session_state.hr_questions[current_idx] not in st.session_state.score:
                st.session_state.score[st.session_state.hr_questions[current_idx]] = {}
                
            # Store main question results
            st.session_state.score[st.session_state.hr_questions[current_idx]].update({
                "question_speech": None,  # In real app: interviewer_bot.convert_question_to_speech(current_question)
                "answer_speech": st.session_state.audio_file,
                "answer_text": answer_text,
                "evaluation": evaluation
            })
            
            # Check if follow-up is needed
            if evaluation['evaluation']['score'] <= 6:
                # Generate follow-up question
                follow_up_question = interviewer_bot.generate_follow_up_question(current_question, answer_text)
                
                # Store follow-up question
                st.session_state.score[st.session_state.hr_questions[current_idx]].update({
                    "follow_up_question_1": follow_up_question
                })
                
                # Move to follow-up stage
                st.session_state.follow_up_stage = 1
                st.session_state.current_answer = ""  # Clear the answer for the next question
            else:
                # Move to next question
                move_to_next_question()
                
        elif follow_up_stage == 1:
            # Store follow-up 1 results
            st.session_state.score[st.session_state.hr_questions[current_idx]].update({
                "follow_up_question_1_speech": None,  # In real app: interviewer_bot.convert_question_to_speech(...)
                "follow_up_answer_1_speech": st.session_state.audio_file,
                "follow_up_answer_1_text": answer_text,
                "evaluation_follow_up_1": evaluation
            })
            
            # Check if second follow-up is needed
            if evaluation['evaluation']['score'] <= 6:
                # Generate second follow-up question
                follow_up_question_2 = interviewer_bot.generate_follow_up_question(
                    st.session_state.hr_questions[current_idx], 
                    answer_text
                )
                
                # Store second follow-up question
                st.session_state.score[st.session_state.hr_questions[current_idx]].update({
                    "follow_up_question_2": follow_up_question_2
                })
                
                # Move to second follow-up stage
                st.session_state.follow_up_stage = 2
                st.session_state.current_answer = ""  # Clear the answer for the next question
            else:
                # Move to next question
                move_to_next_question()
                
        else:  # follow_up_stage == 2
            # Store follow-up 2 results
            st.session_state.score[st.session_state.hr_questions[current_idx]].update({
                "follow_up_question_2_speech": None,  # In real app: interviewer_bot.convert_question_to_speech(...)
                "follow_up_answer_2_speech": st.session_state.audio_file,
                "follow_up_answer_2_text": answer_text,
                "evaluation_follow_up_2": evaluation
            })
            
            # Move to next question
            move_to_next_question()
        
        # Reset for next question
        st.session_state.current_answer = ""

def move_to_next_question():
    """Move to the next question or complete the interview"""
    st.session_state.follow_up_stage = 0  # Reset follow-up stage
    st.session_state.current_question_idx += 1  # Increment question index
    st.session_state.current_answer = ""  # Clear the answer for the next question
    
    # Check if we've completed all questions
    if st.session_state.current_question_idx >= len(st.session_state.hr_questions):
        st.session_state.interview_complete = True

def play_question_audio():
    """Play the audio for the current question"""
    current_idx = st.session_state.current_question_idx
    follow_up_stage = st.session_state.follow_up_stage
    
    if follow_up_stage == 0:
        current_question = st.session_state.hr_questions[current_idx]
    elif follow_up_stage == 1:
        current_question = st.session_state.score[st.session_state.hr_questions[current_idx]]["follow_up_question_1"]
    else:  # follow_up_stage == 2
        current_question = st.session_state.score[st.session_state.hr_questions[current_idx]]["follow_up_question_2"]
    
    with st.spinner("Converting question to speech..."):
        # This would actually use the API in a real implementation
        interviewer_bot.convert_question_to_speech(current_question)
        st.success("Question played!")

# Main UI
st.title('HR Round - 1')

# Display current question or completion message
if st.session_state.interview_complete:
    st.success("🎉 Interview Complete! Here's your performance summary:")
    
    # Show chat history from score dictionary
    for i, question in enumerate(st.session_state.hr_questions):
        score_data = st.session_state.score[question]
        
        st.subheader(f"Question {i+1}")
        
        # Main question
        with st.expander(f"Q: {question}", expanded=True):
            st.write("Your answer:")
            st.info(score_data["answer_text"])
            
            st.write("Evaluation:")
            st.write(f"Score: {score_data['evaluation']['evaluation']['score']}/10")
            st.write(f"Feedback: {score_data['evaluation']['evaluation']['feedback']}")
            
            # Show follow-ups if they exist
            if "follow_up_question_1" in score_data:
                st.write("---")
                st.write(f"Follow-up Question 1: {score_data['follow_up_question_1']}")
                
                if "follow_up_answer_1_text" in score_data:
                    st.write("Your answer:")
                    st.info(score_data["follow_up_answer_1_text"])
                    
                    st.write("Evaluation:")
                    st.write(f"Score: {score_data['evaluation_follow_up_1']['evaluation']['score']}/10")
                    st.write(f"Feedback: {score_data['evaluation_follow_up_1']['evaluation']['feedback']}")
            
            # Show second follow-up if it exists
            if "follow_up_question_2" in score_data:
                st.write("---")
                st.write(f"Follow-up Question 2: {score_data['follow_up_question_2']}")
                
                if "follow_up_answer_2_text" in score_data:
                    st.write("Your answer:")
                    st.info(score_data["follow_up_answer_2_text"])
                    
                    st.write("Evaluation:")
                    st.write(f"Score: {score_data['evaluation_follow_up_2']['evaluation']['score']}/10")
                    st.write(f"Feedback: {score_data['evaluation_follow_up_2']['evaluation']['feedback']}")
    
    # Calculate average score
    total_score = 0
    count = 0
    
    for question in st.session_state.hr_questions:
        if question in st.session_state.score:
            data = st.session_state.score[question]
            total_score += data['evaluation']['evaluation']['score']
            count += 1
            
            if 'evaluation_follow_up_1' in data:
                total_score += data['evaluation_follow_up_1']['evaluation']['score']
                count += 1
                
            if 'evaluation_follow_up_2' in data:
                total_score += data['evaluation_follow_up_2']['evaluation']['score']
                count += 1
    
    avg_score = total_score / count if count > 0 else 0
    st.metric("Overall HR Round Score", f"{avg_score:.1f}/10")
    
    # Button to go to next round or complete interview
    if st.button("Proceed to Technical Round"):
        st.switch_page("pages/technical_round.py")
        
else:
    # Determine current question based on follow-up stage
    current_idx = st.session_state.current_question_idx
    follow_up_stage = st.session_state.follow_up_stage
    
    if follow_up_stage == 0:
        current_question = st.session_state.hr_questions[current_idx]
        st.subheader(f"Question {current_idx + 1}")
    elif follow_up_stage == 1:
        current_question = st.session_state.score[st.session_state.hr_questions[current_idx]]["follow_up_question_1"]
        st.subheader(f"Follow-up Question")
    else:  # follow_up_stage == 2
        current_question = st.session_state.score[st.session_state.hr_questions[current_idx]]["follow_up_question_2"]
        st.subheader(f"Second Follow-up Question")
    
    # Display current question
    question_container = st.container(border=True)
    with question_container:
        st.write(current_question)
        if st.button("🔊 Listen to Question"):
            play_question_audio()
    
    # Text area for typing answers (alternative to voice)
    st.text_area("Type your answer here:", key="current_answer")
    
    # Interview controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎙️ Record Answer", key="record_btn"):
            record_answer()
    
    with col2:
        if st.button("✅ Submit Answer", key="submit_btn"):
            submit_answer()
    
    # Progress indicator
    progress_text = f"Question {current_idx + 1}/{len(st.session_state.hr_questions)}"
    if follow_up_stage > 0:
        progress_text += f" (Follow-up {follow_up_stage})"
    
    st.progress((current_idx + follow_up_stage/3) / len(st.session_state.hr_questions))
    st.text(progress_text)
    
    # Chat input as an alternative way to submit answers
    prompt = st.chat_input("Quick answer:")
    if prompt:
        st.session_state.current_answer = prompt
        submit_answer()