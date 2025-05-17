import streamlit as st
from complex_helper import *

import streamlit.components.v1 as components

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np

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

# Initialize session state for maintaining data across pages
if 'interviewer_bot' not in st.session_state:
    st.session_state.interviewer_bot = InterviewerBot()
    
    # Extract candidate information and store in session state
    st.session_state.candidate_data = st.session_state.interviewer_bot.extract_candidate_info()
    
    # Generate greeting and questions and store in session state
    st.session_state.greet_and_question = st.session_state.interviewer_bot.generate_questions()

# Use the data from session state
interviewer_bot = st.session_state.interviewer_bot
candidate_data = st.session_state.candidate_data
greet_and_question = st.session_state.greet_and_question

st.title('Rupadi - AI Interviewer')

# Display candidate greeting using proper string formatting
st.info(f"Hello, {candidate_data['personal_information']['first_name']}")

# Display greeting script properly (removed the curly braces)
st.warning(greet_and_question['interview']['greeting_script'])

col1, col2 = st.columns(2)

with col1:
    st.image('images/icon.png')

with col2:
    # Create a button and handle navigation properly
    if st.button("Let's start interview ->"):
        st.switch_page("pages/hr_round.py")  # Using correct Streamlit function for page navigation