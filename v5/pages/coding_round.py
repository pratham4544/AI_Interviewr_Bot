import streamlit as st
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

st.title('Coding Round')

# Embed JDoodle widget using st.components.v1.html
components.html(
    """
    <div data-pym-src='https://www.jdoodle.com/embed/v1/703ecf846e1527ab'></div>
    <script src='https://www.jdoodle.com/assets/jdoodle-pym.min.js' type='text/javascript'></script>
    """,
    height=600,  # Adjust the height as needed
    scrolling=True
)
