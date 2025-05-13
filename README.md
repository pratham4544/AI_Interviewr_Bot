````markdown
# 🧠 AI Interviewer Bot

The **AI Interviewer Bot** is a Streamlit-based web application designed to simulate a technical interview experience. Upload your resume and job description, and the bot will generate personalized interview questions, record your audio responses, evaluate them, and store the data for review.

---

## 🚀 Features

- Upload a **Resume** and **Job Description** (PDF)
- Generate **contextual interview questions**
- Use your **microphone to record answers**
- **Speech-to-text transcription** of your answers
- **LLM-based evaluation** of responses
- **Audio playback** of questions
- Store session data in **MongoDB** (text + audio)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **LLMs**: Groq API + Gemini (Google API)
- **Speech Recognition**: Whisper or Groq
- **Text-to-Speech**: ElevenLabs or other
- **Database**: MongoDB (stores audio + text)

---

## 📦 Installation

1. **Clone the repository**:

```bash
git clone https://github.com/your-username/interviewer-bot.git
cd interviewer-bot
````

2. **Create and activate a virtual environment** (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Set up your `.env` file** with API keys:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

5. **Run the app**:

```bash
streamlit run app.py
```

---

## 🗃️ MongoDB Setup

Make sure MongoDB is running locally or provide a remote URI. The audio and evaluation data are stored in the `interviewer_bot.responses` collection.

---

## 📁 Project Structure

```
interviewer-bot/
├── app.py                  # Main Streamlit app
├── helper.py               # Interview logic and audio handling
├── db.py                   # MongoDB handler
├── .env                    # API keys (not included in Git)
├── requirements.txt        # Python dependencies
├── README.md               # Project overview
```

---

## 🔒 Disclaimer

This app stores sensitive data like audio and evaluation results. Use responsibly and consider security if deploying publicly.

---

## 📬 Contact

Created by \[Prathamesh] – feel free to reach out or contribute!

```

