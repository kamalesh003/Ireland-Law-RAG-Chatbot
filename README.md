# Ireland-Legal-Assitant
# Irish Legal AI Bot

A **Retrieval-Augmented Generation (RAG)–driven legal assistant** for Irish law, built with **FastAPI**, **LangChain**, **ChromaDB**, **Upstash Redis**, and **OpenAI GPT-4**.  
The bot provides concise, legally grounded answers with references and practical implications.

---

## 🚀 Features
- **RAG-powered Q&A**: Retrieves relevant legal documents using embeddings before generating answers.
- **Contextual Sessions**: Maintains short-term session history with Redis (auto-expiration).
- **Answer Polishing**: Enhances legal responses with up-to-date fines, amendments, and practical steps.
- **Content Moderation**: Uses OpenAI moderation API for safe responses.
- **Rate Limiting**: Prevents abuse with `slowapi` middleware.
- **CORS-ready**: Configured for deployment on Hugging Face Spaces or other frontends.
- **Static File Hosting**: Serves frontend assets via FastAPI.

---

## 🛠️ Tech Stack
- [FastAPI](https://fastapi.tiangolo.com/) – Web framework
- [LangChain](https://www.langchain.com/) – LLM orchestration
- [ChromaDB](https://www.trychroma.com/) – Vector store for legal documents
- [OpenAI GPT-4 Turbo](https://platform.openai.com/) – LLM for reasoning
- [Upstash Redis](https://upstash.com/) – Session store
- [SlowAPI](https://pypi.org/project/slowapi/) – Rate limiting
- [Pydantic](https://docs.pydantic.dev/) – Data validation

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/irish-legal-ai-bot.git
   cd irish-legal-ai-bot
```

2. **Create a virtual environment**
   
 ```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

Set up environment variables
Create a .env file in the project root:

OPENAI_API_KEY=your_openai_api_key
UPSTASH_REDIS_REST_URL=your_upstash_redis_url
UPSTASH_REDIS_REST_TOKEN=your_upstash_redis_token
VECTOR_DB_PATH=./chroma_db
SESSION_TIMEOUT_MIN=30
RATE_LIMIT=60/minute

## ▶️ Running the App

**Start the server with:**

uvicorn app:app --host 0.0.0.0 --port 7860 --reload


Navigate to:

**API Root: http://localhost:7860**

**Docs: http://localhost:7860/docs**

**📡 API Endpoints**
POST /query

Submit a legal question.
# Request Body:

{ "query": "What is the penalty for late tax filing in Ireland?" }


# Response:

{
  "answer": "Concise legal answer...",
  "session_id": "uuid",
  "sources": ["Source 1", "Source 2"]
}

GET /session/status

Check the current session status.

GET /session/history

Retrieve the session’s Q&A history.

## 🧩 Project Structure
app.py             # Main FastAPI app
static/            # Static frontend assets
index.html         # Default homepage
chroma_db/         # Vector store (persisted) (not committed)
.env               # Environment variables (not committed)
