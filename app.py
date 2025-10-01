import os
import uuid
import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends, Response, Cookie
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # ✅ ADD THIS

from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from upstash_redis.asyncio import Redis
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

from openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ─── SETTINGS ────────────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    OPENAI_API_KEY: str
    UPSTASH_REDIS_REST_URL: str
    UPSTASH_REDIS_REST_TOKEN: str
    VECTOR_DB_PATH: str = "./chroma_db"
    TOP_K: int = 5
    SESSION_TIMEOUT_MIN: int = 30
    RATE_LIMIT: str = "60/minute"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
load_dotenv()

# ─── LOGGING ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger("legal-bot")

# ─── LIFESPAN MANAGEMENT ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis
    redis = Redis(
        url=settings.UPSTASH_REDIS_REST_URL,
        token=settings.UPSTASH_REDIS_REST_TOKEN
    )
    logger.info("Upstash Redis connection established")
    yield
    await redis.close()
    logger.info("Upstash Redis connection closed")

# ─── FASTAPI APP ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Irish Legal AI Bot",
    description="RAG‑driven Irish legal assistant",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS - Updated for Hugging Face Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# ─── SECURITY & MODERATION ───────────────────────────────────────────────────────
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

async def moderate_content(text: str) -> bool:
    try:
        resp = openai_client.moderations.create(input=text)
        return not resp.results[0].flagged
    except Exception as e:
        logger.error(f"Moderation error: {e}")
        return False

# ─── SESSION MANAGEMENT ──────────────────────────────────────────────────────────
class SessionData(BaseModel):
    session_id: str
    created_at: datetime
    expires_at: datetime  # New field for fixed expiration time
    last_activity: datetime
    history: list

async def get_session(session_id: str = Cookie(default=None), response: Response = None) -> SessionData:
    if session_id:
        raw = await redis.get(session_id)
        if raw:
            data = SessionData.parse_raw(raw)
            # Check if session has expired
            if datetime.utcnow() > data.expires_at:
                await redis.delete(session_id)
            else:
                # Update last activity without changing expiration
                data.last_activity = datetime.utcnow()
                # Save without resetting TTL
                remaining_seconds = (data.expires_at - datetime.utcnow()).total_seconds()
                await redis.setex(session_id, int(remaining_seconds), data.json())
                return data
    
    # Create new session with fixed expiration
    new_id = str(uuid.uuid4())
    created_at = datetime.utcnow()
    expires_at = created_at + timedelta(minutes=settings.SESSION_TIMEOUT_MIN)
    data = SessionData(
        session_id=new_id,
        created_at=created_at,
        expires_at=expires_at,
        last_activity=created_at,
        history=[]
    )
    await redis.setex(
        new_id,
        settings.SESSION_TIMEOUT_MIN * 60,
        data.json()
    )
    response.set_cookie(
        key="session_id", 
        value=new_id, 
        httponly=True, 
        secure=True,
        samesite="None",
        path="/"
    )
    return data

# ─── VECTOR & LLM SETUP ─────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
vectordb = Chroma(embedding_function=embeddings, persist_directory=settings.VECTOR_DB_PATH)

LEGAL_PROMPT = PromptTemplate(
    input_variables=["context","question","history"],
    template=(
        "As an Irish legal expert, provide a precise, concise answer using ONLY the context below."
        "\n1. Direct answer (1-2 sentences)\n2. Key legal basis (cite sources)\n3. Practical implications"
        "\n\nContext:\n{context}\n\nHistory:\n{history}\n\nQuestion: {question}\n\nAnswer:")
)

POLISH_PROMPT = PromptTemplate(
    input_variables=["raw_answer","question"],
    template=(
        "Enhance this Irish legal answer with current figures/fines (2024), recent amendments, and practical next steps."
        " Keep response under 150 words.\n\nOriginal:\n{raw_answer}\n\nQuestion: {question}\n\nEnhanced Answer:")
)

legal_chain = LLMChain(
    llm=ChatOpenAI(temperature=0, openai_api_key=settings.OPENAI_API_KEY, model="gpt-4-turbo"), 
    prompt=LEGAL_PROMPT
)

polish_chain = LLMChain(
    llm=ChatOpenAI(temperature=0.3, openai_api_key=settings.OPENAI_API_KEY, model="gpt-4-turbo"), 
    prompt=POLISH_PROMPT
)

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def retrieve_context(query: str):
    docs = vectordb.similarity_search_with_score(query, k=settings.TOP_K)
    snippets = [f"[Source {i+1} | Relevance: {score:.2f}] {doc.page_content.strip()}" for i,(doc,score) in enumerate(docs)]
    sources = [f"Source {i+1}" for i in range(len(docs))]
    return "\n\n".join(snippets), sources

# ─── MODELS ─────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    sources: list

class SessionStatusResponse(BaseModel):
    status: str  # "active", "expired", or "new"
    ttl: int     # seconds until expiration (-2 = expired, -1 = no expiration)
    session_id: str | None
    created_at: datetime | None
    expires_at: datetime | None  # New field
    last_activity: datetime | None
    history_count: int | None

class SessionHistoryResponse(BaseModel):
    history: list
    session_id: str

# ─── ROUTES ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("index.html")

@app.post("/query", response_model=QueryResponse)
@limiter.limit(settings.RATE_LIMIT)
async def handle_query(
    request: Request,    
    req: QueryRequest,
    session: SessionData = Depends(get_session),
    response: Response = None
):
    if not await moderate_content(req.query):
        raise HTTPException(400, "Content policy violation")

    context, sources = retrieve_context(req.query)
    history = session.history[-3:] if session.history else []

    raw = legal_chain.run({"context": context, "question": req.query, "history": history})
    polished = polish_chain.run({"raw_answer": raw, "question": req.query})
    if not await moderate_content(polished):
        polished = "Restricted content."

    # Update session without changing expiration
    session.history.append({"q": req.query, "a": polished, "timestamp": datetime.utcnow().isoformat()})
    if len(session.history) > 5:
        session.history.pop(0)
    
    # Save with original expiration
    remaining_seconds = (session.expires_at - datetime.utcnow()).total_seconds()
    await redis.setex(
        session.session_id,
        int(remaining_seconds),
        session.json()
    )

    return QueryResponse(answer=polished, session_id=session.session_id, sources=sources)

@app.get("/session/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str = Cookie(default=None)):
    if not session_id:
        return SessionStatusResponse(
            status="new",
            ttl=-2,
            session_id=None,
            created_at=None,
            expires_at=None,
            last_activity=None,
            history_count=None
        )
    
    raw = await redis.get(session_id)
    if not raw:
        return SessionStatusResponse(
            status="expired",
            ttl=-2,
            session_id=session_id,
            created_at=None,
            expires_at=None,
            last_activity=None,
            history_count=None
        )
    
    data = SessionData.parse_raw(raw)
    now = datetime.utcnow()
    
    if now > data.expires_at:
        return SessionStatusResponse(
            status="expired",
            ttl=-2,
            session_id=session_id,
            created_at=data.created_at,
            expires_at=data.expires_at,
            last_activity=data.last_activity,
            history_count=len(data.history)
        )
    
    ttl = int((data.expires_at - now).total_seconds())
    return SessionStatusResponse(
        status="active",
        ttl=ttl,
        session_id=session_id,
        created_at=data.created_at,
        expires_at=data.expires_at,
        last_activity=data.last_activity,
        history_count=len(data.history)
    )

@app.get("/session/history", response_model=SessionHistoryResponse)
async def get_session_history(session: SessionData = Depends(get_session)):
    return {
        "history": session.history,
        "session_id": session.session_id
    }

# ─── SERVER LAUNCH ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=4, log_level="info")