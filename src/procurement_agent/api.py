"""The web service: submit requests, see them, and record approval decisions."""

import json
import logging
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from procurement_agent.agent import NotWaitingForApproval, ProcurementAgent
from procurement_agent.config import Settings
from procurement_agent.knowledge import KnowledgeBase
from procurement_agent.llm import GroqClient, LLMClient
from procurement_agent.purchase_orders import PurchaseOrderStore
from procurement_agent.schemas import Decision, ProcurementResponse
from procurement_agent.tools import ToolBox

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("procurement_agent")


def log_event(event: str, **fields) -> None:
    """One JSON line per event, so logs can be searched and counted."""
    log.info(json.dumps({"event": event, **fields}, default=str))


class NewRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000)


class RequestView(BaseModel):
    id: str
    text: str
    response: ProcurementResponse


class RequestSummary(BaseModel):
    id: str
    text: str
    status: str
    pending_approvers: list[str]


class Service:
    """Ties the agent to saved state. One lock keeps two decisions from racing on a request."""

    def __init__(self, data_dir: Path, llm: LLMClient, knowledge: KnowledgeBase):
        data_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(data_dir / "requests.db", check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS requests (id TEXT PRIMARY KEY, text TEXT NOT NULL,"
            " created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        self._db.commit()
        self._lock = threading.Lock()
        checkpoints = sqlite3.connect(data_dir / "checkpoints.db", check_same_thread=False)
        toolbox = ToolBox(orders=PurchaseOrderStore(data_dir / "orders.db"))
        self.agent = ProcurementAgent(
            llm, toolbox, knowledge, checkpointer=SqliteSaver(checkpoints)
        )

        self._knowledge = knowledge
        self._ai_configured = getattr(llm, "configured", True)

    def submit(self, text: str) -> RequestView:
        request_id = str(uuid.uuid4())
        with self._lock:
            self._db.execute("INSERT INTO requests (id, text) VALUES (?, ?)", (request_id, text))
            self._db.commit()
        started = time.perf_counter()
        response = self.agent.run(text, thread_id=request_id)
        # The request text is not logged: it can hold names or other personal details.
        log_event(
            "request_finished",
            request_id=request_id,
            status=response.status,
            seconds=round(time.perf_counter() - started, 2),
            tokens=response.tokens_used,
            tools=response.tools_used,
            checks_passed=response.verification.passed if response.verification else None,
        )
        return RequestView(id=request_id, text=text, response=response)

    def health(self) -> dict:
        try:
            self._db.execute("SELECT 1").fetchone()
            database = True
        except sqlite3.Error:
            database = False
        documents = self._knowledge.size
        return {
            "status": "ok" if database and documents and self._ai_configured else "unavailable",
            "database": database,
            "documents_indexed": documents,
            "ai_key_configured": self._ai_configured,
        }

    def get(self, request_id: str) -> RequestView | None:
        row = self._db.execute("SELECT text FROM requests WHERE id = ?", (request_id,)).fetchone()
        response = self.agent.get(request_id) if row else None
        return RequestView(id=request_id, text=row[0], response=response) if response else None

    def list(self) -> list[RequestSummary]:
        rows = self._db.execute(
            "SELECT id, text FROM requests ORDER BY created_at DESC, rowid DESC LIMIT 50"
        ).fetchall()
        summaries = []
        for request_id, text in rows:
            response = self.agent.get(request_id)
            if response:
                summaries.append(
                    RequestSummary(
                        id=request_id,
                        text=text,
                        status=response.status,
                        pending_approvers=response.pending_approvers,
                    )
                )
        return summaries

    def decide(self, request_id: str, decision: Decision) -> RequestView:
        with self._lock:
            response = self.agent.decide(request_id, decision)
        log_event(
            "decision",
            request_id=request_id,
            role=decision.role,
            action=decision.action,
            status=response.status,
            error=response.approval_error,
        )
        return RequestView(id=request_id, text=self.get(request_id).text, response=response)


def create_app(service: Service | None = None, api_key: str | None = None) -> FastAPI:
    """api_key=None reads it from the environment; an empty key leaves the API open."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    required_key = settings.api_key if api_key is None else api_key
    if not required_key:
        log_event(
            "warning", detail="API_KEY is not set: the API is open to anyone who can reach it."
        )

    header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def check_key(supplied: str | None = Security(header)) -> None:
        if required_key and not secrets.compare_digest(supplied or "", required_key):
            raise HTTPException(401, "A valid access key is required.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.service = service
        else:
            app.state.service = Service(settings.data_dir, GroqClient(settings), KnowledgeBase())
        yield

    app = FastAPI(
        title="Quartermaster",
        description="AI procurement assistant: purchase requests and questions about company documents, with human "
        "approval. Send the access key in the X-API-Key header.",
        lifespan=lifespan,
    )
    protected = [Depends(check_key)]

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        if request.url.path != "/health":
            log_event(
                "http",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                ms=round((time.perf_counter() - started) * 1000),
            )
        return response

    @app.get("/health", tags=["service"])
    def health(request: Request) -> JSONResponse:
        """Whether the service can take requests. Open to monitors, so it reveals no data."""
        report = request.app.state.service.health()
        return JSONResponse(report, status_code=200 if report["status"] == "ok" else 503)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.post("/requests", status_code=201, dependencies=protected)
    def submit(body: NewRequest, request: Request) -> RequestView:
        return request.app.state.service.submit(body.text)

    @app.get("/requests", dependencies=protected)
    def list_requests(request: Request) -> list[RequestSummary]:
        return request.app.state.service.list()

    @app.get("/requests/{request_id}", dependencies=protected)
    def get_request(request_id: str, request: Request) -> RequestView:
        found = request.app.state.service.get(request_id)
        if not found:
            raise HTTPException(404, "Request not found")
        return found

    @app.post("/requests/{request_id}/decisions", dependencies=protected)
    def decide(request_id: str, decision: Decision, request: Request) -> RequestView:
        service = request.app.state.service
        if not service.get(request_id):
            raise HTTPException(404, "Request not found")
        try:
            return service.decide(request_id, decision)
        except NotWaitingForApproval as error:
            raise HTTPException(409, str(error)) from error

    return app


app = create_app()
