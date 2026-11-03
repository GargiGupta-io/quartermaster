"""The web service: submit requests, see them, and record approval decisions."""

import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
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

    def submit(self, text: str) -> RequestView:
        request_id = str(uuid.uuid4())
        with self._lock:
            self._db.execute("INSERT INTO requests (id, text) VALUES (?, ?)", (request_id, text))
            self._db.commit()
        response = self.agent.run(text, thread_id=request_id)
        return RequestView(id=request_id, text=text, response=response)

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
        return RequestView(id=request_id, text=self.get(request_id).text, response=response)


def create_app(service: Service | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.service = service
        else:
            settings = Settings()
            app.state.service = Service(settings.data_dir, GroqClient(settings), KnowledgeBase())
        yield

    app = FastAPI(title="Procurement Agent", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.post("/requests", status_code=201)
    def submit(body: NewRequest, request: Request) -> RequestView:
        return request.app.state.service.submit(body.text)

    @app.get("/requests")
    def list_requests(request: Request) -> list[RequestSummary]:
        return request.app.state.service.list()

    @app.get("/requests/{request_id}")
    def get_request(request_id: str, request: Request) -> RequestView:
        found = request.app.state.service.get(request_id)
        if not found:
            raise HTTPException(404, "Request not found")
        return found

    @app.post("/requests/{request_id}/decisions")
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
