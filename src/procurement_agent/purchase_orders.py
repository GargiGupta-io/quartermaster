"""Store purchase orders so that repeating a request cannot create a second order."""

import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel


class PurchaseOrder(BaseModel):
    po_number: str
    idempotency_key: str
    sku: str
    vendor_id: str
    requested_quantity: int
    quantity: int
    office: str
    department: str
    total_inr: float
    approved_by: list[str]
    justification: str


class KeyConflict(ValueError):
    """The same key was reused for a different order."""


class PurchaseOrderStore:
    def __init__(self, path: Path | str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS purchase_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                idempotency_key TEXT NOT NULL UNIQUE,
                department TEXT NOT NULL,
                total_inr REAL NOT NULL,
                body TEXT NOT NULL
            )"""
        )
        self._db.commit()

    def find(self, key: str) -> PurchaseOrder | None:
        row = self._db.execute(
            "SELECT body FROM purchase_orders WHERE idempotency_key = ?", (key,)
        ).fetchone()
        return PurchaseOrder(**json.loads(row[0])) if row else None

    def create(self, order: dict) -> tuple[PurchaseOrder, bool]:
        """Save an order. Returns the order and whether it already existed."""
        key = order["idempotency_key"]
        with self._db:
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO purchase_orders"
                " (idempotency_key, department, total_inr, body) VALUES (?, ?, ?, '')",
                (key, order["department"], order["total_inr"]),
            )
            if cursor.rowcount == 1:
                saved = PurchaseOrder(po_number=f"PO-{cursor.lastrowid:05d}", **order)
                self._db.execute(
                    "UPDATE purchase_orders SET body = ? WHERE id = ?",
                    (saved.model_dump_json(), cursor.lastrowid),
                )
                return saved, False

        existing = self.find(key)
        if existing.model_dump(exclude={"po_number"}) != order:
            raise KeyConflict(f"Key '{key}' was already used for a different order.")
        return existing, True

    def committed_inr(self, department: str) -> float:
        row = self._db.execute(
            "SELECT COALESCE(SUM(total_inr), 0) FROM purchase_orders WHERE department = ?",
            (department,),
        ).fetchone()
        return row[0]
