"""Load and validate the sample company records."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "company"


class Vendor(BaseModel):
    id: str
    name: str
    status: Literal["approved", "probation", "suspended"]
    currency: Literal["INR", "USD"]
    serves: list[str]


class Product(BaseModel):
    sku: str
    name: str


class CatalogEntry(BaseModel):
    """A vendor's price for one product, excluding GST, in the vendor's currency."""

    sku: str
    vendor_id: str
    unit_price: float = Field(gt=0)
    lead_time_days: int = Field(ge=0)


class StockRecord(BaseModel):
    office: str
    sku: str
    on_hand: int = Field(ge=0)
    reserved: int = Field(ge=0)

    @property
    def available(self) -> int:
        return self.on_hand - self.reserved


class Budget(BaseModel):
    department: str
    allocated: float = Field(ge=0)
    spent: float = Field(ge=0)
    committed: float = Field(ge=0)

    @property
    def remaining(self) -> float:
        return self.allocated - self.spent - self.committed


class CompanyData(BaseModel):
    vendors: list[Vendor]
    products: list[Product]
    catalog: list[CatalogEntry]
    inventory: list[StockRecord]
    budgets: list[Budget]


def _read[T](model: type[T], filename: str, data_dir: Path) -> list[T]:
    raw = json.loads((data_dir / filename).read_text(encoding="utf-8"))
    return TypeAdapter(list[model]).validate_python(raw)


def load_company_data(data_dir: Path = DATA_DIR) -> CompanyData:
    return CompanyData(
        vendors=_read(Vendor, "vendors.json", data_dir),
        products=_read(Product, "products.json", data_dir),
        catalog=_read(CatalogEntry, "catalog.json", data_dir),
        inventory=_read(StockRecord, "inventory.json", data_dir),
        budgets=_read(Budget, "budgets.json", data_dir),
    )
