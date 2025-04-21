from fastapi import APIRouter
from pydantic import BaseModel
import os
import requests

router = APIRouter()

ALPACA_BASE = "https://paper-api.alpaca.markets"
HEADERS = {
    "APCA-API-KEY-ID": os.getenv("ALPACA_KEY", "demo"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET", "demo")
}

class TradeRequest(BaseModel):
    symbol: str
    side: str
    qty: float
    type: str
    limit_price: float = None

@router.post("/place_trade")
def place_trade(req: TradeRequest):
    order = {
        "symbol": req.symbol,
        "qty": req.qty,
        "side": req.side,
        "type": req.type,
        "time_in_force": "gtc"
    }
    if req.limit_price:
        order["limit_price"] = req.limit_price
    r = requests.post(f"{ALPACA_BASE}/v2/orders", json=order, headers=HEADERS)
    return {"status": r.status_code, "order_id": r.json().get("id")}