from fastapi import APIRouter
import os
import requests

router = APIRouter()

ALPACA_BASE = "https://paper-api.alpaca.markets"
HEADERS = {
    "APCA-API-KEY-ID": os.getenv("ALPACA_KEY", "demo"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET", "demo")
}

@router.get("/get_positions")
def get_positions():
    r = requests.get(f"{ALPACA_BASE}/v2/positions", headers=HEADERS)
    positions = r.json()
    parsed = [
        {
            "symbol": pos["symbol"],
            "qty": float(pos["qty"]),
            "avg_entry_price": float(pos["avg_entry_price"]),
            "market_value": float(pos["market_value"])
        }
        for pos in positions
    ]
    return {"positions": parsed} 