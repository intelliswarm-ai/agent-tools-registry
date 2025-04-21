import sys
import ssl  # Ensure ssl module is explicitly imported if environment requires it

# Patch for environments missing SSL (some sandboxed ones)
if not hasattr(ssl, 'SSLContext'):
    print("SSL module is missing or improperly configured. HTTPS requests may fail.")
    sys.exit(1)

from fastapi import FastAPI
from tools.place_trade.run import router as trade_router
from tools_registry import router as registry_router

app = FastAPI(title="AI Tool Registry")

# Register routers
app.include_router(registry_router)
app.include_router(trade_router)
