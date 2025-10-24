import os, json
from fastapi import FastAPI, Header, HTTPException, Query, Path
from fastapi.responses import JSONResponse
from typing import Optional, Dict
from trailerbot_retrieval import TrailerBotRetrieval

# ---- Config ----
MODELS_CSV = os.environ.get("MODELS_CSV", "./trailer_models_seeded.csv")
PARTS_CSV  = os.environ.get("PARTS_CSV", "./trailer_parts_crossref_sample.csv")
DATA_ROOT  = os.environ.get("DATA_ROOT", "./data")  # per-dealer inventory lives here

API_KEYS   = os.environ.get("API_KEYS_JSON", "{}")

try:
    API_KEYS_OBJ = json.loads(API_KEYS)
except Exception:
    API_KEYS_OBJ = {}

app = FastAPI(title="TrailerBot Pro API (Multi-Dealer)", version="0.2.1")

retrieval = TrailerBotRetrieval(models_csv=MODELS_CSV, parts_csv=PARTS_CSV)

# ---- Auth helpers ----
def require_key(x_api_key: Optional[str]) -> str:
    if not API_KEYS_OBJ:
        return ""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key")
    if isinstance(API_KEYS_OBJ, dict):
        if x_api_key not in API_KEYS_OBJ:
            raise HTTPException(status_code=403, detail="Invalid API key")
        return API_KEYS_OBJ[x_api_key]
    if isinstance(API_KEYS_OBJ, list):
        if x_api_key not in API_KEYS_OBJ:
            raise HTTPException(status_code=403, detail="Invalid API key")
        return ""
    raise HTTPException(status_code=403, detail="Invalid API key format")

def resolve_dealer(x_api_key: Optional[str], dealer_code_param: Optional[str]) -> str:
    mapped = require_key(x_api_key)
    if isinstance(API_KEYS_OBJ, dict) and mapped:
        return mapped
    if not dealer_code_param:
        raise HTTPException(status_code=400, detail="dealer_code required")
    return dealer_code_param

def inventory_path_for(dealer_code: str) -> str:
    return os.path.join(DATA_ROOT, dealer_code, "inventory_normalized.json")

def load_inventory_for(dealer_code: str) -> Dict:
    p = inventory_path_for(dealer_code)
    try:
        with open(p, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"items": [], "note": f"No inventory file at {p}"}
    except Exception as e:
        return {"items": [], "error": str(e)}

# ---- Endpoints ----
@app.get("/health")
def health():
    return {"ok": True, "version": "0.2.1"}

@app.get("/v1/answer_specs")
def answer_specs(q: str = Query(..., description="Free-text model query"),
                 x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    return JSONResponse(retrieval.answer_specs(q))

@app.get("/v1/answer_parts")
def answer_parts(q: str = Query(..., description="Free-text model query"),
                 x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    return JSONResponse(retrieval.answer_parts(q))

@app.get("/v1/tw")
def tw(q: str = Query(..., description="Model query"),
       loaded: int = Query(..., description="Loaded trailer weight (lb)"),
       x_api_key: Optional[str] = Header(None)):
    require_key(x_api_key)
    return JSONResponse(retrieval.answer_tongue_weight(q, loaded))

@app.get("/v1/{dealer_code}/inventory")
def inventory_list(dealer_code: str = Path(..., description="Dealer code"),
                   x_api_key: Optional[str] = Header(None)):
    dc = resolve_dealer(x_api_key, dealer_code)
    inv = load_inventory_for(dc).get("items", [])
    return {"dealer_code": dc, "items": inv}

@app.get("/v1/{dealer_code}/inventory/{model_id}")
def inventory_by_model(model_id: str,
                       dealer_code: str = Path(..., description="Dealer code"),
                       x_api_key: Optional[str] = Header(None)):
    dc = resolve_dealer(x_api_key, dealer_code)
    inv = load_inventory_for(dc).get("items", [])
    hits = [item for item in inv if item.get("match_id") == model_id]
    return {"dealer_code": dc, "model_id": model_id, "items": hits}

@app.get("/v1/{dealer_code}/inventory/search")
def inventory_search(q: str = Query(..., description="Brand/Model free-text"),
                     dealer_code: str = Path(..., description="Dealer code"),
                     x_api_key: Optional[str] = Header(None)):
    dc = resolve_dealer(x_api_key, dealer_code)
    candidates = retrieval.answer_specs(q).get("candidates", [])
    ids = [c["id"] for c in candidates]
    inv = load_inventory_for(dc).get("items", [])
    items = [item for item in inv if item.get("match_id") in ids]
    return {"dealer_code": dc, "query": q, "candidate_ids": ids, "items": items}
