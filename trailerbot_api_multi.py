import os, json
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

# Optional: retrieval module (specs/parts). We fall back gracefully if missing.
try:
    from trailerbot_retrieval import TrailerBotRetrieval
except Exception:
    TrailerBotRetrieval = None  # graceful fallback

app = FastAPI(title="TrailerBot Pro API")

# ============================== AUTH ==============================
def load_api_keys() -> Dict[str, str]:
    """
    Expect env var API_KEYS_JSON like: {"sk_demo_wasatch":"wasatch"}
    If missing/invalid, we default to {"sk_demo_wasatch":"wasatch"}.
    """
    raw = os.environ.get("API_KEYS_JSON")
    if not raw:
        return {"sk_demo_wasatch": "wasatch"}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"sk_demo_wasatch": "wasatch"}

API_KEYS = load_api_keys()

def ensure_api_key(api_key: Optional[str]):
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key")
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")
# ================================================================


# ============================= HEALTH ===========================
@app.get("/health")
def health():
    return {"status": "ok"}
# ================================================================


# ============================ RETRIEVAL ==========================
RETRIEVER: Optional[Any] = None
SEED_MODELS_CSV = os.environ.get("SEED_MODELS_CSV", "./trailer_models_seeded.csv")
PARTS_CSV = os.environ.get("PARTS_CSV", "./trailer_parts_crossref_sample.csv")

if TrailerBotRetrieval:
    try:
        RETRIEVER = TrailerBotRetrieval(
            models_csv=SEED_MODELS_CSV,
            parts_csv=PARTS_CSV
        )
    except Exception as e:
        print(f"[warn] TrailerBotRetrieval init failed: {e}")
        RETRIEVER = None
else:
    print("[warn] trailerbot_retrieval.py not found; using simple fallbacks")

def simple_find_model(q: str) -> Dict[str, Any]:
    """Fallback shape when retriever is unavailable."""
    return {
        "brand": None,
        "model": q,
        "type": None,
        "gvwr_lb": None,
        "payload_lb": None,
        "notes": "fallback result (no retriever)"
    }
# ================================================================


# ============================== SPECS ============================
@app.get("/v1/answer_specs")
def answer_specs(
    q: str = Query(..., description="Free-text model query (e.g. 'Aluma 8218')"),
    api_key: str = Header(None, alias="X-API-Key")
):
    ensure_api_key(api_key)

    if RETRIEVER:
        try:
            result = RETRIEVER.answer_specs(q)
            return result
        except Exception as e:
            return {"query": q, "candidates": [simple_find_model(q)], "warning": str(e)}
    return {"query": q, "candidates": [simple_find_model(q)], "warning": "retriever unavailable"}
# ================================================================


# ============================== PARTS ============================
@app.get("/v1/answer_parts")
def answer_parts(
    q: str = Query(..., description="Parts query (e.g. 'Dexter 7k hub bolt pattern')"),
    api_key: str = Header(None, alias="X-API-Key")
):
    ensure_api_key(api_key)
    if RETRIEVER:
        try:
            result = RETRIEVER.answer_parts(q)
            return result
        except Exception as e:
            return {"query": q, "parts": [], "warning": str(e)}
    return {"query": q, "parts": [], "warning": "retriever unavailable"}
# ================================================================


# =========================== TONGUE WEIGHT =======================
@app.get("/v1/tw")
def tongue_weight(
    q: str = Query(..., description="Model query to help estimate TW bounds"),
    loaded: Optional[int] = Query(None, description="Total loaded trailer weight in lb"),
    api_key: str = Header(None, alias="X-API-Key")
):
    """
    If RETRIEVER present, try model-specific pct; else use 10–15% heuristic on
    'loaded' or fallback to 10–15% of GVWR (or 5000 if none).
    """
    ensure_api_key(api_key)

    tw_min_pct, tw_max_pct = 0.10, 0.15
    model_info = None

    if RETRIEVER:
        try:
            spec = RETRIEVER.answer_specs(q)
            cand = (spec.get("candidates") or [None])[0] if isinstance(spec, dict) else None
            model_info = cand or {}
            pct = model_info.get("tw_pct_range") or model_info.get("tongue_weight_pct_range")
            if isinstance(pct, (list, tuple)) and len(pct) == 2:
                tw_min_pct, tw_max_pct = float(pct[0]), float(pct[1])
        except Exception:
            pass

    # decide basis
    basis = None
    if loaded and isinstance(loaded, int):
        basis = loaded
    else:
        gvwr = model_info.get("gvwr_lb") if model_info else None
        basis = int(gvwr) if gvwr else 5000  # conservative fallback

    tw_min = int(round(basis * tw_min_pct))
    tw_max = int(round(basis * tw_max_pct))
    return {
        "query": q,
        "loaded_basis_lb": basis,
        "tw_estimate": {
            "tw_min_lb": tw_min,
            "tw_max_lb": tw_max,
            "pct_range": [tw_min_pct, tw_max_pct]
        }
    }
# ================================================================


# ========================= INVENTORY LOADER ======================
def load_dealer_inventory(dealer_code: str) -> List[Dict[str, Any]]:
    """
    Loads DATA_ROOT/<dealer_code>/inventory_normalized.json and returns the 'items' array.
    """
    data_root = os.environ.get("DATA_ROOT", "./data")
    path = os.path.join(data_root, dealer_code, "inventory_normalized.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    return items if isinstance(items, list) else []
# ================================================================


# ============================== INVENTORY ========================
@app.get("/v1/{dealer_code}/inventory")
def get_inventory(
    dealer_code: str,
    api_key: str = Header(None, alias="X-API-Key")
):
    ensure_api_key(api_key)
    try:
        items = load_dealer_inventory(dealer_code)
        return {"dealer_code": dealer_code, "items": items}
    except FileNotFoundError:
        return {"dealer_code": dealer_code, "items": [], "error": f"not found under DATA_ROOT for {dealer_code}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inventory load error: {e}")


@app.get("/v1/{dealer_code}/inventory/search")
def search_inventory(
    dealer_code: str,
    q: str = Query(..., description="Free-text search across match_id, stock_no, vin, status, price, and all source_row fields"),
    api_key: str = Header(None, alias="X-API-Key")
):
    ensure_api_key(api_key)
    q_norm = (q or "").strip().lower()
    if not q_norm:
        return {"dealer_code": dealer_code, "query": q, "items": []}

    try:
        items = load_dealer_inventory(dealer_code)
    except FileNotFoundError:
        return {"dealer_code": dealer_code, "query": q, "items": [], "error": "inventory file not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inventory load error: {e}")

    terms = [t for t in q_norm.split() if t]
    results = []
    for it in items:
        # Search across common fields + every original CSV field in source_row
        hay = []
        for key in ("match_id", "stock_no", "vin", "price", "status"):
            v = it.get(key)
            if v:
                hay.append(str(v).lower())
        src = it.get("source_row") or {}
        for _, v in src.items():
            if v is not None:
                hay.append(str(v).lower())

        blob = " ".join(hay)
        if all(term in blob for term in terms):
            results.append({
                "match_id": it.get("match_id"),
                "stock_no": it.get("stock_no"),
                "status": it.get("status"),
                "price": it.get("price"),
                "vin": it.get("vin"),
                "source_row": src
            })

    return {"dealer_code": dealer_code, "query": q, "items": results}
# ================================================================


# ============================== DEBUG ============================
# Keep while testing; remove later if you like.
@app.get("/debug_fs")
def debug_fs():
    root = os.environ.get("DATA_ROOT", "./data")
    found = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".json"):
                p = os.path.join(dirpath, f)
                try:
                    sz = os.path.getsize(p)
                except Exception:
                    sz = None
                found.append({"path": p, "size": sz})
    return {"DATA_ROOT": root, "json_files": found}

@app.get("/debug_inventory/{dealer_code}")
def debug_inventory(dealer_code: str):
    root = os.environ.get("DATA_ROOT", "./data")
    path = os.path.join(root, dealer_code, "inventory_normalized.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", [])
        sample = items[0] if items else None
        return {"path": path, "items_count": len(items), "sample": sample}
    except FileNotFoundError:
        return {"path": path, "error": "file_not_found"}
    except Exception as e:
        return {"path": path, "error": str(e)}
# ================================================================


# ============================== ROOT/ECHO ========================
@app.get("/")
def root():
    return {"ok": True, "service": "trailerbot-pro-api"}

@app.get("/env")
def echo_env(api_key: str = Header(None, alias="X-API-Key")):
    ensure_api_key(api_key)
    return {"API_KEYS_JSON": os.environ.get("API_KEYS_JSON", '{"sk_demo_wasatch":"wasatch"}')}
# ================================================================

