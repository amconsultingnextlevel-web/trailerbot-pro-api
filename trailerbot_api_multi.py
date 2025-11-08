import os, json
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
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

def ensure_api_key(api_key_header: Optional[str], api_key_query: Optional[str] = None):
    """
    Accept API key via header X-API-Key or query param ?api_key=.
    """
    token = api_key_header or api_key_query
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key (use header X-API-Key or query ?api_key=)")
    if token not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
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
@app.api_route("/v1/answer_specs", methods=["GET", "HEAD"])
def answer_specs(
    request: Request,
    q: str = Query(..., description="Free-text model query (e.g. 'Aluma 8218')"),
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key_q: Optional[str] = Query(None, alias="api_key")
):
    if request.method == "HEAD":
        return Response(status_code=200)
    ensure_api_key(api_key, api_key_q)

    if RETRIEVER:
        try:
            result = RETRIEVER.answer_specs(q)
            return result
        except Exception as e:
            return {"query": q, "candidates": [simple_find_model(q)], "warning": str(e)}
    return {"query": q, "candidates": [simple_find_model(q)], "warning": "retriever unavailable"}
# ================================================================


# ============================== PARTS ============================
@app.api_route("/v1/answer_parts", methods=["GET", "HEAD"])
def answer_parts(
    request: Request,
    q: str = Query(..., description="Parts query (e.g. 'Dexter 7k hub bolt pattern')"),
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key_q: Optional[str] = Query(None, alias="api_key")
):
    if request.method == "HEAD":
        return Response(status_code=200)
    ensure_api_key(api_key, api_key_q)

    if RETRIEVER:
        try:
            result = RETRIEVER.answer_parts(q)
            return result
        except Exception as e:
            return {"query": q, "parts": [], "warning": str(e)}
    return {"query": q, "parts": [], "warning": "retriever unavailable"}
# ================================================================


# =========================== TONGUE WEIGHT =======================
@app.api_route("/v1/tw", methods=["GET", "HEAD"])
def tongue_weight(
    request: Request,
    q: str = Query(..., description="Model query to help estimate TW bounds"),
    loaded: Optional[int] = Query(None, description="Total loaded trailer weight in lb"),
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key_q: Optional[str] = Query(None, alias="api_key")
):
    """
    If RETRIEVER present, try model-specific pct; else use 10–15% heuristic on
    'loaded' or fallback to 10–15% of GVWR (or 5000 if none).
    """
    if request.method == "HEAD":
        return Response(status_code=200)
    ensure_api_key(api_key, api_key_q)

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
@app.api_route("/v1/{dealer_code}/inventory", methods=["GET", "HEAD"])
def get_inventory(
    request: Request,
    dealer_code: str,
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key_q: Optional[str] = Query(None, alias="api_key")
):
    # Allow HEAD precheck
    if request.method == "HEAD":
        return Response(status_code=200)

    ensure_api_key(api_key, api_key_q)
    try:
        items = load_dealer_inventory(dealer_code)
        return {"dealer_code": dealer_code, "items": items}
    except FileNotFoundError:
        return {"dealer_code": dealer_code, "items": [], "error": f"not found under DATA_ROOT for {dealer_code}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inventory load error: {e}")


@app.api_route("/v1/{dealer_code}/inventory/search", methods=["GET", "HEAD"])
def search_inventory(
    request: Request,
    dealer_code: str,
    q: str = Query(..., description="Free-text search across match_id, stock_no, vin, status, price, and all source_row fields"),
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key_q: Optional[str] = Query(None, alias="api_key")
):
    # Allow HEAD precheck
    if request.method == "HEAD":
        return Response(status_code=200)

    ensure_api_key(api_key, api_key_q)
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


# Fast, voice-friendly summary endpoint
@app.api_route("/v1/{dealer_code}/inventory/quick", methods=["GET", "HEAD"])
def quick_inventory(
    request: Request,
    dealer_code: str,
    q: str = Query(...),
    limit: int = Query(1, ge=1, le=5),
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key_q: Optional[str] = Query(None, alias="api_key"),
):
    if request.method == "HEAD":
        return Response(status_code=200)

    ensure_api_key(api_key, api_key_q)

    try:
        items = load_dealer_inventory(dealer_code)
    except FileNotFoundError:
        return {"dealer_code": dealer_code, "query": q, "count": 0, "items": [], "error": "inventory file not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inventory load error: {e}")

    q_norm = q.strip().lower()
    terms = [t for t in q_norm.split() if t]

    out = []
    for it in items:
        hay = []
        for key in ("match_id", "stock_no", "vin", "price", "status"):
            v = it.get(key)
            if v: hay.append(str(v).lower())
        src = it.get("source_row") or {}
        for _, v in src.items():
            if v is not None: hay.append(str(v).lower())

        if all(t in " ".join(hay) for t in terms):
            out.append({
                "model": src.get("Model") or src.get("model") or it.get("match_id"),
                "stock_no": it.get("stock_no"),
                "status": it.get("status"),
                "price": it.get("price"),
                "vin": it.get("vin"),
            })
            if len(out) >= limit:
                break

    return {"dealer_code": dealer_code, "query": q, "count": len(out), "items": out}
# ================================================================


# ============================== ROOT/ECHO ========================
@app.api_route("/", methods=["GET", "HEAD"])
def root(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    return {"ok": True, "service": "trailerbot-pro-api"}

@app.api_route("/v1/ping", methods=["GET", "HEAD"])
def ping(request: Request, api_key: Optional[str] = Query(None, alias="api_key")):
    # allow anonymous HEAD; require api_key on GET
    if request.method == "HEAD":
        return Response(status_code=200)
    if api_key != "sk_demo_wasatch":
        raise HTTPException(status_code=401, detail="Invalid or missing api_key")
    return {"ok": True, "msg": "pong"}

# Voice-friendly, ready-to-say sentence
@app.api_route("/v1/{dealer_code}/inventory/utter", methods=["GET", "HEAD"])
def inventory_utter(
    request: Request,
    dealer_code: str,
    q: str = Query(...),
    limit: int = Query(1, ge=1, le=5),
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key_q: Optional[str] = Query(None, alias="api_key"),
):
    if request.method == "HEAD":
        return Response(status_code=200)
    ensure_api_key(api_key, api_key_q)

    try:
        items = load_dealer_inventory(dealer_code)
    except FileNotFoundError:
        return {"say": f"I do not see any inventory available for {q} right now at dealer {dealer_code}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inventory load error: {e}")

    q_norm = q.strip().lower()
    terms = [t for t in q_norm.split() if t]

    found = []
    for it in items:
        hay = []
        for key in ("match_id", "stock_no", "vin", "price", "status"):
            v = it.get(key)
            if v: hay.append(str(v).lower())
        src = it.get("source_row") or {}
        for _, v in src.items():
            if v is not None: hay.append(str(v).lower())
        if all(t in " ".join(hay) for t in terms):
            found.append(it)
            if len(found) >= limit:
                break

    if not found:
        return {"say": f"I do not see an exact match for {q} in current inventory. Would you like similar options or a special order?"}

    it = found[0]
    src = it.get("source_row") or {}
    model = src.get("Model") or src.get("model") or it.get("match_id") or q
    stock = it.get("stock_no")
    status = it.get("status") or "in stock"

    # Format response naturally
    return format_inventory_response(it, q, model, stock, status)

@app.get("/env")
def echo_env(
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key_q: Optional[str] = Query(None, alias="api_key")
):
    ensure_api_key(api_key, api_key_q)
    return {"API_KEYS_JSON": os.environ.get("API_KEYS_JSON", '{"sk_demo_wasatch":"wasatch"}')}
# ================================================================


# ============================================
# HELPER FUNCTION FOR NATURAL SPEECH
# ============================================

def format_inventory_response(item, query, model, stock_no, status):
    """
    Format inventory item into natural speech for voice agents.
    Handles sold items and formats numbers naturally.
    """
    # Get source data
    src = item.get("source_row") or {}
    
    # Parse model to extract brand and model number
    model_str = str(model).lower()
    
    # Try to extract brand and model from the model string
    # Common formats: "aluma-8218ta-...", "Aluma 8218", etc.
    brand = ""
    model_num = ""
    
    if "-" in model_str:
        parts = model_str.split("-")
        brand = parts[0].capitalize() if len(parts) > 0 else ""
        model_num = parts[1].upper() if len(parts) > 1 else query
    else:
        # Try to split on space
        words = model_str.split()
        if len(words) >= 2:
            brand = words[0].capitalize()
            model_num = words[1]
        else:
            brand = ""
            model_num = query
    
    # Format model number naturally for speech (e.g., "8218" -> "82 18")
    model_natural = format_model_for_speech(model_num)
    
    # Check if sold
    status_lower = str(status).lower()
    if "sold" in status_lower:
        if brand and model_natural:
            return {
                "say": f"I'm sorry, that {brand} {model_natural} has already been sold. "
                       f"Would you like me to check if we have similar models in stock or can order one for you?"
            }
        else:
            return {
                "say": f"I'm sorry, that {query} has already been sold. "
                       f"Would you like me to check similar options?"
            }
    
    # Build natural response for available trailer
    if brand and model_natural:
        response = f"Great news! We have the {brand} {model_natural} in stock. "
    else:
        response = f"Great news! We have the {query} in stock. "
    
    # Add stock number if available
    if stock_no:
        response += f"The stock number is {stock_no}. "
    
    # Try to add some specs from source_row if available
    length = src.get("Length") or src.get("length")
    width = src.get("Width") or src.get("width")
    if length and width:
        response += f"It's {length} by {width} feet. "
    
    gvwr = src.get("GVWR") or src.get("gvwr")
    if gvwr:
        gvwr_natural = format_weight_for_speech(gvwr)
        response += f"The weight rating is {gvwr_natural}. "
    
    price = item.get("price")
    if price:
        price_natural = format_price_for_speech(price)
        response += f"It's priced at {price_natural}. "
    
    response += "Would you like me to text you the full details?"
    
    return {"say": response}


def format_model_for_speech(model):
    """
    Format model numbers for natural speech.
    Examples: "8218" -> "82 18", "7210" -> "72 10"
    """
    if not model:
        return ""
    
    # Extract just digits
    digits = ''.join(c for c in str(model) if c.isdigit())
    
    # If it's a 4-digit model, split it
    if len(digits) == 4:
        return f"{digits[:2]} {digits[2:]}"
    
    return str(model)


def format_weight_for_speech(weight):
    """
    Format weights naturally.
    Examples: "7000" -> "seven thousand pounds", "3500" -> "thirty-five hundred pounds"
    """
    try:
        # Convert to int, handling strings like "7000lb" or "7000#"
        weight_str = str(weight).lower().replace("lb", "").replace("#", "").replace(",", "").strip()
        weight_int = int(float(weight_str))
        
        if weight_int >= 10000:
            thousands = weight_int // 1000
            return f"about {thousands} thousand pounds"
        elif weight_int >= 1000:
            thousands = weight_int // 1000
            hundreds = (weight_int % 1000) // 100
            if hundreds == 0:
                return f"{thousands} thousand pounds"
            elif hundreds == 5:
                return f"{thousands} thousand five hundred pounds"
            else:
                return f"about {thousands} thousand {hundreds} hundred pounds"
        else:
            return f"{weight_int} pounds"
    except:
        return str(weight)


def format_price_for_speech(price):
    """
    Format prices naturally.
    Examples: "$12,995" -> "around thirteen thousand dollars"
    """
    try:
        # Convert to int, handling strings like "$12,995"
        price_str = str(price).replace("$", "").replace(",", "").strip()
        price_int = int(float(price_str))
        
        if price_int >= 1000:
            thousands = price_int // 1000
            hundreds = (price_int % 1000) // 100
            
            if hundreds == 0:
                return f"{thousands} thousand dollars"
            elif hundreds >= 5:
                return f"around {thousands + 1} thousand dollars"
            else:
                return f"around {thousands} thousand {hundreds} hundred dollars"
        else:
            return f"{price_int} dollars"
    except:
        return str(price)
