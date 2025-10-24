
"""TrailerBot Pro — Retrieval Layer (MVP)
- Loads seeded CSVs
- Provides simple lexical/fuzzy search over models
- Computes payload checks, tongue-weight range, parts hints
- Returns JSON-ready dicts suitable for LLM prompting
"""
import json
import difflib
import pandas as pd

class TrailerBotRetrieval:
    def __init__(self, models_csv: str, parts_csv: str):
        self.df_models = pd.read_csv(models_csv)
        self.df_parts = pd.read_csv(parts_csv) if parts_csv else pd.DataFrame()

    # --------- Basic utils
    def _norm(self, x):
        return str(x).strip().lower() if pd.notna(x) else ""

    def _score(self, query, text):
        query = self._norm(query)
        text  = self._norm(text)
        if not query or not text:
            return 0.0
        # simple composite score = substring bonus + fuzzy ratio
        substr = 1.0 if query in text else 0.0
        ratio = difflib.SequenceMatcher(None, query, text).ratio()
        return substr*0.5 + ratio

    # --------- Model search
    def search_models(self, query: str, top_k: int = 5):
        rows = []
        for _, r in self.df_models.iterrows():
            hay = " ".join([str(r.get(c, "")) for c in ["brand","series","model","type","bolt_pattern","tire_size"]])
            s = self._score(query, hay)
            # extra boost if exact brand/model tokens appear
            tokens = query.lower().split()
            boost = sum(1 for t in tokens if t in hay.lower())
            s += 0.05 * boost
            rows.append((s, r.to_dict()))
        rows.sort(key=lambda x: x[0], reverse=True)
        return [rec for score, rec in rows[:top_k] if score > 0.3]

    def get_model_by_id(self, model_id: str):
        m = self.df_models[self.df_models["id"] == model_id]
        return None if m.empty else m.iloc[0].to_dict()

    # --------- Calculations
    def tongue_weight_range(self, loaded_weight_lb: int, model: dict):
        # choose pct band by coupler type/category
        tw_min = float(model.get("tw_loaded_pct_min", 0.10) or 0.10)
        tw_max = float(model.get("tw_loaded_pct_max", 0.15) or 0.15)
        lo = int(round(loaded_weight_lb * tw_min))
        hi = int(round(loaded_weight_lb * tw_max))
        return {"tw_min_lb": lo, "tw_max_lb": hi, "pct_range": [tw_min, tw_max]}

    def payload_ok(self, cargo_weight_lb: int, model: dict):
        gvwr = int(model.get("gvwr_lb", 0) or 0)
        curb = int(model.get("curb_weight_lb", 0) or 0)
        payload = max(gvwr - curb, 0)
        ok = cargo_weight_lb <= payload
        return {"ok": ok, "payload_lb": payload, "gvwr_lb": gvwr, "curb_weight_lb": curb}

    def axle_load_ok(self, cargo_weight_lb: int, model: dict):
        axles = int(model.get("axle_count", 1) or 1)
        axle_rating = int(model.get("axle_rating_lb", 0) or 0)
        # naive: assume tongue carries 12% if bumper, 15% if gooseneck
        coupler = (model.get("coupler_type") or "bumper").lower()
        pct = 0.15 if coupler == "gooseneck" else 0.12
        on_axles = int(round(cargo_weight_lb * (1 - pct)))
        ok = on_axles <= axles * axle_rating
        return {"ok": ok, "estimated_on_axles_lb": on_axles, "axle_capacity_lb": axles*axle_rating, "assumed_tongue_pct": pct}

    # --------- Parts hints
    def parts_hints(self, model: dict):
        hints = []
        bolt = str(model.get("bolt_pattern") or "")
        axle_rating = int(model.get("axle_rating_lb", 0) or 0)
        # brake kits
        if axle_rating <= 3500:
            hints.append("brake.kit.10in")
            hints.append("axle.dexter.3.5k")
        if axle_rating >= 5200:
            hints.append("brake.kit.12in")
        if axle_rating >= 7000:
            hints.append("axle.dexter.7k")
        # hubs by bolt pattern
        if bolt:
            hints.append(f"hub.{bolt}")
        # de-duplicate
        hints = sorted(set(hints))
        # attach friendly info if parts table loaded
        details = []
        for key in hints:
            if self.df_parts.empty:
                details.append({"key": key})
            else:
                m = self.df_parts[self.df_parts["key"] == key]
                if not m.empty:
                    r = m.iloc[0].to_dict()
                    details.append({"key": key, "part_name": r.get("part_name"), "notes": r.get("notes")})
                else:
                    details.append({"key": key})
        return details

    # --------- High-level Q&A helpers
    def answer_specs(self, query: str):
        cands = self.search_models(query, top_k=5)
        return {
            "query": query,
            "candidates": [
                {
                    "id": m["id"],
                    "brand": m["brand"],
                    "model": m["model"],
                    "type": m["type"],
                    "gvwr_lb": int(m["gvwr_lb"]),
                    "curb_weight_lb": int(m["curb_weight_lb"]),
                    "payload_lb": int(m["payload_lb"]),
                    "bolt_pattern": m.get("bolt_pattern"),
                    "tire_size": m.get("tire_size"),
                    "provenance": {"source_url": m.get("source_url"), "source_date": m.get("source_date"), "data_quality": m.get("data_quality")}
                } for m in cands
            ]
        }

    def answer_parts(self, query: str):
        cands = self.search_models(query, top_k=3)
        answers = []
        for m in cands:
            answers.append({
                "id": m["id"],
                "brand": m["brand"],
                "model": m["model"],
                "bolt_pattern": m.get("bolt_pattern"),
                "axle_rating_lb": int(m.get("axle_rating_lb", 0) or 0),
                "recommended_parts": self.parts_hints(m)
            })
        return {"query": query, "matches": answers}

    def answer_tongue_weight(self, model_query: str, loaded_weight_lb: int):
        cands = self.search_models(model_query, top_k=3)
        if not cands:
            return {"query": model_query, "tw_estimate": None, "note": "no model match"}
        m = cands[0]
        tw = self.tongue_weight_range(loaded_weight_lb, m)
        return {
            "query": model_query,
            "match_id": m["id"],
            "loaded_weight_lb": loaded_weight_lb,
            "tw_estimate": tw,
            "assumptions": "TW% from model category; redistribute load to hit mid-range for best stability."
        }
