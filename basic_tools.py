from datetime import datetime

def query_metrics(metric: str, window: str = "24h") -> dict:
    # Demo tool. Replace with a real metrics API in the next milestone.
    demo = {
        "latency": {"p95_ms": 840, "change_pct": 37.4},
        "error_rate": {"percent": 4.8, "change_pct": 2.1},
        "cpu": {"percent": 91.0, "change_pct": 18.0},
    }
    return {
        "metric": metric,
        "window": window,
        "timestamp": datetime.utcnow().isoformat(),
        "data": demo.get(metric, {"message": "Unknown demo metric"})
    }

def calculate_statistics(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }

def create_action_plan(actions: list[str]) -> dict:
    return {
        "status": "draft",
        "requires_human_approval": True,
        "actions": actions,
    }
