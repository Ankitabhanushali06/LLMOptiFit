"""Refresh only known model metadata from Nebius's public, unauthenticated catalog.
Does not add models, alter heuristic rankings, use a key, or call inference.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
SOURCE = "https://tokenfactory.nebius.com/api/public/models_info"


def main():
    with urllib.request.urlopen(SOURCE, timeout=30) as response:
        remote = json.load(response)
    active = {f["model_id"]: f for m in remote if m.get("status") == "active" for f in m.get("flavors", [])}
    path = ROOT / "config/nebius_catalog.json"
    local = json.loads(path.read_text())
    missing = [m["model_id"] for m in local if m["model_id"] not in active]
    if missing:
        raise SystemExit("Catalog not changed. Review unavailable model IDs: " + ", ".join(missing))
    for model in local:
        source = active[model["model_id"]]
        for target, key in (("input_price_per_million", "input_price_per_million_tokens"), ("output_price_per_million", "output_price_per_million_tokens"), ("context_window_tokens", "max_model_len"), ("tokens_per_second", "tokens_per_second")):
            value = source[key]
            if type(value) not in (int, float) or value < 0:
                raise SystemExit("Catalog not changed: invalid public metadata.")
            model[target] = value
    contents = json.dumps(local, indent=2) + "\n"
    metadata = {"catalog_version": hashlib.sha256(contents.encode()).hexdigest()[:12],
                "verified_at": datetime.now(timezone.utc).isoformat(), "source": SOURCE,
                "verified_fields": ["model_id", "input_price_per_million", "output_price_per_million", "context_window_tokens", "tokens_per_second"],
                "notes": "Public metadata snapshot. Account availability is not guaranteed; ranking and capability flags are curated heuristics."}
    path.write_text(contents)
    (ROOT / "config/catalog_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Updated {len(local)} models. Review the diff before deploying.")


if __name__ == "__main__": main()
